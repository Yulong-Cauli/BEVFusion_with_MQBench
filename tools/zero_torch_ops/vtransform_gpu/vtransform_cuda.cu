/*
  GPU zero-copy vtransform kernels for zero-torch BEVFusion.
  No torch/ATen dependencies — pure CUDA C + Thrust.
*/
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>

static inline bool vt_verbose() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = getenv("VT_VERBOSE");
        cached = (env != nullptr && atoi(env) != 0) ? 1 : 0;
    }
    return cached == 1;
}

#define VT_LOG(...)            \
    do {                       \
        if (vt_verbose()) {    \
            printf(__VA_ARGS__); \
        }                      \
    } while (0)

__device__ float atomicExchFloat(float* addr, float val) {
    return __int_as_float(atomicExch((int*)addr, __float_as_int(val)));
}

__device__ inline void matvec3(const float* M, float x, float y, float z,
                               float& ox, float& oy, float& oz) {
    ox = M[0] * x + M[1] * y + M[2] * z;
    oy = M[3] * x + M[4] * y + M[5] * z;
    oz = M[6] * x + M[7] * y + M[8] * z;
}

/* ------------------------------------------------------------------
   1. compute_depth_map_kernel
   ------------------------------------------------------------------ */
__global__ void compute_depth_map_kernel(
    const float* points,          // [total_points, 3]
    const int* points_prefix_sum, // [B+1]
    int total_points,
    const float* inv_lidar_aug_rot,   // [B, 3, 3]
    const float* inv_lidar_aug_trans, // [B, 3]
    const float* lidar2image_rot,     // [B, N, 3, 3]
    const float* lidar2image_trans,   // [B, N, 3]
    const float* img_aug_rot,         // [B, N, 3, 3]
    const float* img_aug_trans,       // [B, N, 3]
    int B, int N, int iH, int iW,
    float* depth_map)                 // [B, N, iH, iW]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_points) return;

    int b = 0;
    for (int i = 1; i <= B; ++i) {
        if (idx < points_prefix_sum[i]) {
            b = i - 1;
            break;
        }
    }

    float x = points[idx * 3 + 0];
    float y = points[idx * 3 + 1];
    float z = points[idx * 3 + 2];

    const float* lr = inv_lidar_aug_rot + b * 9;
    const float* lt = inv_lidar_aug_trans + b * 3;
    float lx = x - lt[0];
    float ly = y - lt[1];
    float lz = z - lt[2];
    float rx, ry, rz;
    matvec3(lr, lx, ly, lz, rx, ry, rz);

    for (int c = 0; c < N; ++c) {
        const float* l2r = lidar2image_rot + (b * N + c) * 9;
        const float* l2t = lidar2image_trans + (b * N + c) * 3;
        float cx, cy, cz;
        matvec3(l2r, rx, ry, rz, cx, cy, cz);
        cx += l2t[0]; cy += l2t[1]; cz += l2t[2];

        float dist = cz;
        cz = fmaxf(1e-5f, fminf(cz, 1e5f));
        cx /= cz; cy /= cz;

        const float* iar = img_aug_rot + (b * N + c) * 9;
        const float* iat = img_aug_trans + (b * N + c) * 3;
        float ix, iy, iz;
        matvec3(iar, cx, cy, cz, ix, iy, iz);
        ix += iat[0]; iy += iat[1]; iz += iat[2];

        // Take first two, swap to (y, x) for numpy-style image indexing
        float py = iy;
        float px = ix;

        if (px >= 0.0f && px < (float)iW && py >= 0.0f && py < (float)iH) {
            int ix_i = (int)px;
            int iy_i = (int)py;
            if (ix_i >= 0 && ix_i < iW && iy_i >= 0 && iy_i < iH) {
                float* addr = depth_map + ((b * N + c) * iH + iy_i) * iW + ix_i;
                atomicExchFloat(addr, dist);
            }
        }
    }
}

/* ------------------------------------------------------------------
   2. get_geometry_kernel
   ------------------------------------------------------------------ */
__global__ void get_geometry_kernel(
    const float* frustum,        // [D, fH, fW, 3]
    const float* inv_post_rots,  // [B, N, 3, 3]
    const float* post_trans,     // [B, N, 3]
    const float* combine_rots,   // [B, N, 3, 3]  (camera2lidar_rots @ inv_intrins)
    const float* camera2lidar_trans, // [B, N, 3]
    const float* extra_rots,     // [B, 3, 3] or NULL
    const float* extra_trans,    // [B, 3] or NULL
    int B, int N, int D, int fH, int fW,
    float* geom)                 // [B, N, D, fH, fW, 3]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D * fH * fW;
    if (idx >= total) return;

    int fw = idx % fW;
    int fh = (idx / fW) % fH;
    int d = (idx / (fW * fH)) % D;
    int n = (idx / (fW * fH * D)) % N;
    int b = idx / (fW * fH * D * N);

    const float* fr = frustum + ((d * fH + fh) * fW + fw) * 3;
    float px = fr[0];
    float py = fr[1];
    float pz = fr[2];

    const float* pt = post_trans + (b * N + n) * 3;
    px -= pt[0]; py -= pt[1]; pz -= pt[2];

    const float* ipr = inv_post_rots + (b * N + n) * 9;
    float tx, ty, tz;
    matvec3(ipr, px, py, pz, tx, ty, tz);

    // points_xy = points[..., :2] * points[..., 2:3]
    px = tx * tz;
    py = ty * tz;
    pz = tz;

    const float* cr = combine_rots + (b * N + n) * 9;
    matvec3(cr, px, py, pz, tx, ty, tz);

    const float* ct = camera2lidar_trans + (b * N + n) * 3;
    tx += ct[0]; ty += ct[1]; tz += ct[2];

    if (extra_rots != nullptr) {
        const float* er = extra_rots + b * 9;
        matvec3(er, tx, ty, tz, px, py, pz);
        tx = px; ty = py; tz = pz;
    }
    if (extra_trans != nullptr) {
        const float* et = extra_trans + b * 3;
        tx += et[0]; ty += et[1]; tz += et[2];
    }

    float* out = geom + (((((b * N + n) * D + d) * fH + fh) * fW + fw) * 3);
    out[0] = tx; out[1] = ty; out[2] = tz;
}

/* ------------------------------------------------------------------
   3. precompute_bev_indices
   ------------------------------------------------------------------ */
__global__ void discretize_and_rank_kernel(
    const float* geom,        // [B, N, D, fH, fW, 3]
    int B, int N, int D, int fH, int fW,
    float voxel_x, float voxel_y, float voxel_z,
    float coors_x_min, float coors_y_min, float coors_z_min,
    int grid_x, int grid_y, int grid_z,
    uint64_t* ranks,          // [Nprime]
    int* geom_feats)          // [Nprime, 4]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nprime = B * N * D * fH * fW;
    if (idx >= Nprime) return;

    const float* g = geom + idx * 3;
    float fx = (g[0] - (coors_x_min - voxel_x / 2.0f)) / voxel_x;
    float fy = (g[1] - (coors_y_min - voxel_y / 2.0f)) / voxel_y;
    float fz = (g[2] - (coors_z_min - voxel_z / 2.0f)) / voxel_z;

    int ix = (int)fx;
    int iy = (int)fy;
    int iz = (int)fz;

    int b = idx / (N * D * fH * fW);

    if (ix >= 0 && ix < grid_x && iy >= 0 && iy < grid_y && iz >= 0 && iz < grid_z) {
        uint64_t rank = (uint64_t)ix * (grid_y * grid_z * B)
                      + (uint64_t)iy * (grid_z * B)
                      + (uint64_t)iz * B
                      + (uint64_t)b;
        ranks[idx] = rank;
        int* gf = geom_feats + idx * 4;
        gf[0] = ix; gf[1] = iy; gf[2] = iz; gf[3] = b;
    } else {
        ranks[idx] = UINT64_MAX;
        int* gf = geom_feats + idx * 4;
        gf[0] = -1; gf[1] = -1; gf[2] = -1; gf[3] = -1;
    }
}

__global__ void gather_geom_feats_kernel(
    const int* geom_feats,      // [Nprime, 4]
    const int* sort_indices,    // [Nprime]
    int K,
    int* geom_feats_out)        // [K, 4]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;
    int src = sort_indices[idx];
    const int* in = geom_feats + src * 4;
    int* out = geom_feats_out + idx * 4;
    out[0] = in[0]; out[1] = in[1]; out[2] = in[2]; out[3] = in[3];
}

__global__ void gather_cam_feats_half_kernel(
    const __half* cam_feats,    // [Nprime, C_bev]
    const int* sort_indices,    // [Nprime]
    int K, int C_bev,
    __half* sorted_cam_feats)   // [K, C_bev]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * C_bev) return;
    int c = idx % C_bev;
    int k = idx / C_bev;
    int src = sort_indices[k];
    sorted_cam_feats[idx] = cam_feats[src * C_bev + c];
}

__global__ void gather_cam_feats_float_kernel(
    const float* cam_feats,     // [Nprime, C_bev]
    const int* sort_indices,    // [Nprime]
    int K, int C_bev,
    float* sorted_cam_feats)    // [K, C_bev]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * C_bev) return;
    int c = idx % C_bev;
    int k = idx / C_bev;
    int src = sort_indices[k];
    sorted_cam_feats[idx] = cam_feats[src * C_bev + c];
}

__global__ void compute_deltas_kernel(
    const uint64_t* ranks,
    int K,
    int* deltas)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;
    if (idx == 0) {
        deltas[idx] = 1;
    } else {
        deltas[idx] = (ranks[idx] != ranks[idx - 1]) ? 1 : 0;
    }
}

__global__ void scatter_interval_starts_kernel(
    int K,
    const int* deltas,
    const int* starts_pos,
    int* interval_starts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;
    if (deltas[idx] == 1) {
        interval_starts[starts_pos[idx]] = idx;
    }
}

__global__ void compute_interval_lengths_kernel(
    const int* interval_starts,
    int M, int K,
    int* interval_lengths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    int start = interval_starts[idx];
    int end = (idx + 1 < M) ? interval_starts[idx + 1] : K;
    interval_lengths[idx] = end - start;
}

/* ------------------------------------------------------------------
   4. transpose_bdwc_to_bcdhw_kernel
   ------------------------------------------------------------------ */
__global__ void transpose_bdwc_to_bcdhw_kernel(
    const float* in,   // [B, D, H, W, C]
    int B, int D, int H, int W, int C,
    float* out)         // [B, C*D, H, W]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int d = (idx / (C * W * H)) % D;
    int b = idx / (C * W * H * D);

    // out[b, c*D + d, h, w]
    int out_idx = (((b * (C * D) + (c * D + d)) * H + h) * W + w);
    out[out_idx] = in[idx];
}

/* ------------------------------------------------------------------
   5. bev_pool kernels (input float16 or float32, output float32 [B,D,H,W,C])
   ------------------------------------------------------------------ */
__global__ void bev_pool_half_bdwc_kernel(
    int b, int d, int h, int w, int n, int c,
    int n_intervals,
    const __half* __restrict__ x,
    const int* __restrict__ geom_feats,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= n_intervals) return;
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    const int* cur_geom_feats = geom_feats + interval_start * 4;
    const __half* cur_x = x + interval_start * c + cur_c;
    float* cur_out = out + cur_geom_feats[3] * d * h * w * c +
                     cur_geom_feats[2] * h * w * c +
                     cur_geom_feats[0] * w * c +
                     cur_geom_feats[1] * c + cur_c;
    float psum = 0.0f;
    for (int i = 0; i < interval_length; i++) {
        psum += __half2float(cur_x[i * c]);
    }
    *cur_out = psum;
}

__global__ void bev_pool_float_bdwc_kernel(
    int b, int d, int h, int w, int n, int c,
    int n_intervals,
    const float* __restrict__ x,
    const int* __restrict__ geom_feats,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= n_intervals) return;
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    const int* cur_geom_feats = geom_feats + interval_start * 4;
    const float* cur_x = x + interval_start * c + cur_c;
    float* cur_out = out + cur_geom_feats[3] * d * h * w * c +
                     cur_geom_feats[2] * h * w * c +
                     cur_geom_feats[0] * w * c +
                     cur_geom_feats[1] * c + cur_c;
    float psum = 0.0f;
    for (int i = 0; i < interval_length; i++) {
        psum += cur_x[i * c];
    }
    *cur_out = psum;
}

/* ------------------------------------------------------------------
   C entry points
   ------------------------------------------------------------------ */
extern "C" int compute_depth_map_cuda(
    const float* points,
    const int* points_prefix_sum,
    int total_points,
    const float* inv_lidar_aug_rot,
    const float* inv_lidar_aug_trans,
    const float* lidar2image_rot,
    const float* lidar2image_trans,
    const float* img_aug_rot,
    const float* img_aug_trans,
    int B, int N, int iH, int iW,
    float* depth_map)
{
    int block = 256;
    int grid = (total_points + block - 1) / block;
    compute_depth_map_kernel<<<grid, block>>>(
        points, points_prefix_sum, total_points,
        inv_lidar_aug_rot, inv_lidar_aug_trans,
        lidar2image_rot, lidar2image_trans,
        img_aug_rot, img_aug_trans,
        B, N, iH, iW, depth_map);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    return 0;
}

extern "C" size_t vtransform_gpu_workspace_size(
    int B, int N, int D, int fH, int fW,
    int grid_x, int grid_y, int grid_z, int C_bev)
{
    int64_t Nprime = (int64_t)B * N * D * fH * fW;
    int64_t geom_size = Nprime * 3 * sizeof(float);
    int64_t bev_pool_out_size = (int64_t)B * grid_z * grid_x * grid_y * C_bev * sizeof(float);
    int64_t ranks_size = Nprime * sizeof(uint64_t);
    int64_t geom_feats_size = Nprime * 4 * sizeof(int);
    int64_t sort_indices_size = Nprime * sizeof(int);
    int64_t sorted_cam_feats_size = Nprime * C_bev * sizeof(float); // enough for float32 or float16
    int64_t deltas_size = Nprime * sizeof(int);
    int64_t starts_pos_size = Nprime * sizeof(int);
    int64_t intervals_size = Nprime * sizeof(int);
    int64_t total = geom_size + bev_pool_out_size + ranks_size + geom_feats_size
                  + sort_indices_size + sorted_cam_feats_size
                  + deltas_size + starts_pos_size + intervals_size * 2;
    return (size_t)total;
}

extern "C" int vtransform_post_depthnet_cuda(
    // geometry inputs
    const float* frustum,
    const float* inv_post_rots,
    const float* post_trans,
    const float* combine_rots,
    const float* camera2lidar_trans,
    const float* extra_rots,
    const float* extra_trans,
    int B, int N, int D, int fH, int fW,
    // bev grid params
    float voxel_x, float voxel_y, float voxel_z,
    float coors_x_min, float coors_y_min, float coors_z_min,
    int grid_x, int grid_y, int grid_z,
    // depthnet output
    const void* cam_feats, int cam_feats_dtype, // 1 = float16
    int C_bev,
    // outputs
    float* camera_bev,  // [B, C_bev*D, grid_x, grid_y]
    int* geom_feats_out,     // [max_K, 4]
    int* interval_starts_out,// [max_M]
    int* interval_lengths_out,// [max_M]
    int* out_K,
    int* out_M,
    // workspace
    void* workspace,
    size_t workspace_size)
{
    int64_t Nprime = (int64_t)B * N * D * fH * fW;

    // Workspace layout
    char* ws = (char*)workspace;
    size_t offset = 0;

    float* geom = (float*)(ws + offset);
    offset += Nprime * 3 * sizeof(float);

    float* bev_pool_out = (float*)(ws + offset);
    offset += (size_t)B * grid_z * grid_x * grid_y * C_bev * sizeof(float);

    uint64_t* ranks = (uint64_t*)(ws + offset);
    offset += Nprime * sizeof(uint64_t);

    int* geom_feats = (int*)(ws + offset);
    offset += Nprime * 4 * sizeof(int);

    int* sort_indices = (int*)(ws + offset);
    offset += Nprime * sizeof(int);

    float* sorted_cam_feats_float = (float*)(ws + offset);
    __half* sorted_cam_feats_half = (__half*)(ws + offset);
    offset += Nprime * C_bev * sizeof(float);

    int* deltas = (int*)(ws + offset);
    offset += Nprime * sizeof(int);

    int* starts_pos = (int*)(ws + offset);
    offset += Nprime * sizeof(int);

    int* interval_starts = (int*)(ws + offset);
    offset += Nprime * sizeof(int);

    int* interval_lengths = (int*)(ws + offset);
    // offset += Nprime * sizeof(int);

    VT_LOG("[vtransform_post_depthnet_cuda] start Nprime=%ld grid=%dx%dx%d C_bev=%d dtype=%d\n",
           (long)Nprime, grid_x, grid_y, grid_z, C_bev, cam_feats_dtype);

    int block = 256;
    int grid_geom = (int)((Nprime + block - 1) / block);

    // 1. get_geometry
    get_geometry_kernel<<<grid_geom, block>>>(
        frustum, inv_post_rots, post_trans,
        combine_rots, camera2lidar_trans,
        extra_rots, extra_trans,
        B, N, D, fH, fW, geom);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERROR after get_geometry: %d\n", (int)err); return (int)err; }
    VT_LOG("[vtransform_post_depthnet_cuda] get_geometry ok\n");

    // 2. discretize and rank
    discretize_and_rank_kernel<<<grid_geom, block>>>(
        geom, B, N, D, fH, fW,
        voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min,
        grid_x, grid_y, grid_z,
        ranks, geom_feats);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERROR after discretize_and_rank: %d\n", (int)err); return (int)err; }
    VT_LOG("[vtransform_post_depthnet_cuda] discretize_and_rank ok\n");

    // 3. init sort_indices and sort by key
    thrust::device_ptr<uint64_t> ranks_ptr(ranks);
    thrust::device_ptr<int> sort_indices_ptr(sort_indices);
    thrust::sequence(thrust::device, sort_indices_ptr, sort_indices_ptr + Nprime);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERROR after thrust::sequence: %d\n", (int)err); return (int)err; }
    thrust::sort_by_key(thrust::device, ranks_ptr, ranks_ptr + Nprime, sort_indices_ptr);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERROR after thrust::sort_by_key: %d\n", (int)err); return (int)err; }

    // 4. find valid prefix K
    int K = (int)(thrust::lower_bound(thrust::device, ranks_ptr, ranks_ptr + Nprime, (uint64_t)UINT64_MAX) - ranks_ptr);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERROR after thrust::lower_bound: %d\n", (int)err); return (int)err; }

    VT_LOG("[vtransform_post_depthnet_cuda] K=%d\n", K);
    int M = 0;
    if (K > 0) {
        // 5. gather geom_feats and cam_feats according to sort_indices
        int grid_k = (K + block - 1) / block;
        gather_geom_feats_kernel<<<grid_k, block>>>(geom_feats, sort_indices, K, geom_feats_out);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after gather_geom_feats: %d\n", (int)err); return (int)err; }

        int grid_cam = (K * C_bev + block - 1) / block;
        if (cam_feats_dtype == 1) {
            const __half* x_half = (const __half*)cam_feats;
            gather_cam_feats_half_kernel<<<grid_cam, block>>>(x_half, sort_indices, K, C_bev, sorted_cam_feats_half);
        } else {
            const float* x_float = (const float*)cam_feats;
            gather_cam_feats_float_kernel<<<grid_cam, block>>>(x_float, sort_indices, K, C_bev, sorted_cam_feats_float);
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after gather_cam_feats: %d\n", (int)err); return (int)err; }
        VT_LOG("[vtransform_post_depthnet_cuda] gather ok\n");

        // 6. compute deltas from sorted ranks
        compute_deltas_kernel<<<grid_k, block>>>(ranks, K, deltas);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after compute_deltas: %d\n", (int)err); return (int)err; }

        // 7. exclusive scan to get start positions and total M
        thrust::device_ptr<int> deltas_ptr(deltas);
        thrust::device_ptr<int> starts_pos_ptr(starts_pos);
        thrust::exclusive_scan(thrust::device, deltas_ptr, deltas_ptr + K, starts_pos_ptr);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after thrust::exclusive_scan: %d\n", (int)err); return (int)err; }
        int last_delta, last_pos;
        cudaMemcpy(&last_delta, deltas + K - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_pos, starts_pos + K - 1, sizeof(int), cudaMemcpyDeviceToHost);
        M = last_pos + last_delta;
        VT_LOG("[vtransform_post_depthnet_cuda] scan ok M=%d\n", M);

        // 8. scatter interval starts
        scatter_interval_starts_kernel<<<grid_k, block>>>(K, deltas, starts_pos, interval_starts);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after scatter_interval_starts: %d\n", (int)err); return (int)err; }

        // 9. compute interval lengths
        int grid_m = (M + block - 1) / block;
        compute_interval_lengths_kernel<<<grid_m, block>>>(interval_starts, M, K, interval_lengths);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after compute_interval_lengths: %d\n", (int)err); return (int)err; }
        VT_LOG("[vtransform_post_depthnet_cuda] intervals ok\n");

        // 10. bev_pool (input float16 or float32 -> output float32 [B,D,H,W,C])
        int total_threads = M * C_bev;
        int grid_pool = (total_threads + block - 1) / block;
        if (cam_feats_dtype == 1) {
            bev_pool_half_bdwc_kernel<<<grid_pool, block>>>(
                B, grid_z, grid_x, grid_y, K, C_bev, M,
                sorted_cam_feats_half,
                geom_feats_out,
                interval_starts,
                interval_lengths,
                bev_pool_out);
        } else {
            bev_pool_float_bdwc_kernel<<<grid_pool, block>>>(
                B, grid_z, grid_x, grid_y, K, C_bev, M,
                sorted_cam_feats_float,
                geom_feats_out,
                interval_starts,
                interval_lengths,
                bev_pool_out);
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after bev_pool: %d\n", (int)err); return (int)err; }
        VT_LOG("[vtransform_post_depthnet_cuda] bev_pool ok\n");

        // 11. transpose [B,D,H,W,C] -> [B,C*D,H,W]
        int total_transpose = B * grid_z * grid_x * grid_y * C_bev;
        int grid_transpose = (total_transpose + block - 1) / block;
        transpose_bdwc_to_bcdhw_kernel<<<grid_transpose, block>>>(
            bev_pool_out, B, grid_z, grid_x, grid_y, C_bev, camera_bev);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after transpose: %d\n", (int)err); return (int)err; }
        VT_LOG("[vtransform_post_depthnet_cuda] transpose ok\n");

        // 12. copy interval_starts, interval_lengths to outputs
        cudaMemcpy(interval_starts_out, interval_starts, M * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(interval_lengths_out, interval_lengths, M * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("ERROR after cudaMemcpy: %d\n", (int)err); return (int)err; }
    }

    *out_K = K;
    *out_M = M;
    VT_LOG("[vtransform_post_depthnet_cuda] done K=%d M=%d\n", K, M);
    return 0;
}
