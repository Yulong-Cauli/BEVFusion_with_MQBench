// BEVPoolV2 CUDA Kernel
// Matching BEVFusion's existing interval-sum kernel in bev_pool_cuda.cu
//
// Input format:
//   - x: [N, C] flattened features after outer product
//   - geom_feats: [N, 4] voxel coordinates (x, y, z, batch)
//   - interval_starts: [M] interval start indices
//   - interval_lengths: [M] interval lengths
//
// Output: [B, D, H, W, C] BEV features (D=Z, H=X, W=Y, matching Python call order)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

/*
  BEV Pooling V2 Kernel (interval-sum style)
  
  This matches BEVFusion's kernel in mmdet3d/ops/bev_pool/src/bev_pool_cuda.cu
  
  Args:
    b                : batch size
    d                : depth of the feature map (Z)
    h                : height of pooled feature map (Y)
    w                : width of pooled feature map (X)
    n_intervals      : number of unique BEV grids to sum
    c                : number of channels
    x                : input features, FloatTensor[N, C]
    geom_feats       : input coordinates, IntTensor[N, 4] (x, y, z, batch)
    interval_starts  : starting position for pooled point, IntTensor[M]
    interval_lengths : how many points in each pooled point, IntTensor[M]
    out              : output features, FloatTensor[B, D, H, W, C]
*/
__global__ void bev_pool_v2_kernel(
    int b, int d, int h, int w, int n_intervals, int c,
    const float* __restrict__ x,
    const int* __restrict__ geom_feats,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ out)
{
    // Each thread handles one interval (one BEV grid cell) for one channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    
    if (index >= n_intervals) return;
    
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    
    // Get geom feats for first point in this interval
    const int* cur_geom_feats = geom_feats + interval_start * 4;
    int bev_x = cur_geom_feats[0];  // X coordinate
    int bev_y = cur_geom_feats[1];  // Y coordinate
    int bev_z = cur_geom_feats[2];  // Z coordinate
    int batch_idx = cur_geom_feats[3];  // Batch index
    
    // Compute output pointer
    // Output layout: [B, D, H, W, C]
    // Must match original bev_pool_cuda.cu: out[batch][geom[2]][geom[0]][geom[1]][c]
    // geom_feats columns: [x, y, z, batch], so: out[batch][z][x][y][c]
    // With D=Z, H=X, W=Y (matching Python call: bev_pool(x, geom, B, nx[2], nx[0], nx[1]))
    float* cur_out = out +
        batch_idx * d * h * w * c +
        bev_z * h * w * c +
        bev_x * w * c +
        bev_y * c +
        cur_c;
    
    // Sum all points in this interval
    const float* cur_x = x + interval_start * c + cur_c;
    float psum = 0.0f;
    for (int i = 0; i < interval_length; i++) {
        psum += cur_x[i * c];
    }
    
    *cur_out = psum;
}

// C interface for Plugin to call
extern "C" void launch_bev_pool_v2(
    int b, int d, int h, int w, int n_intervals, int c,
    const float* x,
    const int* geom_feats,
    const int* interval_starts,
    const int* interval_lengths,
    float* out, cudaStream_t stream)
{
    int total_threads = n_intervals * c;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    
    // Limit grid size to avoid overflow
    if (grid_size > 65535) grid_size = 65535;
    
    bev_pool_v2_kernel<<<grid_size, block_size, 0, stream>>>(
        b, d, h, w, n_intervals, c,
        x, geom_feats, interval_starts, interval_lengths, out);
}
