/**
 * @file sparse_log2_quant_kernel.cu
 * @brief CUDA kernel for SparseLog2Quant TRT Plugin.
 *
 * 数学实现（与 tools/quant_ptq_minmax.py 中的 SparseLog2FakeQuantize.forward 一致）：
 *   zero_mask = |x| < eps
 *   log2_x = log2(|x|) - base
 *   q = round(log2_x).clamp(-127, 127)
 *   x_dq = sign(x) * 2^(q + base)
 *   x_dq = where(zero_mask, 0, x_dq)
 *
 * 用途：lidar/backbone 的稀疏卷积特征呈二模态分布（大多数为 0），
 *       对数域量化比线性 INT8 精度高很多。
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>  // for uint32_t, int32_t, int64_t

// 计算通道索引（支持 NCHW 格式）
// 对于 [N, C] 格式：c = idx % C
// 对于 [N, C, H, W] 格式：c = (idx / (H*W)) % C
static inline __device__ int32_t get_channel_idx(
    int64_t idx,
    int32_t C,
    int32_t nb_dims,
    int32_t dim2,   // dims[2] = H or D
    int32_t dim3    // dims[3] = W (if nb_dims==4)
) {
    if (nb_dims == 2) {
        // [N, C] 格式
        return idx % C;
    } else if (nb_dims == 4) {
        // [N, C, H, W] 格式
        int32_t HW = dim2 * dim3;
        return (idx / HW) % C;
    } else if (nb_dims == 3) {
        // [N, C, D] 格式（较少见）
        return (idx / dim2) % C;
    } else {
        // 其他格式：假设 channel 维度是第 1 维
        return idx % C;
    }
}

__global__ void sparse_log2_quant_kernel(
    const half* __restrict__ input,
    half*       __restrict__ output,
    const float*  __restrict__ log2_base,  // [1] 或 [C]
    int32_t dim0,                          // N
    int32_t dim1,                          // C
    int32_t dim2,                          // H or D or 1
    int32_t dim3,                          // W or 1
    int32_t nb_dims,                       // 维度数量 (2, 3, or 4)
    bool per_channel,
    int32_t qmin,                          // -127 for INT8
    int32_t qmax,                          // 127 for INT8
    float eps
) {
    // 计算总元素数
    int64_t total_elements = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3;
    
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const int32_t C = dim1;  // 通道数

    // 获取通道索引
    const int32_t c = per_channel ? get_channel_idx(idx, C, nb_dims, dim2, dim3) : 0;
    const float base = log2_base[c];

    // 读取输入（half -> float）
    const float x = __half2float(input[idx]);

    // 零值判断
    if (fabsf(x) < eps) {
        output[idx] = __float2half(0.0f);
        return;
    }

    // 对数域量化
    const float sign_x = (x > 0.0f) ? 1.0f : -1.0f;
    const float log2_x = log2f(fmaxf(fabsf(x), 1e-30f)) - base;
    float q = roundf(log2_x);
    q = fmaxf(static_cast<float>(qmin), fminf(static_cast<float>(qmax), q));

    // 反量化
    const float x_dq = sign_x * exp2f(q + base);
    output[idx] = __float2half(x_dq);
}

// C 接口供 Plugin 调用
// 注意：必须与 sparse_log2_quant_plugin.h 中的声明完全匹配
extern "C"
void launch_sparse_log2_quant(
    const void* input,
    void* output,
    const float* log2_base,
    int32_t dim0,             // N
    int32_t dim1,             // C
    int32_t dim2,             // H or D
    int32_t dim3,             // W
    int32_t nb_dims,          // 维度数量
    bool per_channel,
    cudaStream_t stream
) {
    // 计算总元素数
    int64_t total_elements = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3;

    // 配置 kernel
    const int32_t block_size = 256;
    const int64_t grid_size = (total_elements + block_size - 1) / block_size;

    // 限制 grid size（CUDA 最大 gridDim.x 是 INT32_MAX）
    uint32_t grid_dim_x = (grid_size > INT32_MAX) ? INT32_MAX : static_cast<uint32_t>(grid_size);

    sparse_log2_quant_kernel<<<grid_dim_x, block_size, 0, stream>>>(
        reinterpret_cast<const half*>(input),
        reinterpret_cast<half*>(output),
        log2_base,
        dim0,
        dim1,
        dim2,
        dim3,
        nb_dims,
        per_channel,
        -127,  // qmin
        127,   // qmax
        1e-6f  // eps
    );
}
