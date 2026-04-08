/**
 * Log2 fake quantization CUDA kernel for TV backbone.
 * Matches SparseLog2FakeQuantize forward behavior (inference only).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__global__ void log2_fake_quant_fp16_kernel(__half* data, int n, float log2_base, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = __half2float(data[idx]);
    if (fabsf(x) < eps) {
        data[idx] = __float2half(0.0f);
        return;
    }
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float log2_x = log2f(fabsf(x)) - log2_base;
    float q_int = roundf(log2_x);
    if (q_int < -127.0f) q_int = -127.0f;
    if (q_int > 127.0f) q_int = 127.0f;
    float x_dq = sign * powf(2.0f, q_int + log2_base);
    data[idx] = __float2half(x_dq);
}

__global__ void log2_fake_quant_fp32_kernel(float* data, int n, float log2_base, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = data[idx];
    if (fabsf(x) < eps) {
        data[idx] = 0.0f;
        return;
    }
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float log2_x = log2f(fabsf(x)) - log2_base;
    float q_int = roundf(log2_x);
    if (q_int < -127.0f) q_int = -127.0f;
    if (q_int > 127.0f) q_int = 127.0f;
    float x_dq = sign * powf(2.0f, q_int + log2_base);
    data[idx] = x_dq;
}

void tv_log2_fake_quant_fp16(void* ptr, int n, float log2_base, float eps, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    log2_fake_quant_fp16_kernel<<<grid, block, 0, stream>>>(static_cast<__half*>(ptr), n, log2_base, eps);
}

void tv_log2_fake_quant_fp32(void* ptr, int n, float log2_base, float eps, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    log2_fake_quant_fp32_kernel<<<grid, block, 0, stream>>>(static_cast<float*>(ptr), n, log2_base, eps);
}

// ---------------------------------------------------------------------------
// BN forward inplace: x = x * scale + shift  (element-wise, per-channel)
// scale/shift are fp32 buffers on GPU; features may be fp16 or fp32.
// ---------------------------------------------------------------------------

__global__ void bn_forward_fp16_kernel(__half* data, const float* scale, const float* shift, int n, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int ch = idx % c;
    float x = __half2float(data[idx]);
    x = x * scale[ch] + shift[ch];
    data[idx] = __float2half(x);
}

__global__ void bn_forward_fp32_kernel(float* data, const float* scale, const float* shift, int n, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int ch = idx % c;
    data[idx] = data[idx] * scale[ch] + shift[ch];
}

void tv_bn_forward_inplace_fp16(void* data, const void* scale, const void* shift, int n, int c, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    bn_forward_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__half*>(data),
        static_cast<const float*>(scale),
        static_cast<const float*>(shift),
        n, c);
}

void tv_bn_forward_inplace_fp32(void* data, const void* scale, const void* shift, int n, int c, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    bn_forward_fp32_kernel<<<grid, block, 0, stream>>>(
        static_cast<float*>(data),
        static_cast<const float*>(scale),
        static_cast<const float*>(shift),
        n, c);
}

} // extern "C"
