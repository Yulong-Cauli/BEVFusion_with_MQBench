/*
  C API wrapper for bev_pool_forward (inference only).
  No torch/ATen/pybind11 dependencies — callable via ctypes.
*/
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernels declared in bev_pool_cuda.cu
extern "C" void bev_pool(int b, int d, int h, int w, int n, int c,
                         int n_intervals, const float *x,
                         const int *geom_feats,
                         const int *interval_starts,
                         const int *interval_lengths, float *out);

extern "C" void bev_pool_half(int b, int d, int h, int w, int n, int c,
                                int n_intervals, const __half *x,
                                const int *geom_feats,
                                const int *interval_starts,
                                const int *interval_lengths, float *out);

extern "C" int bev_pool_forward_cuda(int b, int d, int h, int w, int n, int c,
                                       int n_intervals, const float *x,
                                       const int *geom_feats,
                                       const int *interval_starts,
                                       const int *interval_lengths,
                                       float *out) {
  bev_pool(b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts,
           interval_lengths, out);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return (int)err;  // return positive CUDA error code for diagnosis
  }
  return 0;
}

extern "C" int bev_pool_forward_cuda_half(int b, int d, int h, int w, int n, int c,
                                            int n_intervals, const __half *x,
                                            const int *geom_feats,
                                            const int *interval_starts,
                                            const int *interval_lengths,
                                            float *out) {
  bev_pool_half(b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts,
                interval_lengths, out);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return (int)err;
  }
  return 0;
}
