/*
  Pillar pooling CUDA kernels (inference only).
  Modified from mmdet3d/ops/bev_pool/src/bev_pool_cuda.cu
  Removed all torch/ATen dependencies — pure CUDA C.
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>

__global__ void bev_pool_kernel(int b, int d, int h, int w, int n, int c,
                                int n_intervals,
                                const float *__restrict__ x,
                                const int *__restrict__ geom_feats,
                                const int *__restrict__ interval_starts,
                                const int *__restrict__ interval_lengths,
                                float *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int *cur_geom_feats = geom_feats + interval_start * 4;
  const float *cur_x = x + interval_start * c + cur_c;
  float *cur_out = out + cur_geom_feats[3] * d * h * w * c +
                   cur_geom_feats[2] * h * w * c +
                   cur_geom_feats[0] * w * c +
                   cur_geom_feats[1] * c + cur_c;
  float psum = 0;
  for (int i = 0; i < interval_length; i++) {
    psum += cur_x[i * c];
  }
  *cur_out = psum;
}

__global__ void bev_pool_half_kernel(int b, int d, int h, int w, int n, int c,
                                     int n_intervals,
                                     const __half *__restrict__ x,
                                     const int *__restrict__ geom_feats,
                                     const int *__restrict__ interval_starts,
                                     const int *__restrict__ interval_lengths,
                                     float *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int *cur_geom_feats = geom_feats + interval_start * 4;
  const __half *cur_x = x + interval_start * c + cur_c;
  float *cur_out = out + cur_geom_feats[3] * d * h * w * c +
                   cur_geom_feats[2] * h * w * c +
                   cur_geom_feats[0] * w * c +
                   cur_geom_feats[1] * c + cur_c;
  float psum = 0;
  for (int i = 0; i < interval_length; i++) {
    psum += __half2float(cur_x[i * c]);
  }
  *cur_out = psum;
}

extern "C" void bev_pool(int b, int d, int h, int w, int n, int c, int n_intervals,
              const float *x, const int *geom_feats,
              const int *interval_starts, const int *interval_lengths,
              float *out) {
  int total_threads = n_intervals * c;
  int block_size = 256;
  int grid_size = (total_threads + block_size - 1) / block_size;
  bev_pool_kernel<<<grid_size, block_size>>>(
      b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts,
      interval_lengths, out);
}

extern "C" void bev_pool_half(int b, int d, int h, int w, int n, int c, int n_intervals,
                                const __half *x, const int *geom_feats,
                                const int *interval_starts, const int *interval_lengths,
                                float *out) {
  int total_threads = n_intervals * c;
  int block_size = 256;
  int grid_size = (total_threads + block_size - 1) / block_size;
  bev_pool_half_kernel<<<grid_size, block_size>>>(
      b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts,
      interval_lengths, out);
}
