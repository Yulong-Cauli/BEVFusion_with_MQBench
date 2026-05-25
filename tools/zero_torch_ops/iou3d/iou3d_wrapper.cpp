/*
  C API wrapper for iou3d CUDA ops (inference only).
  No torch/ATen/pybind11 dependencies — callable via ctypes.
*/
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

// Kernel launchers declared in iou3d_kernel.cu
extern "C" void boxesoverlapLauncher(const int num_a, const float *boxes_a,
                                     const int num_b, const float *boxes_b,
                                     float *ans_overlap);
extern "C" void boxesioubevLauncher(const int num_a, const float *boxes_a,
                                    const int num_b, const float *boxes_b,
                                    float *ans_iou);
extern "C" void nmsLauncher(const float *boxes, unsigned long long *mask,
                              int boxes_num, float nms_overlap_thresh);
extern "C" void nmsNormalLauncher(const float *boxes, unsigned long long *mask,
                                    int boxes_num, float nms_overlap_thresh);

extern "C" int boxes_overlap_bev_gpu_cuda(const int num_a, const float *boxes_a,
                                          const int num_b, const float *boxes_b,
                                          float *ans_overlap) {
  boxesoverlapLauncher(num_a, boxes_a, num_b, boxes_b, ans_overlap);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  return 0;
}

extern "C" int boxes_iou_bev_gpu_cuda(const int num_a, const float *boxes_a,
                                      const int num_b, const float *boxes_b,
                                      float *ans_iou) {
  boxesioubevLauncher(num_a, boxes_a, num_b, boxes_b, ans_iou);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  return 0;
}

extern "C" int nms_gpu_cuda(const float *boxes_data, int64_t *keep_data,
                            int boxes_num, float nms_overlap_thresh,
                            int device_id) {
  cudaSetDevice(device_id);

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_data = nullptr;
  cudaError_t err = cudaMalloc((void **)&mask_data,
                               boxes_num * col_blocks * sizeof(unsigned long long));
  if (err != cudaSuccess) return (int)err;

  nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(mask_data);
    return (int)err;
  }

  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
  err = cudaMemcpy(&mask_cpu[0], mask_data,
                   boxes_num * col_blocks * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
  cudaFree(mask_data);
  if (err != cudaSuccess) return (int)err;

  unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;
    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  delete[] remv_cpu;

  err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  return num_to_keep;
}

extern "C" int nms_normal_gpu_cuda(const float *boxes_data, int64_t *keep_data,
                                   int boxes_num, float nms_overlap_thresh,
                                   int device_id) {
  cudaSetDevice(device_id);

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_data = nullptr;
  cudaError_t err = cudaMalloc((void **)&mask_data,
                               boxes_num * col_blocks * sizeof(unsigned long long));
  if (err != cudaSuccess) return (int)err;

  nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(mask_data);
    return (int)err;
  }

  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
  err = cudaMemcpy(&mask_cpu[0], mask_data,
                   boxes_num * col_blocks * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
  cudaFree(mask_data);
  if (err != cudaSuccess) return (int)err;

  unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;
    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  delete[] remv_cpu;

  err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  return num_to_keep;
}
