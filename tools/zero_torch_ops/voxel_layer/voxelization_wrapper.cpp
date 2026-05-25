/*
  C API wrapper for voxel_layer CUDA ops (hard_voxelize, dynamic_voxelize).
  No torch/ATen/pybind11 dependencies — callable via ctypes.
*/
#include <cuda_runtime.h>
#include <cstdint>

// Kernel launchers declared in voxelization_cuda.cu
extern "C" void dynamicVoxelizeLauncher(
    const float *points, int *coors,
    float voxel_x, float voxel_y, float voxel_z,
    float coors_x_min, float coors_y_min, float coors_z_min,
    float coors_x_max, float coors_y_max, float coors_z_max,
    int grid_x, int grid_y, int grid_z,
    int num_points, int num_features, int NDim);

extern "C" void dynamicVoxelizeLauncherSmallBlock(
    const float *points, int *coors,
    float voxel_x, float voxel_y, float voxel_z,
    float coors_x_min, float coors_y_min, float coors_z_min,
    float coors_x_max, float coors_y_max, float coors_z_max,
    int grid_x, int grid_y, int grid_z,
    int num_points, int num_features, int NDim);

extern "C" void pointToVoxelidxLauncher(
    const int *coor, int *point_to_voxelidx, int *point_to_pointidx,
    int max_points, int max_voxels, int num_points, int NDim);

extern "C" void determinVoxelNumLauncher(
    int *num_points_per_voxel, int *point_to_voxelidx,
    int *point_to_pointidx, int *coor_to_voxelidx, int *voxel_num,
    int max_points, int max_voxels, int num_points);

extern "C" void assignPointToVoxelLauncher(
    int nthreads, const float *points, int *point_to_voxelidx,
    int *coor_to_voxelidx, float *voxels,
    int max_points, int num_features, int num_points, int NDim);

extern "C" void assignVoxelCoorsLauncher(
    int nthreads, const int *coor, int *point_to_voxelidx,
    int *coor_to_voxelidx, int *voxel_coors,
    int num_points, int NDim);

extern "C" int hard_voxelize_gpu_cuda(
    const float *points, float *voxels, int *coors,
    int *num_points_per_voxel, int num_points, int num_features,
    float voxel_x, float voxel_y, float voxel_z,
    float coors_x_min, float coors_y_min, float coors_z_min,
    float coors_x_max, float coors_y_max, float coors_z_max,
    int grid_x, int grid_y, int grid_z,
    int max_points, int max_voxels, int NDim, int device_id) {
  cudaSetDevice(device_id);

  // Zero outputs for safety
  cudaError_t err = cudaMemset(
      voxels, 0, max_voxels * max_points * num_features * sizeof(float));
  if (err != cudaSuccess) return -(int)err;
  err = cudaMemset(coors, 0, max_voxels * NDim * sizeof(int));
  if (err != cudaSuccess) return -(int)err;
  err = cudaMemset(num_points_per_voxel, 0, max_voxels * sizeof(int));
  if (err != cudaSuccess) return -(int)err;

  int *temp_coors = nullptr;
  int *point_to_pointidx = nullptr;
  int *point_to_voxelidx = nullptr;
  int *coor_to_voxelidx = nullptr;
  int *voxel_num = nullptr;
  int voxel_num_cpu = 0;

  err = cudaMalloc((void **)&temp_coors, num_points * NDim * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc((void **)&point_to_pointidx, num_points * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc((void **)&point_to_voxelidx, num_points * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc((void **)&coor_to_voxelidx, num_points * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc((void **)&voxel_num, sizeof(int));
  if (err != cudaSuccess) goto cleanup;

  // Step 1: dynamic voxelize
  dynamicVoxelizeLauncher(
      points, temp_coors, voxel_x, voxel_y, voxel_z, coors_x_min, coors_y_min,
      coors_z_min, coors_x_max, coors_y_max, coors_z_max, grid_x, grid_y,
      grid_z, num_points, num_features, NDim);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) goto cleanup;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  // Step 2: map point to voxel idx / find duplicates
  err = cudaMemset(point_to_pointidx, -1, num_points * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemset(point_to_voxelidx, -1, num_points * sizeof(int));
  if (err != cudaSuccess) goto cleanup;

  pointToVoxelidxLauncher(temp_coors, point_to_voxelidx, point_to_pointidx,
                          max_points, max_voxels, num_points, NDim);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) goto cleanup;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  // Step 3: determine voxel num and coor_to_voxelidx
  err = cudaMemset(coor_to_voxelidx, -1, num_points * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemset(voxel_num, 0, sizeof(int));
  if (err != cudaSuccess) goto cleanup;

  determinVoxelNumLauncher(num_points_per_voxel, point_to_voxelidx,
                           point_to_pointidx, coor_to_voxelidx, voxel_num,
                           max_points, max_voxels, num_points);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) goto cleanup;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  // Step 4: copy point features to voxels
  assignPointToVoxelLauncher(num_points * num_features, points,
                             point_to_voxelidx, coor_to_voxelidx, voxels,
                             max_points, num_features, num_points, NDim);

  // Step 5: copy voxel coordinates
  assignVoxelCoorsLauncher(num_points * NDim, temp_coors, point_to_voxelidx,
                           coor_to_voxelidx, coors, num_points, NDim);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) goto cleanup;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpy(&voxel_num_cpu, voxel_num, sizeof(int),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto cleanup;

cleanup:
  if (temp_coors) cudaFree(temp_coors);
  if (point_to_pointidx) cudaFree(point_to_pointidx);
  if (point_to_voxelidx) cudaFree(point_to_voxelidx);
  if (coor_to_voxelidx) cudaFree(coor_to_voxelidx);
  if (voxel_num) cudaFree(voxel_num);

  if (err != cudaSuccess) return -(int)err;
  return voxel_num_cpu;
}

extern "C" int dynamic_voxelize_gpu_cuda(
    const float *points, int *coors, int num_points, int num_features,
    float voxel_x, float voxel_y, float voxel_z,
    float coors_x_min, float coors_y_min, float coors_z_min,
    float coors_x_max, float coors_y_max, float coors_z_max,
    int grid_x, int grid_y, int grid_z, int NDim, int device_id) {
  cudaSetDevice(device_id);

  dynamicVoxelizeLauncherSmallBlock(
      points, coors, voxel_x, voxel_y, voxel_z, coors_x_min, coors_y_min,
      coors_z_min, coors_x_max, coors_y_max, coors_z_max, grid_x, grid_y,
      grid_z, num_points, num_features, NDim);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return -(int)err;
  err = cudaGetLastError();
  if (err != cudaSuccess) return -(int)err;
  return 0;
}
