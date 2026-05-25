// Modified from mmdet3d/ops/voxel/src/voxelization_cuda.cu
// Removed all torch/ATen dependencies — pure CUDA C.

#include <cuda_runtime.h>
#include <math.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void dynamic_voxelize_kernel_float_int(
    const float* points, int* coors,
    const float voxel_x, const float voxel_y, const float voxel_z,
    const float coors_x_min, const float coors_y_min, const float coors_z_min,
    const float coors_x_max, const float coors_y_max, const float coors_z_max,
    const int grid_x, const int grid_y, const int grid_z,
    const int num_points, const int num_features, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floorf((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floorf((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floorf((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_x;
      coors_offset[1] = c_y;
      coors_offset[2] = c_z;
    }
  }
}

__global__ void point_to_voxelidx_kernel_int(
    const int* coor, int* point_to_voxelidx, int* point_to_pointidx,
    const int max_points, const int max_voxels,
    const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    if ((index >= num_points) || (coor_offset[0] == -1)) return;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) continue;

      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          return;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    }
  }
}

__global__ void determin_voxel_num_int(
    int* num_points_per_voxel, int* point_to_voxelidx,
    int* point_to_pointidx, int* coor_to_voxelidx, int* voxel_num,
    const int max_points, const int max_voxels, const int num_points) {
  for (int i = 0; i < num_points; ++i) {
    int point_pos_in_voxel = point_to_voxelidx[i];
    if (point_pos_in_voxel == -1) {
      continue;
    } else if (point_pos_in_voxel == 0) {
      int voxelidx = voxel_num[0];
      if (voxel_num[0] >= max_voxels) continue;
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        coor_to_voxelidx[i] = voxelidx;
        num_points_per_voxel[voxelidx] += 1;
      }
    }
  }
}

__global__ void assign_point_to_voxel_float_int(
    const int nthreads, const float* points,
    int* point_to_voxelidx, int* coor_to_voxelidx, float* voxels,
    const int max_points, const int num_features,
    const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int index = thread_idx / num_features;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      auto voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;
      int k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

__global__ void assign_voxel_coors_int(
    const int nthreads, const int* coor,
    int* point_to_voxelidx, int* coor_to_voxelidx, int* voxel_coors,
    const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int index = thread_idx / NDim;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      auto coors_offset = voxel_coors + voxelidx * NDim;
      int k = thread_idx % NDim;
      coors_offset[k] = coor[thread_idx];
    }
  }
}

// ------------------------------------------------------------------
// Host launchers (no torch)
// ------------------------------------------------------------------

extern "C" void dynamicVoxelizeLauncher(
    const float* points, int* coors,
    float voxel_x, float voxel_y, float voxel_z,
    float coors_x_min, float coors_y_min, float coors_z_min,
    float coors_x_max, float coors_y_max, float coors_z_max,
    int grid_x, int grid_y, int grid_z,
    int num_points, int num_features, int NDim) {
  dim3 grid((num_points + 512 - 1) / 512);
  if (grid.x > 4096) grid.x = 4096;
  dim3 block(512);
  dynamic_voxelize_kernel_float_int<<<grid, block>>>(
      points, coors, voxel_x, voxel_y, voxel_z,
      coors_x_min, coors_y_min, coors_z_min,
      coors_x_max, coors_y_max, coors_z_max,
      grid_x, grid_y, grid_z, num_points, num_features, NDim);
}

extern "C" void dynamicVoxelizeLauncherSmallBlock(
    const float* points, int* coors,
    float voxel_x, float voxel_y, float voxel_z,
    float coors_x_min, float coors_y_min, float coors_z_min,
    float coors_x_max, float coors_y_max, float coors_z_max,
    int grid_x, int grid_y, int grid_z,
    int num_points, int num_features, int NDim) {
  dim3 grid((num_points + 64 - 1) / 64);
  dim3 block(64);
  dynamic_voxelize_kernel_float_int<<<grid, block>>>(
      points, coors, voxel_x, voxel_y, voxel_z,
      coors_x_min, coors_y_min, coors_z_min,
      coors_x_max, coors_y_max, coors_z_max,
      grid_x, grid_y, grid_z, num_points, num_features, NDim);
}

extern "C" void pointToVoxelidxLauncher(
    const int* coor, int* point_to_voxelidx, int* point_to_pointidx,
    int max_points, int max_voxels, int num_points, int NDim) {
  dim3 grid((num_points + 512 - 1) / 512);
  if (grid.x > 4096) grid.x = 4096;
  dim3 block(512);
  point_to_voxelidx_kernel_int<<<grid, block>>>(
      coor, point_to_voxelidx, point_to_pointidx,
      max_points, max_voxels, num_points, NDim);
}

extern "C" void determinVoxelNumLauncher(
    int* num_points_per_voxel, int* point_to_voxelidx,
    int* point_to_pointidx, int* coor_to_voxelidx, int* voxel_num,
    int max_points, int max_voxels, int num_points) {
  determin_voxel_num_int<<<1, 1>>>(
      num_points_per_voxel, point_to_voxelidx, point_to_pointidx,
      coor_to_voxelidx, voxel_num, max_points, max_voxels, num_points);
}

extern "C" void assignPointToVoxelLauncher(
    int nthreads, const float* points, int* point_to_voxelidx,
    int* coor_to_voxelidx, float* voxels,
    int max_points, int num_features, int num_points, int NDim) {
  dim3 grid((nthreads + 512 - 1) / 512);
  if (grid.x > 4096) grid.x = 4096;
  dim3 block(512);
  assign_point_to_voxel_float_int<<<grid, block>>>(
      nthreads, points, point_to_voxelidx, coor_to_voxelidx,
      voxels, max_points, num_features, num_points, NDim);
}

extern "C" void assignVoxelCoorsLauncher(
    int nthreads, const int* coor, int* point_to_voxelidx,
    int* coor_to_voxelidx, int* voxel_coors,
    int num_points, int NDim) {
  dim3 grid((nthreads + 512 - 1) / 512);
  if (grid.x > 4096) grid.x = 4096;
  dim3 block(512);
  assign_voxel_coors_int<<<grid, block>>>(
      nthreads, coor, point_to_voxelidx, coor_to_voxelidx,
      voxel_coors, num_points, NDim);
}
