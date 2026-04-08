# CMakeLists.txt for BEVPoolV2 TRT Plugin
#
# 编译命令（服务器端）：
#   cd tools/trt_plugins/bev_pool_v2
#   mkdir -p build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#
# 输出：libbev_pool_v2_plugin.so

cmake_minimum_required(VERSION 3.14)
project(bev_pool_v2 CUDA CXX)

# C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── 环境配置（服务器端路径）─────────────────────────────────────────────────
set(TRT_ROOT  "/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6")
set(CUDA_ROOT "/usr/local/cuda")

# CUDA 架构：RTX 3090 (Ampere SM 8.6)
set(CMAKE_CUDA_ARCHITECTURES "86")

# ── 查找依赖 ─────────────────────────────────────────────────────────────────
find_package(CUDA REQUIRED)

# 包含目录
include_directories(
    ${TRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 库目录
link_directories(
    ${TRT_ROOT}/lib
    ${CUDA_ROOT}/lib64
)

# ── 编译 Plugin ─────────────────────────────────────────────────────────────

# 使用 add_library + CUDA 源文件
add_library(bev_pool_v2_plugin SHARED
    src/bev_pool_v2_plugin.cpp
    src/bev_pool_v2_kernel.cu
)

# 设置 CUDA 语言属性
set_target_properties(bev_pool_v2_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET default
    CUDA_VISIBILITY_PRESET default
)

# 防止符号被链接器优化掉（关键！）
target_link_options(bev_pool_v2_plugin PRIVATE
    "-Wl,--no-as-needed"
    "-Wl,-soname,libbev_pool_v2_plugin.so"
    "-Wl,--whole-archive"
    "-Wl,--no-whole-archive"
)

# 强制保留所有符号
set_target_properties(bev_pool_v2_plugin PROPERTIES
    LINK_FLAGS "-Wl,--export-dynamic"
)

# 链接库
target_link_libraries(bev_pool_v2_plugin
    nvinfer
    nvinfer_plugin
    cudart
)

# ── 编译选项 ─────────────────────────────────────────────────────────────────
target_compile_options(bev_pool_v2_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
)

# ── 安装规则（可选）──────────────────────────────────────────────────────────
install(TARGETS bev_pool_v2_plugin
    LIBRARY DESTINATION lib
)

# ── 打印配置信息 ─────────────────────────────────────────────────────────────
message(STATUS "=== BEVPoolV2 Plugin Configuration ===")
message(STATUS "TRT_ROOT: ${TRT_ROOT}")
message(STATUS "CUDA_ROOT: ${CUDA_ROOT}")
message(STATUS "CUDA_ARCH: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
