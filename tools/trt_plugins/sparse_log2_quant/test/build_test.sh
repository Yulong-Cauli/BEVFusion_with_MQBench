#!/bin/bash
# Build script for test_plugin_registration.cpp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PLUGIN_BUILD_DIR="${SCRIPT_DIR}/../build"

# TRT and CUDA paths
TRT_ROOT="/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6"
CUDA_ROOT="/usr/local/cuda"

echo "=== Building Plugin Registration Test ==="
echo "TRT_ROOT: ${TRT_ROOT}"
echo "CUDA_ROOT: ${CUDA_ROOT}"
echo ""

# Check if plugin is built
if [ ! -f "${PLUGIN_BUILD_DIR}/libsparse_log2_quant_plugin.so" ]; then
    echo "Error: Plugin library not found at ${PLUGIN_BUILD_DIR}/libsparse_log2_quant_plugin.so"
    echo "Please build the plugin first:"
    echo "  cd ${SCRIPT_DIR}/.. && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# Create test build directory
mkdir -p "${BUILD_DIR}"

echo "Compiling test_plugin_registration.cpp..."

g++ -std=c++14 -O2 \
    -I"${TRT_ROOT}/include" \
    -I"${CUDA_ROOT}/include" \
    -I"${SCRIPT_DIR}/../src" \
    "${SCRIPT_DIR}/test_plugin_registration.cpp" \
    -o "${BUILD_DIR}/test_plugin_registration" \
    -L"${TRT_ROOT}/lib" \
    -L"${CUDA_ROOT}/lib64" \
    -L"${PLUGIN_BUILD_DIR}" \
    -lnvinfer \
    -lcudart \
    -lsparse_log2_quant_plugin \
    -Wl,-rpath,"${PLUGIN_BUILD_DIR}"

echo ""
echo "=== Build Successful ==="
echo "Executable: ${BUILD_DIR}/test_plugin_registration"
echo ""
echo "To run the test:"
echo "  export LD_LIBRARY_PATH=${TRT_ROOT}/lib:\$LD_LIBRARY_PATH"
echo "  ${BUILD_DIR}/test_plugin_registration"
