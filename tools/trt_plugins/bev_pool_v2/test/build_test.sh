#!/bin/bash
# Build BEVPoolV2 Plugin Registration Test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Environment
TRT_ROOT="/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6"
CUDA_ROOT="/usr/local/cuda"
PLUGIN_SO="${SCRIPT_DIR}/../build/libbev_pool_v2_plugin.so"

# Check if plugin library exists
if [ ! -f "$PLUGIN_SO" ]; then
    echo "Error: Plugin library not found at $PLUGIN_SO"
    echo "Please build the plugin first: cd ../build && make"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Compile test program
echo "Compiling test_plugin_registration.cpp..."
g++ -std=c++14 -O2 \
    -I"${TRT_ROOT}/include" \
    -I"${CUDA_ROOT}/include" \
    -I"../src" \
    -L"${TRT_ROOT}/lib" \
    -L"${CUDA_ROOT}/lib64" \
    ../test_plugin_registration.cpp \
    -o test_plugin_registration \
    "${PLUGIN_SO}" \
    -lnvinfer -lnvinfer_plugin -lcudart \
    -Wl,-rpath,"${TRT_ROOT}/lib" \
    -Wl,-rpath,"${SCRIPT_DIR}/../build"

echo "Build successful!"
echo ""
echo "To run the test:"
echo "  export LD_LIBRARY_PATH=\"${TRT_ROOT}/lib:${SCRIPT_DIR}/../build:\$LD_LIBRARY_PATH\""
echo "  ./test_plugin_registration"
