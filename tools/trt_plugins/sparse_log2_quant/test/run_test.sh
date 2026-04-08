#!/bin/bash
# Run script for test_plugin_registration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PLUGIN_BUILD_DIR="${SCRIPT_DIR}/../build"
TRT_ROOT="/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6"

# Check if executable exists
if [ ! -f "${BUILD_DIR}/test_plugin_registration" ]; then
    echo "Error: Test executable not found. Please run build_test.sh first."
    exit 1
fi

echo "=== Running Plugin Registration Test ==="
echo ""

# Set library path and run
export LD_LIBRARY_PATH="${TRT_ROOT}/lib:${PLUGIN_BUILD_DIR}:${LD_LIBRARY_PATH}"
"${BUILD_DIR}/test_plugin_registration"
