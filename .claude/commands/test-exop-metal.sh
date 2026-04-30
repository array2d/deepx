#!/usr/bin/env bash
# Build exop-metal and run tests (shm cross-process tests)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OP_DIR="$SCRIPT_DIR/../../executor/exop-metal"
BUILD_DIR="/tmp/deepx/exop-metal/build"

echo "=== Build exop-metal ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$OP_DIR"
cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

echo ""
echo "=== Run SHM Cross-Process Test ==="
if [ -f "$BUILD_DIR/test/shm/test_cross_process" ]; then
    "$BUILD_DIR/test/shm/test_cross_process"
    echo "SHM test passed."
else
    echo "Test binary not found."
fi
