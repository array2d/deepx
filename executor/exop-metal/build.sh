#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/deepx/exop-metal"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$DIR"
cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
# Copy runtime dependencies (rpath: @rpath/libdeepx_metal.dylib)
cp -f libdeepx_metal.dylib default.metallib "$BUILD_DIR/" 2>/dev/null || true
echo "Built: $BUILD_DIR/deepx-exop-metal"
echo "Test:   $BUILD_DIR/test/shm/test_cross_process"
