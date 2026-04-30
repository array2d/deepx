#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/deepx/heap-metal/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$DIR"
cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
# Copy runtime dependencies (rpath)
cp -f common-metal/libdeepx_common_metal.a "$BUILD_DIR/" 2>/dev/null || true
echo "Built: $BUILD_DIR/deepx-heap-metal"
