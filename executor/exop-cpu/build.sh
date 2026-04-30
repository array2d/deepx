#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/deepx/exop-cpu/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$DIR"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Built: $BUILD_DIR/deepx-exop-cpu"
