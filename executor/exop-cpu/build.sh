#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 跨平台: OS + 芯片架构 ──
OS=$(uname -s | tr '[:upper:]' '[:lower:]')           # darwin / linux
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')  # arm64 / amd64
PLAT="${OS}-${ARCH}"

BUILD_DIR="/tmp/deepx/exop-cpu/build/${PLAT}"
BIN_NAME="deepx-exop-cpu-${PLAT}"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ── 平台特定 CMake 参数 ──
CMAKE_EXTRA=""
case "$OS" in
    darwin)
        CMAKE_EXTRA="-DOpenMP_ROOT=/opt/homebrew/opt/libomp"
        ;;
    linux)
        CMAKE_EXTRA="-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake"
        ;;
esac

cmake "$DIR" -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/openblas;/opt/homebrew/opt/jemalloc;/opt/homebrew/opt/yaml-cpp" $CMAKE_EXTRA
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# 重命名产物为平台特定名
if [ -f "$BUILD_DIR/deepx-exop-cpu" ]; then
    mv "$BUILD_DIR/deepx-exop-cpu" "$BUILD_DIR/$BIN_NAME"
fi

echo "Built: $BUILD_DIR/$BIN_NAME"
