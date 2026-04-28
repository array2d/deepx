#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/deepx/common-metal/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$DIR"
cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
