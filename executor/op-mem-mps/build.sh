#!/usr/bin/env bash
set -euo pipefail

mkdir -p build
cd build
rm -rf ./*
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu)
