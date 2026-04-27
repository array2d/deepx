#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/deepx/executor/op-cuda/build
cd /tmp/deepx/executor/op-cuda/build
rm -rf ./*
cmake "$(cd "$(dirname "$0")" && pwd)"
make -j$(nproc)
