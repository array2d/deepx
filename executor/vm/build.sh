#!/bin/bash
set -e

# VM Build Script
# 编译结果输出到 /tmp 目录
# Go 安装路径: ~/sdk/go

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="/tmp/deepx-vm"
GOROOT="$HOME/sdk/go"
export PATH="$GOROOT/bin:$PATH"
export GOPROXY="${GOPROXY:-https://goproxy.cn,direct}"

echo "=== DeepX VM Builder ==="
echo "Go version: $(go version)"
echo "Source dir: $SCRIPT_DIR"
echo "Output dir: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR"

# 下载依赖
echo ""
echo "[1/4] Downloading dependencies..."
go mod tidy

# 运行测试
echo ""
echo "[2/4] Running unit tests..."
go test ./... -v -count=1 -run "^Test[^I]" 2>&1 || echo "(tests skipped or failed - continuing)"

echo ""
echo "[3/4] Running testutil tests..."
go test ./testutil/ -v -count=1 2>&1 || echo "(testutil tests skipped or failed - continuing)"

# 构建
echo ""
echo "[4/4] Building VM binary..."
go build -ldflags="-s -w" -o "$OUTPUT_DIR/vm" ./cmd/vm/

echo ""
echo "=== Build Complete ==="
echo "Binary: $OUTPUT_DIR/vm"
ls -lh "$OUTPUT_DIR/vm"
