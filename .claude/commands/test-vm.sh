#!/usr/bin/env bash
# Run VM unit tests (non-integration)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VM_DIR="$SCRIPT_DIR/../../executor/vm"
GOROOT="${GOROOT:-$HOME/sdk/go}"
export PATH="$GOROOT/bin:$PATH"
export GOPROXY="${GOPROXY:-https://goproxy.cn,direct}"

cd "$VM_DIR"

echo "=== VM Unit Tests ==="
echo "Go version: $(go version)"

# Unit tests (exclude integration tests)
go test ./... -v -count=1 -run "^Test[^I]" 2>&1

echo "=== VM testutil Tests ==="
go test ./testutil/ -v -count=1 2>&1

echo "=== Tests Complete ==="
