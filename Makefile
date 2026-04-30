# ═══════════════════════════════════════════════════════════════
# DeepX 统一构建入口
# ═══════════════════════════════════════════════════════════════
#
# 所有构建必须通过 make 执行:
#   make build-all       构建全部项目
#   make build-vm        构建 VM + loader (Go)
#   make build-deepxctl  构建 deepxctl CLI (Go)
#   make build-op-metal  构建 Metal 计算平面 (C++/Metal)
#   make build-heap-metal 构建 Metal 堆管理平面 (C++/ObjC++)
#   make build-io-metal   构建 I/O 平面 (C++)
#   make test-vm         运行 VM 单元测试
#   make pipeline        完整流水线
#
# 更多目标见 make help

# ── Go 项目路径 ──
VM_DIR          := executor/vm
DEEPXCTL_DIR    := tool/deepxctl
OP_METAL_DIR    := executor/op-metal
HEAP_METAL_DIR  := executor/heap-metal
IO_METAL_DIR    := executor/io-metal

# ── Go 环境 ──
GOROOT          ?= $(HOME)/sdk/go
GOPATH          := $(shell $(GOROOT)/bin/go env GOPATH 2>/dev/null || echo $(HOME)/go)
export GOROOT
export GOPROXY   ?= https://goproxy.cn,direct
export PATH      := $(GOROOT)/bin:$(GOPATH)/bin:$(PATH)

# ── 输出目录 ──
VM_OUT          := /tmp/deepx-vm
OP_METAL_OUT    := /tmp/deepx/op-metal/build
HEAP_METAL_OUT  := /tmp/deepx/heap-metal/build
IO_METAL_OUT    := /tmp/deepx/io-metal/build

# ── Redis 配置 (用于联调) ──
REDIS_ADDR      ?= 127.0.0.1:16379

.PHONY: help \
        build-all build-vm build-deepxctl build-op-metal build-heap-metal build-io-metal \
        test-vm test-integration \
        start-services stop-services status \
        pipeline reset-redis \
        clean clean-all

# ═══════════════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════════════

help:
	@echo "DeepX Makefile (Root)"
	@echo ""
	@echo "BUILD:"
	@echo "  make build-all          Build all projects"
	@echo "  make build-vm           Build VM + loader (Go)          → $(VM_OUT)/vm, $(VM_OUT)/loader"
	@echo "  make build-deepxctl     Build deepxctl (Go)             → $(DEEPXCTL_DIR)/deepxctl"
	@echo "  make build-op-metal     Build Metal compute plane (C++) → $(OP_METAL_OUT)/deepx-op-metal"
	@echo "  make build-heap-metal   Build Metal heap plane (C++)    → $(HEAP_METAL_OUT)/deepx-heap-metal"
	@echo "  make build-io-metal     Build I/O plane (C++)           → $(IO_METAL_OUT)/deepx-io-metal"
	@echo ""
	@echo "TEST:"
	@echo "  make test-vm            Run VM unit tests"
	@echo "  make test-integration   Run VM integration tests (needs Redis)"
	@echo ""
	@echo "SERVICES (daemon):"
	@echo "  make start-services     Start op-metal + heap-metal in background"
	@echo "  make stop-services      Stop all background services"
	@echo "  make status             Check service/Redis status"
	@echo ""
	@echo "PIPELINE:"
	@echo "  make pipeline           Full cycle: build → start → reset → stop"
	@echo ""
	@echo "UTILS:"
	@echo "  make reset-redis        Reset Redis (FLUSHDB)"
	@echo "  make clean              Remove build artifacts"
	@echo "  make clean-all          Clean all including temp output dirs"
	@echo ""
	@echo "Config via env:"
	@echo "  REDIS_ADDR=$(REDIS_ADDR)   GOROOT=$(GOROOT)   GOPROXY=$(GOPROXY)"

# ═══════════════════════════════════════════════════════════════
# Build — All
# ═══════════════════════════════════════════════════════════════

build-all: build-vm build-deepxctl build-op-metal build-heap-metal build-io-metal
	@echo "=== build-all complete ==="

# ═══════════════════════════════════════════════════════════════
# Build — Go Projects
# ═══════════════════════════════════════════════════════════════

build-vm:
	@echo "=== Building VM ==="
	@command -v go >/dev/null 2>&1 || (echo "ERROR: go not found in PATH (GOROOT=$(GOROOT))" && exit 1)
	@echo "Go version: $$(go version)"
	mkdir -p $(VM_OUT)
	cd $(VM_DIR) && go mod tidy
	cd $(VM_DIR) && go build -ldflags="-s -w" -o $(VM_OUT)/vm ./cmd/vm/
	cd $(VM_DIR) && go build -ldflags="-s -w" -o $(VM_OUT)/loader ./cmd/loader/
	@echo "  → $(VM_OUT)/vm"
	@echo "  → $(VM_OUT)/loader"

build-deepxctl:
	@echo "=== Building deepxctl ==="
	@command -v go >/dev/null 2>&1 || (echo "ERROR: go not found in PATH (GOROOT=$(GOROOT))" && exit 1)
	@echo "Go version: $$(go version)"
	cd $(DEEPXCTL_DIR) && go mod tidy
	cd $(DEEPXCTL_DIR) && go build -ldflags="-s -w" -o deepxctl .
	@echo "  → $(DEEPXCTL_DIR)/deepxctl"

# ═══════════════════════════════════════════════════════════════
# Build — C++ Projects (delegate to executor/Makefile)
# ═══════════════════════════════════════════════════════════════

build-op-metal:
	@echo "=== Building op-metal ==="
	cd executor && $(MAKE) build-op

build-heap-metal:
	@echo "=== Building heap-metal ==="
	cd executor && $(MAKE) build-heap

build-io-metal:
	@echo "=== Building io-metal ==="
	cd executor && $(MAKE) build-io

# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

test-vm:
	cd $(VM_DIR) && go test ./... -count=1 -run "^Test[^I]" -v

test-integration:
	cd executor && $(MAKE) test-integration REDIS_ADDR=$(REDIS_ADDR)

# ═══════════════════════════════════════════════════════════════
# Services & Pipeline (delegate to executor/Makefile)
# ═══════════════════════════════════════════════════════════════

start-services:
	cd executor && $(MAKE) start-services REDIS_ADDR=$(REDIS_ADDR)

stop-services:
	cd executor && $(MAKE) stop-services

status:
	cd executor && $(MAKE) status REDIS_ADDR=$(REDIS_ADDR)

reset-redis:
	cd executor && $(MAKE) reset-redis REDIS_ADDR=$(REDIS_ADDR)

pipeline:
	cd executor && $(MAKE) pipeline REDIS_ADDR=$(REDIS_ADDR)

# ═══════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════

clean:
	cd executor && $(MAKE) clean
	cd $(DEEPXCTL_DIR) && rm -f deepxctl
	cd $(VM_DIR) && go clean -testcache

clean-all: clean
	rm -rf $(VM_OUT)
	rm -rf $(OP_METAL_OUT)
	rm -rf $(HEAP_METAL_OUT)
	rm -rf $(IO_METAL_OUT)
