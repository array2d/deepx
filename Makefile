# ═══════════════════════════════════════════════════════════════
# DeepX 统一构建入口
# ═══════════════════════════════════════════════════════════════
#
# 所有构建必须通过 make 执行:
#   make build-all       构建全部项目
#   make build-vm        构建 VM + loader (Go)
#   make build-deepxctl  构建 deepxctl CLI (Go)
#   make build-exop-metal  构建 Metal 计算平面 (C++/Metal)
#   make build-heap-metal 构建 Metal 堆管理平面 (C++/ObjC++)
#   make build-io-metal   构建 I/O 平面 (C++)
#   make test-vm         运行 VM 单元测试
#   make pipeline        完整流水线
#
# 更多目标见 make help

# ── Go 项目路径 ──
VM_DIR          := executor/vm
DEEPXCTL_DIR    := tool/deepxctl
DASHBOARD_DIR   := tool/dashboard
EXOP_METAL_DIR  := executor/exop-metal
HEAP_METAL_DIR  := executor/heap-metal
IO_METAL_DIR    := executor/io-metal
EXOP_CPU_DIR    := executor/exop-cpu
HEAP_CPU_DIR    := executor/heap-cpu

# ── Go 环境 ──
GOROOT          ?= $(HOME)/sdk/go
GOPATH          := $(shell $(GOROOT)/bin/go env GOPATH 2>/dev/null || echo $(HOME)/go)
export GOROOT
export GOPROXY   ?= https://goproxy.cn,direct
export PATH      := $(GOROOT)/bin:$(GOPATH)/bin:$(PATH)

# ── 输出目录 ──
VM_OUT          := /tmp/deepx-vm
EXOP_METAL_OUT  := /tmp/deepx/exop-metal/build
HEAP_METAL_OUT  := /tmp/deepx/heap-metal/build
IO_METAL_OUT    := /tmp/deepx/io-metal/build
# ── 跨平台: 产物按 OS-ARCH 命名 ──
CPU_PLAT        := $(shell uname -s | tr '[:upper:]' '[:lower:]')-$(shell uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
EXOP_CPU_OUT    := /tmp/deepx/exop-cpu/build/$(CPU_PLAT)
BIN_EXOP_CPU    := $(EXOP_CPU_OUT)/deepx-exop-cpu-$(CPU_PLAT)
HEAP_CPU_OUT    := /tmp/deepx/heap-cpu/build/$(CPU_PLAT)
BIN_HEAP_CPU    := $(HEAP_CPU_OUT)/deepx-heap-cpu-$(CPU_PLAT)
DASHBOARD_OUT   := /tmp/deepx-dashboard

# ── Redis 配置 (用于联调) ──
REDIS_ADDR      ?= 127.0.0.1:16379

.PHONY: help \
        build-all build-vm build-deepxctl build-dashboard \
        build-exop-metal build-heap-metal build-io-metal \
        build-exop-cpu build-heap-cpu \
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
	@echo "  make build-exop-metal     Build Metal compute plane (C++) → $(EXOP_METAL_OUT)/deepx-exop-metal"
	@echo "  make build-heap-metal   Build Metal heap plane (C++)    → $(HEAP_METAL_OUT)/deepx-heap-metal"
	@echo "  make build-io-metal     Build I/O plane (C++)           → $(IO_METAL_OUT)/deepx-io-metal"
	@echo "  make build-exop-cpu      Build CPU compute plane (C++/Redis)→ $(BIN_EXOP_CPU)"
	@echo "  make build-heap-cpu      Build CPU heap plane (C++/Redis)  → $(BIN_HEAP_CPU)"
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

build-all:
	@echo "=== build-all ==="
	@pass=0; fail=0; \
	targets="build-vm build-deepxctl build-dashboard build-exop-metal build-heap-metal build-io-metal build-exop-cpu build-heap-cpu"; \
	for t in $$targets; do \
		if $(MAKE) --no-print-directory $$t; then \
			pass=$$((pass+1)); \
		else \
			fail=$$((fail+1)); \
			echo "  [FAIL] $$t (continuing)"; \
		fi; \
	done; \
	echo "=== build-all done: $$pass passed, $$fail failed ==="; \
	[ $$fail -eq 0 ]

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

build-dashboard:
	@echo "=== Building dashboard ==="
	@command -v go >/dev/null 2>&1 || (echo "ERROR: go not found in PATH (GOROOT=$(GOROOT))" && exit 1)
	@echo "Go version: $$(go version)"
	cd $(DASHBOARD_DIR) && go mod tidy
	cd $(DASHBOARD_DIR) && go build -ldflags="-s -w" -o dashboard .
	@echo "  → $(DASHBOARD_DIR)/dashboard"

# ═══════════════════════════════════════════════════════════════
# Build — C++ Projects
# ═══════════════════════════════════════════════════════════════

build-exop-metal:
	@echo "=== Building exop-metal ==="
	bash $(EXOP_METAL_DIR)/build.sh
	@echo "  → $(EXOP_METAL_OUT)/deepx-exop-metal"

build-heap-metal:
	@echo "=== Building heap-metal ==="
	bash $(HEAP_METAL_DIR)/build.sh
	@echo "  → $(HEAP_METAL_OUT)/deepx-heap-metal"

build-io-metal:
	@echo "=== Building io-metal ==="
	bash $(IO_METAL_DIR)/build.sh
	@echo "  → $(IO_METAL_OUT)/deepx-io-metal"

build-exop-cpu:
	@echo "=== Building exop-cpu ($(CPU_PLAT)) ==="
	bash $(EXOP_CPU_DIR)/build.sh
	@echo "  → $(BIN_EXOP_CPU)         (计算执行器，对接 Redis)"

build-heap-cpu:
	@echo "=== Building heap-cpu ($(CPU_PLAT)) ==="
	bash $(HEAP_CPU_DIR)/build.sh
	@echo "  → $(BIN_HEAP_CPU)         (生命周期管理器，对接 Redis)"

# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

test-vm:
	cd $(VM_DIR) && go test ./... -count=1 -run "^Test[^I]" -v

test-integration:
	cd $(VM_DIR) && REDIS_ADDR=$(REDIS_ADDR) go test -tags=integration -count=1 -v -run 'TestIntegration' ./...

# ═══════════════════════════════════════════════════════════════
# Services (daemon)
# ═══════════════════════════════════════════════════════════════

start-services:
	@echo "Starting services with deepxctl boot..."
	cd $(DEEPXCTL_DIR) && go run . boot -r $(REDIS_ADDR)

stop-services:
	@echo "Stopping services with deepxctl shutdown..."
	cd $(DEEPXCTL_DIR) && go run . shutdown

status:
	@echo "=== deepx Status ==="
	@echo "Redis: $(REDIS_ADDR)"
	@redis-cli -h $(word 1,$(subst :, ,$(REDIS_ADDR))) -p $(word 2,$(subst :, ,$(REDIS_ADDR))) PING 2>/dev/null || echo "  → NOT REACHABLE"
	@cat /tmp/deepx-boot.json 2>/dev/null || echo "  No boot state file"

reset-redis:
	@echo "Resetting Redis at $(REDIS_ADDR)..."
	redis-cli -h $(word 1,$(subst :, ,$(REDIS_ADDR))) -p $(word 2,$(subst :, ,$(REDIS_ADDR))) FLUSHDB

pipeline: build-all start-services reset-redis stop-services
	@echo "=== Pipeline complete ==="

# ═══════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════

clean:
	rm -rf $(EXOP_METAL_OUT) $(HEAP_METAL_OUT) $(IO_METAL_OUT) $(EXOP_CPU_OUT) $(HEAP_CPU_OUT)
	cd $(DEEPXCTL_DIR) && rm -f deepxctl
	cd $(VM_DIR) && go clean -testcache

clean-all: clean
	rm -rf $(VM_OUT)
	rm -rf $(EXOP_METAL_OUT)
	rm -rf $(HEAP_METAL_OUT)
	rm -rf $(IO_METAL_OUT)
	rm -rf $(EXOP_CPU_OUT)
	rm -rf $(HEAP_CPU_OUT)
	rm -rf $(DASHBOARD_OUT)
