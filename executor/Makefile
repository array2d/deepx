# ═══════════════════════════════════════════════════════════════
# deepx Executor — Build & Integration Test Lifecycle Management
# ═══════════════════════════════════════════════════════════════

# ── Config ──
REDIS_ADDR      ?= 127.0.0.1:16379
REDIS_HOST      := $(word 1,$(subst :, ,$(REDIS_ADDR)))
REDIS_PORT      := $(word 2,$(subst :, ,$(REDIS_ADDR)))

BUILD_DIR       := /tmp/deepx/op-metal/build
HEAP_BUILD_DIR  := /tmp/deepx/heap-metal/build
IO_BUILD_DIR    := /tmp/deepx/io-metal/build
EXOP_CPU_BUILD_DIR := /tmp/deepx/exop-cpu/build
HEAP_CPU_BUILD_DIR := /tmp/deepx/heap-cpu/build
OP_BIN          := $(BUILD_DIR)/deepx-op-metal
HEAP_BIN        := $(HEAP_BUILD_DIR)/deepx-heap-metal
IO_BIN          := $(IO_BUILD_DIR)/deepx-io-metal
EXOP_CPU_BIN    := $(EXOP_CPU_BUILD_DIR)/deepx-exop-cpu
HEAP_CPU_BIN    := $(HEAP_CPU_BUILD_DIR)/deepx-heap-cpu

LOG_DIR         := /tmp/deepx-logs
OP_LOG          := $(LOG_DIR)/op-metal.log
HEAP_LOG        := $(LOG_DIR)/heap-metal.log
IO_LOG          := $(LOG_DIR)/io-metal.log
OP_PID_FILE     := $(LOG_DIR)/op-metal.pid
HEAP_PID_FILE   := $(LOG_DIR)/heap-metal.pid
IO_PID_FILE     := $(LOG_DIR)/io-metal.pid

.PHONY: all build build-op build-heap build-io \
        start-services start-op start-heap start-io \
        stop-services stop-op stop-heap stop-io \
        status check-services \
        test-unit test-integration \
        reset-redis clean clean-logs

# ═══════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════

all: build

build: build-op build-heap build-io build-exop-cpu build-heap-cpu

build-op:
	@echo "=== Building op-metal ==="
	bash op-metal/build.sh
	@echo "  → $(OP_BIN)"

build-heap:
	@echo "=== Building heap-metal ==="
	bash heap-metal/build.sh
	@echo "  → $(HEAP_BIN)"

build-io:
	@echo "=== Building io-metal ==="
	bash io-metal/build.sh
	@echo "  → $(IO_BIN)"

build-exop-cpu:
	@echo "=== Building exop-cpu ==="
	bash exop-cpu/build.sh 2>/dev/null || (mkdir -p $(EXOP_CPU_BUILD_DIR) && cd $(EXOP_CPU_BUILD_DIR) && cmake ../../exop-cpu && cmake --build . -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4))
	@echo "  → $(EXOP_CPU_BIN)"

build-heap-cpu:
	@echo "=== Building heap-cpu ==="
	bash heap-cpu/build.sh
	@echo "  → $(HEAP_CPU_BIN)"

# ═══════════════════════════════════════════════════════════════
# Service Lifecycle (daemon mode)
# ═══════════════════════════════════════════════════════════════

start-services: start-op start-heap start-io
	@echo "All services started."

start-op: check-services
	@if [ -f $(OP_PID_FILE) ] && kill -0 $$(cat $(OP_PID_FILE)) 2>/dev/null; then \
		echo "[op-metal] already running (pid=$$(cat $(OP_PID_FILE)))"; \
	else \
		mkdir -p $(LOG_DIR); \
		echo "Starting op-metal ($(REDIS_HOST):$(REDIS_PORT)) → $(OP_LOG)"; \
		$(OP_BIN) $(REDIS_HOST) $(REDIS_PORT) > $(OP_LOG) 2>&1 & \
		echo $$! > $(OP_PID_FILE); \
		sleep 1; \
		echo "[op-metal] started (pid=$$(cat $(OP_PID_FILE)))"; \
	fi

start-heap: check-services
	@if [ -f $(HEAP_PID_FILE) ] && kill -0 $$(cat $(HEAP_PID_FILE)) 2>/dev/null; then \
		echo "[heap-metal] already running (pid=$$(cat $(HEAP_PID_FILE)))"; \
	else \
		mkdir -p $(LOG_DIR); \
		echo "Starting heap-metal ($(REDIS_HOST):$(REDIS_PORT)) → $(HEAP_LOG)"; \
		$(HEAP_BIN) $(REDIS_HOST) $(REDIS_PORT) > $(HEAP_LOG) 2>&1 & \
		echo $$! > $(HEAP_PID_FILE); \
		sleep 1; \
		echo "[heap-metal] started (pid=$$(cat $(HEAP_PID_FILE)))"; \
	fi

stop-services: stop-op stop-heap stop-io
	@echo "All services stopped."

stop-op:
	@if [ -f $(OP_PID_FILE) ]; then \
		pid=$$(cat $(OP_PID_FILE)); \
		if kill -0 $$pid 2>/dev/null; then \
			echo "Stopping op-metal (pid=$$pid)"; \
			kill $$pid 2>/dev/null || true; \
			sleep 1; \
			if kill -0 $$pid 2>/dev/null; then \
				echo "Force killing op-metal"; \
				kill -9 $$pid 2>/dev/null || true; \
			fi; \
		fi; \
		rm -f $(OP_PID_FILE); \
	fi

stop-heap:
	@if [ -f $(HEAP_PID_FILE) ]; then \
		pid=$$(cat $(HEAP_PID_FILE)); \
		if kill -0 $$pid 2>/dev/null; then \
			echo "Stopping heap-metal (pid=$$pid)"; \
			kill $$pid 2>/dev/null || true; \
			sleep 1; \
			if kill -0 $$pid 2>/dev/null; then \
				echo "Force killing heap-metal"; \
				kill -9 $$pid 2>/dev/null || true; \
			fi; \
		fi; \
		rm -f $(HEAP_PID_FILE); \
	fi

start-io: check-services
	@if [ -f $(IO_PID_FILE) ] && kill -0 $$(cat $(IO_PID_FILE)) 2>/dev/null; then \
		echo "[io-metal] already running (pid=$$(cat $(IO_PID_FILE)))"; \
	else \
		mkdir -p $(LOG_DIR); \
		echo "Starting io-metal ($(REDIS_HOST):$(REDIS_PORT)) → $(IO_LOG)"; \
		$(IO_BIN) $(REDIS_HOST) $(REDIS_PORT) > $(IO_LOG) 2>&1 & \
		echo $$! > $(IO_PID_FILE); \
		sleep 1; \
		echo "[io-metal] started (pid=$$(cat $(IO_PID_FILE)))"; \
	fi

stop-io:
	@if [ -f $(IO_PID_FILE) ]; then \
		pid=$$(cat $(IO_PID_FILE)); \
		if kill -0 $$pid 2>/dev/null; then \
			echo "Stopping io-metal (pid=$$pid)"; \
			kill $$pid 2>/dev/null || true; \
			sleep 1; \
			if kill -0 $$pid 2>/dev/null; then \
				echo "Force killing io-metal"; \
				kill -9 $$pid 2>/dev/null || true; \
			fi; \
		fi; \
		rm -f $(IO_PID_FILE); \
	fi

# ── Status ──

status:
	@echo "=== deepx Executor Status ==="
	@echo "Redis: $(REDIS_ADDR)"
	@redis-cli -h $(REDIS_HOST) -p $(REDIS_PORT) PING 2>/dev/null || echo "  → NOT REACHABLE"
	@echo ""
	@if [ -f $(OP_PID_FILE) ] && kill -0 $$(cat $(OP_PID_FILE)) 2>/dev/null; then \
		echo "op-metal:    RUNNING (pid=$$(cat $(OP_PID_FILE)))"; \
	else \
		echo "op-metal:    STOPPED"; \
	fi
	@if [ -f $(HEAP_PID_FILE) ] && kill -0 $$(cat $(HEAP_PID_FILE)) 2>/dev/null; then \
		echo "heap-metal:  RUNNING (pid=$$(cat $(HEAP_PID_FILE)))"; \
	else \
		echo "heap-metal:  STOPPED"; \
	fi
	@if [ -f $(IO_PID_FILE) ] && kill -0 $$(cat $(IO_PID_FILE)) 2>/dev/null; then \
		echo "io-metal:    RUNNING (pid=$$(cat $(IO_PID_FILE)))"; \
	else \
		echo "io-metal:    STOPPED"; \
	fi

check-services:
	@redis-cli -h $(REDIS_HOST) -p $(REDIS_PORT) PING > /dev/null 2>&1 || \
		(echo "ERROR: Redis not reachable at $(REDIS_ADDR)" && exit 1)

# ═══════════════════════════════════════════════════════════════
# Testing
# ═══════════════════════════════════════════════════════════════

# Unit tests (no Redis needed)
test-unit:
	cd vm && go test -count=1 ./...

# Integration tests (Redis needed, no plats)
test-integration:
	cd vm && REDIS_ADDR=$(REDIS_ADDR) go test -tags=integration -count=1 -v -run 'TestIntegration' ./...

# ═══════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════

# all-in-one: build → start → reset → stop
pipeline: build start-services reset-redis stop-services
	@echo "=== Pipeline complete ==="

# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

reset-redis:
	@echo "Resetting Redis at $(REDIS_ADDR)..."
	redis-cli -h $(REDIS_HOST) -p $(REDIS_PORT) FLUSHDB

clean:
	rm -rf $(BUILD_DIR) $(HEAP_BUILD_DIR) $(IO_BUILD_DIR) $(EXOP_CPU_BUILD_DIR) $(HEAP_CPU_BUILD_DIR)
	cd vm && go clean -testcache

clean-logs:
	rm -rf $(LOG_DIR)

# Show logs
logs-op:
	tail -f $(OP_LOG)

logs-heap:
	tail -f $(HEAP_LOG)

logs-io:
	tail -f $(IO_LOG)

help:
	@echo "deepx Executor Makefile"
	@echo ""
	@echo "BUILD:"
	@echo "  make build            Build all executors"
	@echo "  make build-op         Build op-metal only"
	@echo "  make build-heap       Build heap-metal only"
	@echo "  make build-io         Build io-metal only"
	@echo "  make build-exop-cpu   Build exop-cpu (CPU extended ops)"
	@echo "  make build-heap-cpu   Build heap-cpu (CPU heap manager)"
	@echo ""
	@echo "SERVICES (daemon):"
	@echo "  make start-services   Start op-metal + heap-metal + io-metal in background"
	@echo "  make stop-services    Stop all services"
	@echo "  make status           Check service status"
	@echo ""
	@echo "TESTING:"
	@echo "  make test-unit        Run VM unit tests (no Redis)"
	@echo "  make test-integration Run VM integration tests (Redis, no plats)"
	@echo ""
	@echo "PIPELINE:"
	@echo "  make pipeline         Full cycle: build → start → reset → stop"
	@echo ""
	@echo "UTILS:"
	@echo "  make reset-redis      FLUSHDB (clean test state)"
	@echo "  make clean            Remove build artifacts"
	@echo "  make logs-op          Tail op-metal logs"
	@echo "  make logs-heap        Tail heap-metal logs"
	@echo "  make logs-io          Tail io-metal logs"
	@echo ""
	@echo "Config via env:"
	@echo "  REDIS_ADDR=$(REDIS_ADDR)"
	@echo "  OP_BIN=$(OP_BIN)"
	@echo "  HEAP_BIN=$(HEAP_BIN)"
	@echo "  IO_BIN=$(IO_BIN)"
