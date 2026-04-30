# ═══════════════════════════════════════════════════════════════
# deepx Executor — Build & Integration Test Lifecycle Management
# ═══════════════════════════════════════════════════════════════

# ── Config ──
REDIS_ADDR      ?= 127.0.0.1:16379
REDIS_HOST      := $(word 1,$(subst :, ,$(REDIS_ADDR)))
REDIS_PORT      := $(word 2,$(subst :, ,$(REDIS_ADDR)))

BUILD_DIR       := build
HEAP_BUILD_DIR  := build-heap
OP_BIN          := $(BUILD_DIR)/deepx-op-metal
HEAP_BIN        := $(HEAP_BUILD_DIR)/deepx-heap-metal

LOG_DIR         := /tmp/deepx-logs
OP_LOG          := $(LOG_DIR)/op-metal.log
HEAP_LOG        := $(LOG_DIR)/heap-metal.log
OP_PID_FILE     := $(LOG_DIR)/op-metal.pid
HEAP_PID_FILE   := $(LOG_DIR)/heap-metal.pid

.PHONY: all build build-op build-heap \
        start-services start-op start-heap \
        stop-services stop-op stop-heap \
        status check-services \
        test-unit test-integration test-real \
        reset-redis clean clean-logs

# ═══════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════

all: build

build: build-op build-heap

build-op:
	@echo "=== Building op-metal ==="
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake ../op-metal && $(MAKE) -j$$(sysctl -n hw.ncpu 2>/dev/null || nproc)
	@echo "  → $(OP_BIN)"

build-heap:
	@echo "=== Building heap-metal ==="
	mkdir -p $(HEAP_BUILD_DIR) && cd $(HEAP_BUILD_DIR) && cmake ../heap-metal && $(MAKE) -j$$(sysctl -n hw.ncpu 2>/dev/null || nproc)
	@echo "  → $(HEAP_BIN)"

# ═══════════════════════════════════════════════════════════════
# Service Lifecycle (daemon mode)
# ═══════════════════════════════════════════════════════════════

start-services: start-op start-heap
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

stop-services: stop-op stop-heap
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

# Real integration tests (Redis + heap-metal + op-metal binariess)
# Prerequisites: start-services (or running plats), built binaries
test-real:
	cd vm && \
		REDIS_ADDR=$(REDIS_ADDR) \
		HEAP_METAL_BIN=$(realpath $(HEAP_BIN)) \
		OP_METAL_BIN=$(realpath $(OP_BIN)) \
		go test -tags=integration_real -count=1 -v -run 'TestIntegrationReal' ./...

# ═══════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════

# all-in-one: build → start → reset → test → stop
pipeline: build start-services reset-redis test-real stop-services
	@echo "=== Pipeline complete ==="

# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

reset-redis:
	@echo "Resetting Redis at $(REDIS_ADDR)..."
	redis-cli -h $(REDIS_HOST) -p $(REDIS_PORT) FLUSHDB

clean:
	rm -rf $(BUILD_DIR) $(HEAP_BUILD_DIR)
	cd vm && go clean -testcache

clean-logs:
	rm -rf $(LOG_DIR)

# Show op-metal / heap-metal logs
logs-op:
	tail -f $(OP_LOG)

logs-heap:
	tail -f $(HEAP_LOG)

help:
	@echo "deepx Executor Makefile"
	@echo ""
	@echo "BUILD:"
	@echo "  make build            Build op-metal + heap-metal"
	@echo "  make build-op         Build op-metal only"
	@echo "  make build-heap       Build heap-metal only"
	@echo ""
	@echo "SERVICES (daemon):"
	@echo "  make start-services   Start op-metal + heap-metal in background"
	@echo "  make stop-services    Stop all services"
	@echo "  make status           Check service status"
	@echo ""
	@echo "TESTING:"
	@echo "  make test-unit        Run VM unit tests (no Redis)"
	@echo "  make test-integration Run VM integration tests (Redis, no plats)"
	@echo "  make test-real        Run real integration tests (Redis + plats)"
	@echo ""
	@echo "PIPELINE:"
	@echo "  make pipeline         Full cycle: build → start → reset → test → stop"
	@echo ""
	@echo "UTILS:"
	@echo "  make reset-redis      FLUSHDB (clean test state)"
	@echo "  make clean            Remove build artifacts"
	@echo "  make logs-op          Tail op-metal logs"
	@echo "  make logs-heap        Tail heap-metal logs"
	@echo ""
	@echo "Config via env:"
	@echo "  REDIS_ADDR=$(REDIS_ADDR)"
	@echo "  OP_BIN=$(OP_BIN)"
	@echo "  HEAP_BIN=$(HEAP_BIN)"
