# ═══════════════════════════════════════════════════════════════
# dashboard — Web 可视化面板构建
# ═══════════════════════════════════════════════════════════════
#
#   make build     构建前端 + 后端 → /tmp/deepx-dashboard
#   make preview   构建并启动 (需先 deepxctl boot)
#   make dev       开发模式 (Vite HMR + Go 后端)
#   make vet       类型检查 (Go vet + tsc)
#   make clean     清理构建产物
#

# ── 路径 ──
ROOT        := $(shell git -C $(dir $(realpath $(firstword $(MAKEFILE_LIST)))) rev-parse --show-toplevel 2>/dev/null || pwd)
FRONTEND    := $(ROOT)/tool/dashboard/frontend
OUT_BIN     := /tmp/deepx-dashboard

# ── 工具 ──
GO          := $(or $(shell which go),$(HOME)/sdk/go/bin/go)
NPM         := $(or $(shell which npm),$(HOME)/.nvm/versions/node/*/bin/npm)
NPX         := node_modules/.bin
TSC         := $(FRONTEND)/node_modules/typescript/bin/tsc
VITE        := $(FRONTEND)/node_modules/vite/bin/vite.js

# ── 标志 ──
REDIS_ADDR  ?= 127.0.0.1:16379
LOADER_BIN  ?= /tmp/deepx-vm/loader
HTTP_ADDR   ?= :8080

.PHONY: help build preview dev vet clean

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── 构建 ──

build: vendor build-frontend build-backend ## 完整构建 (前端 + 后端)
	@echo "  ✓  output: $(OUT_BIN)"
	@ls -lh $(OUT_BIN)

vendor: ## 安装依赖
	@echo "── npm install ──"
	cd $(FRONTEND) && $(NPM) install --no-bin-links --silent
	@echo "── go mod tidy ──"
	cd $(ROOT)/tool/dashboard && $(GO) mod tidy

build-frontend: ## 构建 React 前端 → frontend/dist/
	@echo "── tsc + vite build ──"
	cd $(FRONTEND) && $(TSC) && $(VITE) build
	@echo "  ✓  frontend built → frontend/dist/"

build-backend: ## 构建 Go 后端 → /tmp/deepx-dashboard
	@echo "── go build ──"
	cd $(ROOT)/tool/dashboard && $(GO) build -o $(OUT_BIN) .
	@echo "  ✓  backend built → $(OUT_BIN)"

# ── 开发 ──

preview: build ## 构建并启动 (生产模式)
	@echo ""
	@echo "  ┌──────────────────────────────────────────┐"
	@echo "  │  dashboard  http://localhost$(HTTP_ADDR)     │"
	@echo "  │  redis       $(REDIS_ADDR)                 │"
	@echo "  └──────────────────────────────────────────┘"
	@echo ""
	$(OUT_BIN) -addr=$(HTTP_ADDR) -redis=$(REDIS_ADDR) -loader=$(LOADER_BIN)

dev: vendor ## 开发模式 (Go 后端 + Vite HMR，需要两个终端)
	@echo "── 启动 Go 后端 (terminal 1) ──"
	cd $(ROOT)/tool/dashboard && $(GO) run . -addr=$(HTTP_ADDR) -redis=$(REDIS_ADDR) -loader=$(LOADER_BIN) &
	@sleep 1
	@echo "── 启动 Vite dev server (terminal 2) ──"
	cd $(FRONTEND) && $(NPM) run dev

# ── 检查 ──

vet: ## 类型检查 (Go vet + tsc --noEmit)
	@echo "── go vet ──"
	cd $(ROOT)/tool/dashboard && $(GO) vet ./...
	@echo "── tsc --noEmit ──"
	cd $(FRONTEND) && $(TSC) --noEmit
	@echo "  ✓  all checks passed"

# ── 清理 ──

clean: ## 清理构建产物
	rm -f $(OUT_BIN)
	rm -rf $(FRONTEND)/dist
	@echo "  ✓  cleaned"
