#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec make -C "$PROJECT_ROOT" reset-redis REDIS_ADDR="${1:-${REDIS_ADDR:-127.0.0.1:16379}}"
