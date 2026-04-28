#!/usr/bin/env bash
# Reset Redis for VM testing — delete all /vthread/, /src/func/, op-plat, done keys.
# Usage: /reset-redis [redis_addr]
#   Default addr: 127.0.0.1:16379 (override with REDIS_ADDR env or arg)

set -euo pipefail

ADDR="${1:-${REDIS_ADDR:-127.0.0.1:16379}}"
HOST="${ADDR%:*}"
PORT="${ADDR##*:}"

REDIS_CLI="redis-cli"
if ! command -v "$REDIS_CLI" &>/dev/null; then
  # try common locations
  for p in /opt/homebrew/bin/redis-cli /usr/local/bin/redis-cli; do
    if [ -x "$p" ]; then REDIS_CLI="$p"; break; fi
  done
fi

echo "=== Reset Redis at $ADDR ==="

# FLUSHDB is simplest and cleanest for test environments
"$REDIS_CLI" -h "$HOST" -p "$PORT" FLUSHDB 2>&1 || {
  echo "FLUSHDB failed, trying key-by-key delete..."
  for pattern in '/vthread/*' '/src/func/*' '/op/*' '/sys/*' 'done:*' 'cmd:*' 'notify:vm'; do
    "$REDIS_CLI" -h "$HOST" -p "$PORT" --raw KEYS "$pattern" 2>/dev/null | \
      xargs -r "$REDIS_CLI" -h "$HOST" -p "$PORT" DEL 2>/dev/null || true
  done
}

echo "Redis reset done."
"$REDIS_CLI" -h "$HOST" -p "$PORT" DBSIZE
