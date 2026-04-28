#!/usr/bin/env bash
set -euo pipefail
exec ./executor/heapmem-metal/build.sh "$@"
