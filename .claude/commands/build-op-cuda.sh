#!/usr/bin/env bash
set -euo pipefail
exec ./executor/op-cuda/build.sh "$@"
