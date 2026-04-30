#!/usr/bin/env bash
set -euo pipefail
exec ./executor/vm/build.sh "$@"
# Builds: /tmp/deepx-vm/vm + /tmp/deepx-vm/loader
