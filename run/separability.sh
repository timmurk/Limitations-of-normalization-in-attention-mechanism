#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
[ -f "$DATA" ] || bash "$(dirname "$0")/collect.sh"
python scripts/separability.py --data "$DATA" --length "${LENGTH:-1024}" --out "$OUT/$NAME"
