#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
[ -f "$DATA" ] || bash "$(dirname "$0")/collect.sh"
python scripts/distance.py --data "$DATA" --n-fixed 5 --l-fixed "${LENGTH:-1024}" --out "$OUT/$NAME"
python scripts/critical_n.py --data "$DATA" --alpha "${ALPHA:-0.01}" --out "$OUT/$NAME"
