#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
python scripts/gradient.py --model "$MODEL" $TEXT_ARG --length "${LENGTH:-1024}" \
  --temps 1e-4,1e-3,1e-2,1e-1,1,10,100 --eps 1e-3,1e-1,10 --device "$DEVICE" --out "$OUT/$NAME"
