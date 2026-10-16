#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
python scripts/collect.py --model "$MODEL" $TEXT_ARG --length "${LENGTH:-1024}" --chunks "${CHUNKS:-1}" \
  --vectors "${VECTORS:-value}" --device "$DEVICE" --out "$DATA"
