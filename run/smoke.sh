#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
o="$OUT/smoke"
python tests/test_metrics.py
python scripts/collect.py --model "$MODEL" $TEXT_ARG --length 128 --l-values 32,64,100,128 --n-values 1,2,4,5,8,10,16,20 \
  --device "$DEVICE" --out "$o/collect.npz"
python scripts/distance.py --data "$o/collect.npz" --l-fixed 128 --l-list 32,64,128 --n-list 1,5,10,20 --out "$o"
python scripts/critical_n.py --data "$o/collect.npz" --l-list 64,100,128 --out "$o"
python scripts/separability.py --data "$o/collect.npz" --length 128 --n-list 1,2,4,8,16 --out "$o"
python scripts/gradient.py --model "$MODEL" $TEXT_ARG --length 128 --temps 1e-2,1,100 --eps 1e-1 --device "$DEVICE" --out "$o"
