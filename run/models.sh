#!/usr/bin/env bash
set -euo pipefail
for m in gpt2 gpt2-medium gpt2-large; do
  MODEL=$m bash "$(dirname "$0")/all.sh"
done
