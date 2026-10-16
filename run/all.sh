#!/usr/bin/env bash
set -euo pipefail
dir="$(dirname "$0")"
bash "$dir/collect.sh"
bash "$dir/distance.sh"
bash "$dir/separability.sh"
bash "$dir/gradient.sh"
