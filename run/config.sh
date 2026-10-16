export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}"
MODEL=${MODEL:-gpt2}
OUT=${OUT:-outputs}
DEVICE=${DEVICE:-cpu}
TEXT_ARG=""
if [ -n "${TEXT:-}" ]; then TEXT_ARG="--text $TEXT"; fi
NAME=$(basename "$MODEL")
DATA="$OUT/collect_${NAME}.npz"
