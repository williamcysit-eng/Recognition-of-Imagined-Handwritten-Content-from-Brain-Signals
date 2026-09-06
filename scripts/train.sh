#!/bin/sh
set -eu

seed="${1:-42}"
epochs="${2:-150}"
run_id="${3:-loop-eval-seed-${seed}}"

if [ -n "${PYTHON:-}" ]; then
    python_executable="$PYTHON"
elif [ -x .venv/bin/python ]; then
    python_executable=.venv/bin/python
else
    python_executable=python3
fi

exec "$python_executable" src/train.py \
    --model ensemble \
    --development-only \
    --seed "$seed" \
    --epochs "$epochs" \
    --run-id "$run_id"
