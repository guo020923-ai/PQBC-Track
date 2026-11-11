#!/usr/bin/env bash
set -euo pipefail
project_root="${PQBC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$project_root"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for argfile in /proc/[0-9]*/cmdline; do
 if [[ ! -r "$argfile" ]]; then continue; fi
 if tr '\0' '\n' < "$argfile" 2>/dev/null | grep -Eq '(^|/)(tracking/(train|test)\.py|train_directional_center_joint_20261001\.py|test_directional_center_joint_20261001\.py)$'; then
  printf 'Refusing concurrent train/test: %s\n' "$argfile"
  exit 1
 fi
done
python_bin="${PYTHON_BIN:-python}"
script=change/rqc_dev_v3/audit_directional_center_train_same_state_20261002.py
"$python_bin" -m py_compile "$script"
printf '%s TRAIN_DOMAIN_SAME_STATE_START workers=8 training=false\n' "$(date +%FT%T%z)"
"$python_bin" "$script" --output change/directional_center_train_same_state_20261002 --workers 8 --per-domain 20 --frames 512
printf '%s TRAIN_DOMAIN_SAME_STATE_FINISHED training=false\n' "$(date +%FT%T%z)"
