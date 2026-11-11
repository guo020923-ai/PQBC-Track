#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/UVLTrack-master
root="$base/change/rqc_jq_20260929_isolated"
config=baseline_base_head_rqc_jq_bs16_30ep_20260929
log="$base/terminal_logs/train_uvltrack_${config}.log"
grep -q JQ_CHECK_PASS "$root/change/rqc_dev_v3/jq_real_check.log"
grep -q DEPLOYED_EQUIVALENCE_PASS "$root/change/rqc_dev_v3/jq_real_check.log"
for target in "$log" "$base/logs/uvltrack-${config}.log" "$base/checkpoints/train/uvltrack/$config"; do
  [[ ! -e "$target" ]] || { echo "REFUSE existing $target"; exit 1; }
done
if pgrep -f '[l]ib/train/run_training.py|[t]racking/train.py|[c]heck_jq_real_20260929.py|[t]racking/test.py' >/dev/null; then
  echo 'Active experiment: refuse launch'; exit 1
fi
cd "$root"
test -f "$base/checkpoints/train/uvltrack/baseline_base/UVLTrack_ep0300.pth.tar"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
export MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset LOCAL_RANK RANK WORLD_SIZE
set -o noclobber
nohup /root/miniconda3/envs/uvltrack/bin/python tracking/train.py --script uvltrack --config "$config" --save_dir "$base" --mode single > "$log" 2>&1 < /dev/null &
echo "$(date -Is) PID=$! LOG=$log"
