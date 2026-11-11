#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
base=/root/autodl-tmp/UVLTrack-master
config=baseline_base_head_rqc_jq_bs16_30ep_20260929
py=/root/miniconda3/envs/uvltrack/bin/python
testlog="$base/terminal_logs/test_${config}_ep30_tnl2k_FULL_run92931_resume20260930.log"
evallog="$base/terminal_logs/eval_${config}_ep30_tnl2k_FULL_run92931_resume20260930.log"
[[ ! -e "$testlog" && ! -e "$evallog" ]] || { echo 'REFUSE existing log'; exit 1; }
if pgrep -f '[t]racking/test.py|[t]racking/train.py' >/dev/null; then echo 'REFUSE active experiment'; exit 1; fi
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
export MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$py" change/rqc_dev_v3/validate_jq_results_20260930.py
set -o noclobber
echo "$(date -Is) RESUME_TEST"
"$py" tracking/test.py --tracker_name uvltrack --tracker_param "$config" --dataset_name tnl2k --runid 92931 --threads 8 --num_gpus 1 --debug 0 > "$testlog" 2>&1
"$py" change/rqc_dev_v3/validate_jq_results_20260930.py --complete
echo "$(date -Is) EVAL_START"
"$py" change/experiment_tools/evaluate_checkpoint_run.py --tracker-name uvltrack --tracker-param "$config" --dataset tnl2k --report-name tnl2k_NLBBOX_030 --runid 92931 --display-name JQ-E30-FULL > "$evallog" 2>&1
echo "$(date -Is) ALL_COMPLETE"
