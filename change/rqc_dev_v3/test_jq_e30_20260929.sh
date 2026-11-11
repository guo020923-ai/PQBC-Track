#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
base=/root/autodl-tmp/UVLTrack-master
config=baseline_base_head_rqc_jq_bs16_30ep_20260929
runid=92931
py=/root/miniconda3/envs/uvltrack/bin/python
testlog="$base/terminal_logs/test_${config}_ep30_tnl2k_FULL_run${runid}.log"
evallog="$base/terminal_logs/eval_${config}_ep30_tnl2k_FULL_run${runid}.log"
cd "$root"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
export MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for target in "$testlog" "$evallog" "$root/test/tracking_results/uvltrack/${config}_${runid}"; do
  [[ ! -e "$target" ]] || { echo "REFUSE existing $target"; exit 1; }
done
"$py" - <<'PY'
from pathlib import Path
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
t=Tracker('uvltrack','baseline_base_head_rqc_jq_bs16_30ep_20260929','tnl2k',run_id=92931)
assert t.params.cfg.TEST.MODE=='NLBBOX' and t.params.cfg.TEST.EPOCH==30
assert t.params.cfg.MODEL.RQC.ENABLED and not t.params.cfg.MODEL.HEAD.ERC.ENABLED
assert t.params.cfg.MODEL.RQC.DESIGN=='candidate_quality_jq'
assert Path(t.params.checkpoint).is_file()
assert len(get_dataset('tnl2k'))==700
print('PREFLIGHT',t.params.checkpoint,t.results_dir,flush=True)
PY
set -o noclobber
echo "$(date -Is) TEST_START"
"$py" tracking/test.py --tracker_name uvltrack --tracker_param "$config" --dataset_name tnl2k --runid "$runid" --threads 8 --num_gpus 1 --debug 0 > "$testlog" 2>&1
"$py" - <<'PY'
import numpy as np
from pathlib import Path
from lib.test.evaluation import get_dataset
root=Path('test/tracking_results/uvltrack/baseline_base_head_rqc_jq_bs16_30ep_20260929_92931/tnl2k_NLBBOX_030')
dataset=get_dataset('tnl2k')
for s in dataset:
    p=root/(s.name+'.txt')
    a=np.loadtxt(str(p),delimiter='\t',ndmin=2)
    assert a.shape==(len(s.frames),4) and np.isfinite(a).all(),str(p)
print('COMPLETE 700/700: finite boxes and frame counts verified',flush=True)
PY
echo "$(date -Is) EVAL_START"
"$py" change/experiment_tools/evaluate_checkpoint_run.py --tracker-name uvltrack --tracker-param "$config" --dataset tnl2k --report-name tnl2k_NLBBOX_030 --runid "$runid" --display-name JQ-E30-FULL > "$evallog" 2>&1
echo "$(date -Is) ALL_COMPLETE"
