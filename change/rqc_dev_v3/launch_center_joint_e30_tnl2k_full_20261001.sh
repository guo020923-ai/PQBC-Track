#!/usr/bin/env bash
# FULL only; official reader/runner/writer/metrics via independent center adapter.
set -euo pipefail
cd /root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
python_bin=/root/miniconda3/envs/uvltrack/bin/python
config=baseline_base_head_output_center_safe_joint_bs16_30ep_20260930
checkpoint="change/center_joint_training_exact_metadata_20260930/${config}/CenterJoint_ep0030.pth.tar"
runid=100130
test_log="terminal_logs/test_center_joint_ep30_tnl2k_FULL_run${runid}_20261001.log"
eval_log="terminal_logs/eval_center_joint_ep30_tnl2k_FULL_run${runid}_20261001.log"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$python_bin" - <<'PY'
import json,pathlib,sys,subprocess
import numpy as np
import torch,yaml
root=pathlib.Path.cwd();base=root/'change/rqc_dev_v3';sys.path.insert(0,str(base))
from center_joint_contract_20260930 import NAME,file_hash,register_config
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from center_joint_tracker_20260930 import JointCenterEvaluationTracker
register_config()
checkpoint=root/'change/center_joint_training_exact_metadata_20260930'/NAME/'CenterJoint_ep0030.pth.tar'
saved=torch.load(str(checkpoint),map_location='cpu')
assert saved.get('variant')=='center_safe_joint' and not saved.get('synthetic_check')
assert saved['epoch']==30 and 'head' in saved and 'optimizer' in saved
identity=saved['identity'];assert identity['config']==NAME
config=root/'experiments/uvltrack'/(NAME+'.yaml')
assert file_hash(config)==identity['config_sha256']
assert yaml.safe_load(config.read_text())['TEST']==dict(MODE='NLBBOX',EPOCH=30,
    SEARCH_FACTOR=4.0,SEARCH_SIZE=256,TEMPLATE_FACTOR=2.0,TEMPLATE_SIZE=128,UPDATE_INTERVAL=20)
assert file_hash(pathlib.Path(identity['source_checkpoint']))==identity['source_checkpoint_sha256']
for name,digest in identity['source_files'].items():assert file_hash(base/name)==digest,name
records=[json.loads(v) for v in (checkpoint.parent/'stats.jsonl').read_text().splitlines() if v.strip()]
assert [r['epoch'] for r in records]==list(range(1,31))
assert json.loads((checkpoint.parent/'protocol.json').read_text())==identity
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert all(torch.isfinite(t).all() for t in saved['head'].values())
assert all(torch.isfinite(t).all() for t in saved['model'].values())
dataset=get_dataset('tnl2k');assert len(dataset)==700
assert len(set(s.name for s in dataset))==700
frames=0
for seq in dataset:
    assert len(seq.frames)==len(seq.ground_truth_rect) and len(seq.frames)>0,seq.name
    assert isinstance(seq.language,str) and seq.language.strip(),seq.name
    assert pathlib.Path(seq.frames[0]).is_file() and pathlib.Path(seq.frames[-1]).is_file(),seq.name
    assert np.isfinite(seq.ground_truth_rect[0]).all() and (seq.ground_truth_rect[0,2:]>0).all(),seq.name
    frames+=len(seq.frames)
tracker=JointCenterEvaluationTracker('tnl2k',checkpoint,100130,epoch=30)
params=tracker.params
assert params.cfg.TEST.MODE=='NLBBOX' and params.cfg.TEST.EPOCH==30
assert pathlib.Path(params.center_checkpoint)==checkpoint
assert params.checkpoint==identity['source_checkpoint']
assert not params.cfg.MODEL.HEAD.ERC.ENABLED and not params.cfg.MODEL.RQC.ENABLED
result=pathlib.Path(tracker.results_dir)
assert not result.exists(),'Refuse to overwrite '+str(result)
report=root/'change/center_joint_evaluation_20260930'/f'{NAME}_tnl2k_e0030_run100130'
assert not report.exists(),'Refuse to overwrite report '+str(report)
for name in ('test_center_joint_ep30_tnl2k_FULL_run100130_20261001.log',
             'eval_center_joint_ep30_tnl2k_FULL_run100130_20261001.log'):
    assert not (root/'terminal_logs'/name).exists(),'Refuse to overwrite log '+name
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    for arg in args[:4]:
        name=pathlib.Path(arg.decode(errors='ignore')).name
        assert name not in ('train_center_joint_20260930.py','test_center_joint_20260930.py',
            'evaluate_center_joint_20260930.py','train.py','test.py'),(p.name,name)
print('TEST_PREFLIGHT_PASS',json.dumps(dict(config=NAME,epoch=30,dataset='tnl2k',mode='NLBBOX',
    sequences=len(dataset),frames=frames,runid=100130,workers=8,num_gpus=1,
    checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),results=str(result),report=str(report))),flush=True)
PY
if [[ "${1:-}" != "--run" ]]; then
  printf 'Preflight only; pass --run to test then evaluate.\n'
  exit 0
fi
mkdir -p terminal_logs
(set -o noclobber; : > "$test_log"; : > "$eval_log")
trap 'status=$?; printf "%s PIPELINE_FAILED status=%s test_log=%s eval_log=%s\n" "$(date +%FT%T%z)" "$status" "$test_log" "$eval_log"; exit "$status"' ERR
printf '%s TEST_START checkpoint=%s runid=%s workers=8 num_gpus=1\n' "$(date +%FT%T%z)" "$checkpoint" "$runid"
"$python_bin" change/rqc_dev_v3/test_center_joint_20260930.py \
  --dataset tnl2k --checkpoint "$checkpoint" --epoch 30 --runid "$runid" --threads 8 \
  >> "$test_log" 2>&1
printf '%s TEST_FINISHED EVALUATION_START\n' "$(date +%FT%T%z)"
"$python_bin" change/rqc_dev_v3/evaluate_center_joint_20260930.py \
  --dataset tnl2k --epoch 30 --runid "$runid" >> "$eval_log" 2>&1
printf '%s PIPELINE_COMPLETE test_log=%s eval_log=%s\n' "$(date +%FT%T%z)" "$test_log" "$eval_log"
