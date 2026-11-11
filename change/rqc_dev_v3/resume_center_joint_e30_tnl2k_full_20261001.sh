#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
python_bin=/root/miniconda3/envs/uvltrack/bin/python
config=baseline_base_head_output_center_safe_joint_bs16_30ep_20260930
checkpoint="change/center_joint_training_exact_metadata_20260930/${config}/CenterJoint_ep0030.pth.tar"
test_log=terminal_logs/test_center_joint_ep30_tnl2k_FULL_run100130_resume4_20261001.log
eval_log=terminal_logs/eval_center_joint_ep30_tnl2k_FULL_run100130_resume4_20261001.log
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$python_bin" - <<'PY'
import json,pathlib,sys,torch
root=pathlib.Path.cwd();base=root/'change/rqc_dev_v3';sys.path.insert(0,str(base))
from center_joint_contract_20260930 import NAME,file_hash
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from test_center_joint_20260930 import verify_outputs
report=json.loads((base/'center_joint_lifecycle_check_20261001.json').read_text())
assert report['boxes_state_prompt_exact'] and report['prediction_frames']==46
assert report['retention']['safe']['retained_networks']==0
assert not report['model_prediction_changed'] and report['optimizer_steps']==0
checkpoint=root/'change/center_joint_training_exact_metadata_20260930'/NAME/'CenterJoint_ep0030.pth.tar'
saved=torch.load(str(checkpoint),map_location='cpu')
assert saved['epoch']==30 and saved['variant']=='center_safe_joint' and 'head' in saved
assert not saved.get('synthetic_check')
assert file_hash(root/'experiments/uvltrack'/(NAME+'.yaml'))==saved['identity']['config_sha256']
assert file_hash(pathlib.Path(saved['identity']['source_checkpoint']))==saved['identity']['source_checkpoint_sha256']
for name,digest in saved['identity']['source_files'].items():assert file_hash(base/name)==digest,name
tracker=Tracker('uvltrack',NAME,'tnl2k',run_id=100130,eval=True)
result=pathlib.Path(tracker.results_dir)
manifest=json.loads((result/'center_joint_run_manifest.json').read_text())
expected=dict(config=NAME,dataset='tnl2k',epoch=30,runid=100130,checkpoint=str(checkpoint),
    checkpoint_sha256=file_hash(checkpoint),identity=saved['identity'])
assert manifest==expected,'Existing run/checkpoint identity mismatch'
dataset=get_dataset('tnl2k');assert len(dataset)==700
completed=verify_outputs(dataset,tracker,30,require_all=False)
assert 0<completed<700,'Not a partially completed run'
snapshot={}
for seq in dataset:
    bbox=result/'tnl2k_NLBBOX_030'/(seq.name+'.txt')
    if not bbox.exists():continue
    for path in (bbox,bbox.with_name(seq.name+'_time.txt')):
        if path.exists():snapshot[str(path)]=dict(sha256=file_hash(path),mtime_ns=path.stat().st_mtime_ns)
sidecar=result/'resume_lifecycle_20261001.json'
assert not sidecar.exists(),'Resume already launched; refuse duplicate'
destination=root/'change/center_joint_evaluation_20260930'/f'{NAME}_tnl2k_e0030_run100130'
assert not destination.exists(),'Existing report must not be overwritten'
for name in ('test_center_joint_ep30_tnl2k_FULL_run100130_resume4_20261001.log',
             'eval_center_joint_ep30_tnl2k_FULL_run100130_resume4_20261001.log'):
    assert not (root/'terminal_logs'/name).exists(),'Existing log '+name
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    for arg in args[:4]:
        name=pathlib.Path(arg.decode(errors='ignore')).name
        assert name not in ('test_center_joint_20260930.py','test_center_joint_lifecycle_safe_20261001.py',
            'check_center_joint_lifecycle_20261001.py','train_center_joint_20260930.py','test.py','train.py'),(p.name,name)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
record=dict(checkpoint_sha256=manifest['checkpoint_sha256'],completed_before=completed,
    existing_files=snapshot,workers=4,model_prediction_changed=False,
    adapter_sources={name:file_hash(base/name) for name in (
        'center_joint_lifecycle_20261001.py','test_center_joint_lifecycle_safe_20261001.py')})
with sidecar.open('x') as out:json.dump(record,out,indent=2)
print('RESUME_PREFLIGHT_PASS',json.dumps(dict(completed=completed,missing=700-completed,
    threads=4,runid=100130,checkpoint=str(checkpoint),results=str(result))),flush=True)
PY
mkdir -p terminal_logs
(set -o noclobber; : > "$test_log"; : > "$eval_log")
trap 'status=$?; printf "%s RESUME_FAILED status=%s test_log=%s eval_log=%s\n" "$(date +%FT%T%z)" "$status" "$test_log" "$eval_log"; exit "$status"' ERR
printf '%s RESUME_TEST_START runid=100130 workers=4 num_gpus=1\n' "$(date +%FT%T%z)"
"$python_bin" change/rqc_dev_v3/test_center_joint_lifecycle_safe_20261001.py \
  --dataset tnl2k --checkpoint "$checkpoint" --epoch 30 --runid 100130 --threads 4 --resume-results \
  >> "$test_log" 2>&1
"$python_bin" - <<'PY'
import json,pathlib,sys
base=pathlib.Path.cwd()/'change/rqc_dev_v3';sys.path.insert(0,str(base))
from center_joint_contract_20260930 import NAME,file_hash
from lib.test.evaluation.tracker import Tracker
result=pathlib.Path(Tracker('uvltrack',NAME,'tnl2k',run_id=100130,eval=True).results_dir)
saved=json.loads((result/'resume_lifecycle_20261001.json').read_text())
for path,expected in saved['existing_files'].items():
    p=pathlib.Path(path)
    assert file_hash(p)==expected['sha256'] and p.stat().st_mtime_ns==expected['mtime_ns'],'Old result changed: '+path
print('EXISTING_RESULTS_PRESERVED',saved['completed_before'],flush=True)
PY
printf '%s TEST_FINISHED EVALUATION_START\n' "$(date +%FT%T%z)"
"$python_bin" change/rqc_dev_v3/evaluate_center_joint_20260930.py \
  --dataset tnl2k --epoch 30 --runid 100130 >> "$eval_log" 2>&1
printf '%s PIPELINE_COMPLETE test_log=%s eval_log=%s\n' "$(date +%FT%T%z)" "$test_log" "$eval_log"
