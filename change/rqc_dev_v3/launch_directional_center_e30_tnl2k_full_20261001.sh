#!/usr/bin/env bash
set -euo pipefail
project_root="${PQBC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$project_root"
python_bin="${PYTHON_BIN:-python}"
config=baseline_base_head_directional_center_guard_bs16_30ep_20261001
checkpoint="change/directional_center_joint_training_20261001/${config}/DirectionalCenterJoint_ep0030.pth.tar"
runid=101230
test_log=terminal_logs/test_directional_center_ep30_tnl2k_FULL_run101230_20261001.log
eval_log=terminal_logs/eval_directional_center_ep30_tnl2k_FULL_run101230_20261001.log
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$python_bin" - <<'PY'
import pathlib,json,sys,torch
root=pathlib.Path.cwd();base=root/'change/rqc_dev_v3';sys.path.insert(0,str(base))
from directional_center_joint_contract_20261001 import NAME,file_hash
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
checkpoint=root/'change/directional_center_joint_training_20261001'/NAME/'DirectionalCenterJoint_ep0030.pth.tar'
saved=torch.load(str(checkpoint),map_location='cpu')
assert saved['variant']=='directional_center_guard_joint' and saved['epoch']==30 and not saved.get('synthetic_check')
assert 'head' in saved and 'model' in saved and 'optimizer' in saved
identity=saved['identity'];assert identity['config']==NAME
assert identity['epochs']==30 and identity['head_trainable']>0 and identity['backbone_trainable']==0
assert identity['refiner_trainable']==31808 and identity['native_decision_box_weight']==.5 and identity['native_decision_rank_weight']==.25
assert file_hash(root/'experiments/uvltrack'/(NAME+'.yaml'))==identity['config_sha256']
assert file_hash(identity['source_checkpoint'])==identity['source_checkpoint_sha256']
for name,digest in identity['source_files'].items():assert file_hash(base/name)==digest,name
dest=checkpoint.parent
assert json.loads((dest/'protocol.json').read_text())==identity
rows=[json.loads(s) for s in (dest/'stats.jsonl').read_text().splitlines() if s.strip()]
assert [r['epoch'] for r in rows]==list(range(1,31))
dataset=get_dataset('tnl2k');assert len(dataset)==700
result=pathlib.Path(Tracker('uvltrack',NAME,'tnl2k',run_id=101230,eval=True).results_dir)
report=root/'change/directional_center_joint_evaluation_20261001'/f'{NAME}_tnl2k_e0030_run101230'
assert not result.exists(),'Existing test results: '+str(result)
assert not report.exists(),'Existing evaluation: '+str(report)
for name in ('test_directional_center_ep30_tnl2k_FULL_run101230_20261001.log',
             'eval_directional_center_ep30_tnl2k_FULL_run101230_20261001.log'):
 assert not (root/'terminal_logs'/name).exists(),'Existing log '+name
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:args=(path/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 for arg in args[:4]:
  name=pathlib.Path(arg.decode(errors='ignore')).name
  assert not (name.startswith(('train_','test_directional_center_','test_center_joint_')) or name in ('test.py','train.py')),(path.name,name)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
print('E30_FULL_PREFLIGHT_PASS',json.dumps(dict(config=NAME,checkpoint=str(checkpoint),
 checkpoint_sha256=file_hash(checkpoint),dataset='tnl2k',mode='NLBBOX',sequences=700,
 frames=sum(len(seq.frames) for seq in dataset),epoch=30,runid=101230,workers=4,num_gpus=1,
 results=str(result),report=str(report))),flush=True)
PY
mkdir -p terminal_logs
(set -o noclobber; : > "$test_log"; : > "$eval_log")
trap 'status=$?; printf "%s PIPELINE_FAILED status=%s test_log=%s eval_log=%s\n" "$(date +%FT%T%z)" "$status" "$test_log" "$eval_log"; exit "$status"' ERR
printf '%s FULL_TEST_START runid=101230 workers=4 num_gpus=1\n' "$(date +%FT%T%z)"
"$python_bin" change/rqc_dev_v3/test_directional_center_joint_20261001.py \
 --dataset tnl2k --checkpoint "$checkpoint" --epoch 30 --runid "$runid" --threads 4 >> "$test_log" 2>&1
printf '%s FULL_TEST_FINISHED EVALUATION_START\n' "$(date +%FT%T%z)"
"$python_bin" change/rqc_dev_v3/evaluate_directional_center_joint_20261001.py \
 --dataset tnl2k --epoch 30 --runid "$runid" >> "$eval_log" 2>&1
printf '%s PIPELINE_COMPLETE test_log=%s eval_log=%s\n' "$(date +%FT%T%z)" "$test_log" "$eval_log"
