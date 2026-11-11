#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
python_bin=/root/miniconda3/envs/uvltrack/bin/python
config=baseline_base_head_directional_center_guard_bs16_30ep_20261001
checkpoint="change/directional_center_joint_training_20261001/${config}/DirectionalCenterJoint_ep0030.pth.tar"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
trap 'status=$?; printf "%s PIPELINE_FAILED status=%s dataset=%s test_log=%s eval_log=%s\n" "$(date +%FT%T%z)" "$status" "${dataset:-preflight}" "${test_log:-unset}" "${eval_log:-unset}"; exit "$status"' ERR
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
assert file_hash(checkpoint)=='1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd'
identity=saved['identity'];assert identity['config']==NAME
assert file_hash(root/'experiments/uvltrack'/(NAME+'.yaml'))==identity['config_sha256']
assert file_hash(identity['source_checkpoint'])==identity['source_checkpoint_sha256']
for name,digest in identity['source_files'].items():assert file_hash(base/name)==digest,name
assert json.loads((checkpoint.parent/'protocol.json').read_text())==identity
rows=[json.loads(s) for s in (checkpoint.parent/'stats.jsonl').read_text().splitlines() if s.strip()]
assert [r['epoch'] for r in rows]==list(range(1,31))
for dataset,runid,expected in [('lasot',101231,280),('otb99',101232,48)]:
 ds=get_dataset(dataset);assert len(ds)==expected
 for seq in ds:
  assert seq.language,seq.name
  assert pathlib.Path(seq.frames[0]).is_file() and pathlib.Path(seq.frames[-1]).is_file(),seq.name
 result=pathlib.Path(Tracker('uvltrack',NAME,dataset,run_id=runid,eval=True).results_dir)
 report=root/'change/directional_center_joint_evaluation_20261001'/f'{NAME}_{dataset}_e0030_run{runid}'
 assert not result.exists(),'Existing results: '+str(result)
 assert not report.exists(),'Existing report: '+str(report)
 for prefix in ['test','eval']:
  log=root/'terminal_logs'/f'{prefix}_directional_center_ep30_{dataset}_FULL_run{runid}_20261001.log'
  assert not log.exists(),'Existing log: '+str(log)
 print('E30_FULL_PREFLIGHT_PASS',json.dumps(dict(config=NAME,checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),dataset=dataset,mode='NLBBOX',sequences=len(ds),frames=sum(len(s.frames) for s in ds),epoch=30,runid=runid,workers=8,num_gpus=1,results=str(result),report=str(report))),flush=True)
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:args=(path/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 for arg in args[:4]:
  name=pathlib.Path(arg.decode(errors='ignore')).name
  assert not (name.startswith(('train_','test_directional_center_','test_center_joint_')) or name in ('test.py','train.py')),(path.name,name)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
PY
mkdir -p terminal_logs
for dataset in lasot otb99; do
 if [[ "$dataset" == lasot ]]; then runid=101231; else runid=101232; fi
 test_log="terminal_logs/test_directional_center_ep30_${dataset}_FULL_run${runid}_20261001.log"
 eval_log="terminal_logs/eval_directional_center_ep30_${dataset}_FULL_run${runid}_20261001.log"
 (set -o noclobber; : > "$test_log"; : > "$eval_log")
 printf '%s FULL_TEST_START dataset=%s epoch=30 runid=%s workers=8 num_gpus=1\n' "$(date +%FT%T%z)" "$dataset" "$runid"
 "$python_bin" change/rqc_dev_v3/test_directional_center_joint_20261001.py \
  --dataset "$dataset" --checkpoint "$checkpoint" --epoch 30 --runid "$runid" --threads 8 >> "$test_log" 2>&1
 printf '%s FULL_TEST_FINISHED EVALUATION_START dataset=%s runid=%s\n' "$(date +%FT%T%z)" "$dataset" "$runid"
 "$python_bin" change/rqc_dev_v3/evaluate_directional_center_joint_20261001.py \
  --dataset "$dataset" --epoch 30 --runid "$runid" >> "$eval_log" 2>&1
 printf '%s DATASET_COMPLETE dataset=%s runid=%s test_log=%s eval_log=%s\n' "$(date +%FT%T%z)" "$dataset" "$runid" "$test_log" "$eval_log"
done
printf '%s PIPELINE_COMPLETE datasets=lasot,otb99 epoch=30 runids=101231,101232\n' "$(date +%FT%T%z)"
