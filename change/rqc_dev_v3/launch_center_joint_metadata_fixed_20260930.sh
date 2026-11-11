#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
python_bin=/root/miniconda3/envs/uvltrack/bin/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$python_bin" - <<'PY'
import hashlib,json,pathlib,shutil
root=pathlib.Path.cwd();base=root/'change/rqc_dev_v3'
real=json.loads((base/'center_joint_real_check_metadata_fixed_20260930.json').read_text())
data=json.loads((base/'center_joint_metadata_full_epoch_20260930.json').read_text())
assert real['identity']==data['identity']
assert real['optimizer_steps']==data['optimizer_steps']==0
assert real['frozen_backbone_and_bn_unchanged'] and real['head_weights_unchanged']
assert data['boundary']['original_images_labels_exact'] and data['boundary']['rng_exact']
assert data['boundary']['assertion_tolerance_unchanged']
assert len(data['loaders'])==6 and data['loaders'][0]['batches']==1875
io_lines=(base/'center_joint_io_metadata_fixed_20260930.log').read_text().splitlines()
io=json.loads([v for v in io_lines if v.startswith('JOINT_CENTER_IO_PASS ')][-1].split(' ',1)[1])
assert io['optimizer_steps']==0 and io['eight_worker_sequences']==2 and io['resume_roundtrip']
assert io['synthetic_formal_test_and_resume_rejected']
i=real['identity']
assert i['epochs']==30 and i['batch']==16 and i['samples_per_epoch']==30000
assert i['backbone_trainable']==0 and i['head_trainable']==15718505 and i['refiner_trainable']==19202
assert i['head_lr']==1e-6 and i['lr']==1e-4 and not i['ddp']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert sha(root/'experiments/uvltrack'/(i['config']+'.yaml'))==i['config_sha256']
for name,digest in i['source_files'].items():assert sha(base/name)==digest,name
assert sha(pathlib.Path(i['source_checkpoint']))==i['source_checkpoint_sha256']
output=root/'change/center_joint_training_exact_metadata_20260930'/i['config']
assert not output.exists(),'Refusing to overwrite existing training output'
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    for arg in args[:4]:
        name=pathlib.Path(arg.decode(errors='ignore')).name
        assert name not in ('train_center_joint_20260930.py','train_center_formal_20260930.py',
            'test_center_joint_20260930.py','check_center_joint_io_20260930.py',
            'check_center_joint_metadata_exact_20260930.py','train.py','test.py'),(p.name,name)
free=shutil.disk_usage(root).free/1024**3;assert free>=10
print('RESTART_PREFLIGHT_PASS',json.dumps(dict(config=i['config'],output=str(output),
    free_GiB=round(free,2),data_only_batches=sum(x['batches'] for x in data['loaders']))),flush=True)
PY
mkdir -p terminal_logs
log="terminal_logs/train_center_joint_metadata_fixed_30ep_$(date +%Y%m%d_%H%M%S).log"
(set -o noclobber; : > "$log")
nohup "$python_bin" change/rqc_dev_v3/train_center_joint_20260930.py --train >> "$log" 2>&1 < /dev/null &
training_pid=$!
printf 'TRAIN_PID=%s\nTRAIN_LOG=%s/%s\n' "$training_pid" "$PWD" "$log"
sleep 3
kill -0 "$training_pid"
tail -n 4 "$log"
