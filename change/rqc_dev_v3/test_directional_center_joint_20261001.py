"""Use official datasets, runner and result writer with isolated center tracker."""
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from directional_center_joint_tracker_20261001 import JointCenterEvaluationTracker
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset

def verify_outputs(dataset,tracker,epoch,require_all):
    import numpy as np
    count=0
    for seq in dataset:
        path=Path(tracker.results_dir)/(seq.dataset+'_NLBBOX_'+f'{epoch:03d}')/(seq.name+'.txt')
        if not path.is_file():
            if require_all:raise RuntimeError('Missing sequence output: '+str(path))
            continue
        values=np.loadtxt(str(path)).reshape(-1,4)
        if values.shape!=(len(seq.frames),4) or not np.isfinite(values).all() or (values[:,2:]<=0).any():
            raise RuntimeError('Incomplete/invalid output; refusing to silently skip: '+str(path))
        count+=1
    return count

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['tnl2k','lasot','otb99'],required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--runid',type=int,required=True)
    p.add_argument('--epoch',type=int,default=30)
    p.add_argument('--resume-results',action='store_true',help='Resume only an identical manifest/checkpoint run')
    p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    os.chdir(str(ROOT));checkpoint=Path(args.checkpoint).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(str(checkpoint))
    if not 1<=args.threads<=8:raise ValueError('Single GPU uses 1..8 workers')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from directional_center_joint_contract_20261001 import NAME,file_hash
    saved=torch.load(str(checkpoint),map_location='cpu')
    if saved.get('synthetic_check') or saved.get('variant')!='directional_center_guard_joint' or 'optimizer' not in saved or 'head' not in saved:
        raise ValueError('Not a completed formal-training checkpoint (synthetic check weights are forbidden)')
    if int(saved['epoch'])!=args.epoch or checkpoint.name!=f'DirectionalCenterJoint_ep{args.epoch:04d}.pth.tar':
        raise ValueError('Requested epoch/checkpoint filename/metadata mismatch')
    if saved['identity']['config']!=NAME:raise ValueError('Wrong experiment checkpoint')
    for name,digest in saved['identity']['source_files'].items():
        if file_hash(Path(__file__).parent/name)!=digest:raise ValueError('Code changed since training: '+name)
    tracker=JointCenterEvaluationTracker(args.dataset,checkpoint,args.runid,epoch=args.epoch)
    dest=Path(tracker.results_dir)
    manifest=dict(config=NAME,dataset=args.dataset,epoch=args.epoch,runid=args.runid,
        checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),identity=saved['identity'])
    if args.resume_results:
        if json.loads((dest/'center_joint_run_manifest.json').read_text())!=manifest:
            raise ValueError('Existing run identity mismatch; refusing to resume')
    else:
        dest.mkdir(parents=True,exist_ok=False)
        (dest/'center_joint_run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('CHECKPOINT',checkpoint,'RESULTS',tracker.results_dir,flush=True)
    dataset=get_dataset(args.dataset)
    verify_outputs(dataset,tracker,args.epoch,require_all=False)
    run_dataset(dataset,[tracker],debug=0,threads=args.threads,num_gpus=1)
    count=verify_outputs(dataset,tracker,args.epoch,require_all=True)
    print('TEST_COMPLETE',json.dumps(dict(sequences=count,expected=len(dataset),results=str(dest))),flush=True)

if __name__=='__main__':main()
