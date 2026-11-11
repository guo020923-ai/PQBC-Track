"""Two truncated TRAIN-source sequences, eight native runner workers; no fit."""
import os,sys,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from center_formal_contract_20260930 import NAME,register_config,file_hash
from center_safe_refiner_20260930 import CenterSafeRefiner
from center_output_tracker_20260930 import CenterEvaluationTracker
from prepare_locked_refiner_protocol_20260930 import load_role
from test_center_formal_20260930 import verify_outputs
from lib.test.evaluation import Sequence
from lib.test.evaluation.running import run_dataset

def main():
    os.chdir(str(ROOT));register_config();torch.set_num_threads(1)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    dest=ROOT/'change/rqc_dev_v3'/('center_runner_check_'+time.strftime('%Y%m%d_%H%M%S'))
    dest.mkdir(exist_ok=False)
    source=ROOT/'checkpoints/train/uvltrack/baseline_base/UVLTrack_ep0300.pth.tar'
    identity=dict(config=NAME,config_sha256=file_hash(ROOT/'experiments/uvltrack'/(NAME+'.yaml')),
        source_checkpoint_sha256=file_hash(source),output_only=True,head_trainable=0)
    checkpoint=dest/'synthetic_zero_not_trained.pt'
    torch.save(dict(variant='center_safe',epoch=0,identity=identity,model=CenterSafeRefiner().state_dict(),synthetic_check=True),checkpoint)
    # Formal test command must reject these engineering-only weights BEFORE creating results.
    result=subprocess.run([sys.executable,str(Path(__file__).parent/'test_center_formal_20260930.py'),
        '--dataset','tnl2k','--checkpoint',str(checkpoint),'--runid','93098'],capture_output=True,text=True)
    assert result.returncode!=0 and 'synthetic check weights are forbidden' in result.stderr
    manifest=load_role(ROOT/'change/rqc_dev_v3/locked_refiner_protocol_20260930','dev');sequences=[]
    for domain in ('lasot_train','tnl2k_train'):
        m=next(m for m in manifest if m['domain']==domain);path=Path(m['path']);tnl=domain=='tnl2k_train'
        frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))[:24]
        gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2)[:len(frames)]
        language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
        visible=None
        if not tnl:
            occlusion=np.loadtxt(path/'full_occlusion.txt',delimiter=',').reshape(-1)[:len(frames)]
            out_of_view=np.loadtxt(path/'out_of_view.txt',delimiter=',').reshape(-1)[:len(frames)]
            visible=(occlusion==0)&(out_of_view==0)
        sequences.append(Sequence(m['name'],[str(f) for f in frames],
            'tnl2k' if tnl else 'lasot',gt,language=language,target_visible=visible))
    tracker=CenterEvaluationTracker('tnl2k',checkpoint,93098)
    tracker.results_dir=str(dest/'runner_results')
    run_dataset(sequences,[tracker],debug=0,threads=8,num_gpus=1)
    count=verify_outputs(sequences,tracker,30,require_all=True)
    from evaluate_center_formal_20260930 import run_official_metrics
    for seq in sequences:
        report=dest/(seq.dataset+'_synthetic_metric_check')
        run_official_metrics([tracker],[seq],seq.dataset+'_NLBBOX_030',report)
        text=(report/'metrics.txt').read_text()
        assert 'AUC' in text and 'OP50' in text and 'Norm Precision' in text
    # Incomplete rows are caught rather than counted as a successful sequence.
    fake=Sequence('incomplete_smoke',['a','b'],'tnl2k',np.ones((2,4)),language='synthetic')
    bad=Path(tracker.results_dir)/'tnl2k_NLBBOX_030'/'incomplete_smoke.txt'
    np.savetxt(str(bad),np.ones((1,4)))
    try:verify_outputs([fake],tracker,30,require_all=True)
    except RuntimeError:pass
    else:raise AssertionError('Incomplete output accepted')
    print('CENTER_RUNNER_CHECK_PASS',json.dumps(dict(sequences=count,frames=sum(len(s.frames) for s in sequences),
        processes=8,num_gpus=1,optimizer_steps=0,formal_synthetic_weights_rejected=True,
        incomplete_outputs_rejected=True,directory=str(dest))),flush=True)

if __name__=='__main__':main()
