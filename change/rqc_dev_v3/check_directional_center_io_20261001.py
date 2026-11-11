"""Independent synthetic fixtures: native equivalence and output-only integration."""
import argparse,sys,os,json,copy,gc,subprocess
from pathlib import Path
import torch,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from directional_center_joint_contract_20261001 import NAME,file_hash
from directional_center_refiner_20261001 import DirectionalCenterRefiner
from directional_center_joint_tracker_20261001 import JointCenterEvaluationTracker
from lib.test.tracker.uvltrack import UVLTrack
from lib.test.evaluation import get_dataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(1);cv2.setNumThreads(1)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    report=json.loads((ROOT/'change/rqc_dev_v3/directional_center_real_check_verified_20261001.json').read_text())
    identity=report['identity'];source=Path(identity['source_checkpoint'])
    assert file_hash(source)==identity['source_checkpoint_sha256']
    original=torch.load(str(source),map_location='cpu')['net']
    head={k[len('box_head.'):]:v for k,v in original.items() if k.startswith('box_head.')}
    model=DirectionalCenterRefiner();fixture=dest/'DirectionalCenterJoint_ep0030.pth.tar'
    torch.save(dict(variant='directional_center_guard_joint',epoch=30,identity=identity,head=head,
        model=model.state_dict(),synthetic_check=True),fixture);del original,head
    wrapper=JointCenterEvaluationTracker('tnl2k',fixture,101130,epoch=30)
    read=lambda path:cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    zero_frames=nonzero_frames=changed=0
    for seq in get_dataset('tnl2k')[:2]:
        a=UVLTrack(copy.deepcopy(wrapper.params),'tnl2k');b=wrapper.tracker_class(wrapper.params,'tnl2k')
        try:
            for tracker in (a,b):
                tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
                tracker.initialize(read(seq.frames[0]),copy.deepcopy(dict(seq.init_info(),language=seq.language)))
            for t,path in enumerate(seq.frames[1:24],1):
                if t==12:
                    with torch.no_grad():b.refiner.net[-1].weight.normal_(std=.1)
                image=read(path);x=a.track(image);y=b.track(image)
                np.testing.assert_array_equal(a.state,b.state);assert torch.equal(a.prompt,b.prompt)
                if t<12:
                    np.testing.assert_array_equal(x['target_bbox'],y['target_bbox']);zero_frames+=1
                else:
                    nonzero_frames+=1;changed+=int(not np.array_equal(x['target_bbox'],y['target_bbox']))
                    assert np.isfinite(y['target_bbox']).all()
        finally:
            b.close();del a,b;gc.collect();torch.cuda.empty_cache()
    assert changed>0
    cli=[sys.executable,str(ROOT/'change/rqc_dev_v3/test_directional_center_joint_20261001.py'),
        '--dataset','tnl2k','--checkpoint',str(fixture),'--epoch','30','--runid','101130','--threads','4']
    rejected=subprocess.run(cli,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert rejected.returncode!=0 and 'synthetic check weights are forbidden' in rejected.stdout
    record=dict(zero_exact_frames=zero_frames,nonzero_same_state_prompt_frames=nonzero_frames,
        changed_output_frames=changed,synthetic_formal_test_rejected=True,optimizer_steps=0,
        formal_training_started=False,fixture=str(fixture))
    with (dest/'check_report.json').open('x') as f:json.dump(record,f,indent=2)
    print('DIRECTIONAL_CENTER_IO_CHECK_PASS',json.dumps(record),flush=True)
if __name__=='__main__':
    with torch.no_grad():main()
