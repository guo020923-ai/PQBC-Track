"""Formal E30: reproduce reference retention and verify prediction equivalence."""
import sys,json,gc,weakref,copy,os,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np,cv2
from center_joint_tracker_20260930 import JointCenterEvaluationTracker
from center_joint_lifecycle_20261001 import MemorySafeCenterEvaluationTracker
from center_joint_contract_20260930 import NAME
from lib.test.evaluation import get_dataset

def clean():
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);args=p.parse_args()
    report=Path(args.report)
    if report.exists():raise FileExistsError(report)
    os.chdir(ROOT);torch.set_num_threads(1);cv2.setNumThreads(1)
    checkpoint=ROOT/'change/center_joint_training_exact_metadata_20260930'/NAME/'CenterJoint_ep0030.pth.tar'
    old=JointCenterEvaluationTracker('tnl2k',checkpoint,100130,epoch=30)
    safe=MemorySafeCenterEvaluationTracker('tnl2k',checkpoint,100130,epoch=30)
    retention={}
    clean()
    # Keep GC disabled only during the diagnostic to reveal reference cycles.
    # It is restored immediately after each case, not used in formal testing.
    for label,wrapper in [('old',old),('safe',safe)]:
        clean();before=torch.cuda.memory_allocated();refs=[];growth=[]
        gc.disable()
        try:
            for repeat in range(3):
                candidate=wrapper.tracker_class(wrapper.params,'tnl2k')
                # torch.load can re-enable Python GC internally. Disable it
                # again *after* construction to isolate the close() reference
                # lifetime, rather than assuming its initial mode survived I/O.
                gc.disable()
                refs.append(weakref.ref(candidate.network))
                candidate.close();del candidate;torch.cuda.synchronize()
                growth.append(torch.cuda.memory_allocated()-before)
            # Measure before re-enabling GC: an automatic collection on the
            # following Python allocation would otherwise hide the cycle.
            retained=sum(ref() is not None for ref in refs)
            retention[label]=dict(retained_networks=retained,
                cuda_growth_mib=[round(v/2**20,2) for v in growth])
        finally:gc.enable()
        clean();assert all(ref() is None for ref in refs)
    print('LIFECYCLE_REFERENCE_CHECK',json.dumps(retention),flush=True)
    assert retention['old']['retained_networks']>0,'Expected bound-method cycle not reproduced'
    assert retention['safe']['retained_networks']==0
    assert max(retention['safe']['cuda_growth_mib'])<1
    seqs=get_dataset('tnl2k');checked=0
    for seq in seqs[:2]:
        a=old.tracker_class(old.params,'tnl2k');b=safe.tracker_class(safe.params,'tnl2k')
        # Official Tracker._track_sequence injects language before initialize.
        # Direct equivalence probes must pass the same initialization contract.
        info=dict(seq.init_info(),language=seq.language)
        read=lambda path:cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        try:
            for tracker in (a,b):
                tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
                tracker.initialize(read(seq.frames[0]),copy.deepcopy(info))
            for path in seq.frames[1:24]:
                image=read(path);x=a.track(image);y=b.track(image)
                np.testing.assert_array_equal(x['target_bbox'],y['target_bbox'])
                np.testing.assert_array_equal(a.state,b.state)
                assert torch.equal(a.prompt,b.prompt)
                checked+=1
        finally:
            a.close();b.close();del a,b;clean()
    # Repeated real-frame safe sequences exercise template/prompt tensor release.
    release=[]
    for repeat in range(5):
        seq=seqs[repeat%2];candidate=safe.tracker_class(safe.params,'tnl2k')
        ref=weakref.ref(candidate.network)
        candidate.max_score=0;candidate.out_dict=None;candidate.pred_box_net=None
        candidate.initialize(read(seq.frames[0]),copy.deepcopy(dict(seq.init_info(),language=seq.language)))
        for path in seq.frames[1:4]:candidate.track(read(path))
        candidate.close();del candidate;clean()
        assert ref() is None
        release.append(round(torch.cuda.memory_allocated()/2**20,2))
    assert max(release)-min(release)<1,'CUDA allocations accumulate after real sequences'
    result=dict(prediction_frames=checked,boxes_state_prompt_exact=True,retention=retention,
        repeated_sequence_allocated_mib=release,checkpoint=str(checkpoint),optimizer_steps=0,
        model_prediction_changed=False)
    with report.open('x') as out:json.dump(result,out,indent=2)
    print('LIFECYCLE_CHECK_PASS',json.dumps(result),flush=True)

if __name__=='__main__':main()
