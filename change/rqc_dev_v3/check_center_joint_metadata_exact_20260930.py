"""Full-epoch data-only and boundary regression checks; never optimizer.step."""
import argparse,copy,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from train_center_joint_20260930 import prepare
from lib.utils import TensorDict
from lib.train.data.processing import TrackProcessing
from center_joint_metadata_exact_20260930 import ExactMetadataTrackProcessing

def rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state())
def set_rng(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
def assert_rng(a,b):
    assert a[0]==b[0] and a[1][0]==b[1][0]
    assert np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:]
    assert torch.equal(a[2],b[2])

def regression(processor):
    native=copy.deepcopy(processor);native.__class__=TrackProcessing
    native.transform=dict(native.transform)
    native.transform['search']=native.transform['search'].original
    fixed=copy.deepcopy(processor)
    boundary=torch.tensor([.756011962890625,455.1298522949219,125.260498046875,679.5283813476562])
    x,y,w,h=boundary.tolist()
    old_side=math.ceil(math.sqrt(float(boundary[2]*boundary[3]))*4)
    native_side=math.ceil(math.sqrt(w*h)*4)
    assert (old_side,native_side)==(1167,1168)
    def synthetic():
        image=np.arange(800*1000*3,dtype=np.uint8).reshape(800,1000,3)
        box=torch.tensor([245.,213.,242.,394.])
        return TensorDict(dict(template_images=[image.copy()],template_anno=[box.clone()],
            search_images=[image.copy(),image.copy()],search_anno=[box.clone(),box.clone()]))
    before=rng();original_jitter=TrackProcessing._get_jittered_box
    def forced(self,box,mode):
        return boundary.clone() if mode=='search' else original_jitter(self,box,mode)
    results=[]
    try:
        TrackProcessing._get_jittered_box=forced
        # Both flip branches and several augmentation rolls must remain exact.
        for seed in range(8):
            random.seed(9000+seed);np.random.seed(9000+seed);torch.manual_seed(9000+seed)
            state=rng();a=native.track_process(synthetic());after_native=rng()
            set_rng(state);b=fixed.track_process(synthetic());after_fixed=rng()
            assert a['valid'] and b['valid'];assert_rng(after_native,after_fixed)
            for key in a:
                if torch.is_tensor(a[key]):assert torch.equal(a[key],b[key]),key
                else:assert a[key]==b[key],key
            assert b['center_side'].eq(1168).all()
            assert isinstance(fixed,ExactMetadataTrackProcessing)
            results.extend(b['center_flip'].tolist())
        assert any(results) and not all(results),'Both flip branches required'
    finally:
        TrackProcessing._get_jittered_box=original_jitter;set_rng(before)
    return dict(old_side=old_side,native_side=native_side,fixed_side=1168,
        original_images_labels_exact=True,rng_exact=True,flip_and_no_flip=True,
        assertion_tolerance_unchanged=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);args=p.parse_args()
    report=Path(args.report)
    if report.exists():raise FileExistsError(report)
    start=time.time();net,actor,loaders,optimizer,scheduler,identity=prepare()
    boundary=regression(loaders[0].dataset.processing)
    print('BOUNDARY_REGRESSION_PASS',json.dumps(boundary),flush=True)
    checked=[]
    for loader in loaders:
        count=0;frames=0
        for index,data in enumerate(loader):
            count=index+1;frames+=data['center_side'].numel()
            assert torch.isfinite(data['center_gt']).all()
            assert isinstance(loader.dataset.processing,ExactMetadataTrackProcessing)
            if count%250==0:print('DATA_CHECK',loader.name,count,len(loader),flush=True)
        assert count==len(loader)
        checked.append(dict(loader=loader.name,batches=count,search_frames=frames))
        print('LOADER_COMPLETE',json.dumps(checked[-1]),flush=True)
    assert not optimizer.state
    result=dict(identity=identity,boundary=boundary,loaders=checked,
        seconds=time.time()-start,optimizer_steps=0,formal_training_started=False)
    with report.open('x') as out:json.dump(result,out,indent=2)
    print('EXACT_METADATA_FULL_EPOCH_PASS',json.dumps(result),flush=True)

if __name__=='__main__':main()
