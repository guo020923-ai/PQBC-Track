"""Independent native trajectory equivalence with zero and synthetic residuals."""
import os,sys,json,time,copy
from pathlib import Path
import numpy as np
import torch,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from center_formal_contract_20260930 import NAME,register_config,file_hash
from center_safe_refiner_20260930 import CenterSafeRefiner
from center_output_tracker_20260930 import CenterSafeTracker,CenterEvaluationTracker
from prepare_locked_refiner_protocol_20260930 import load_role
from lib.test.tracker.uvltrack import UVLTrack
from lib.test.parameter.uvltrack import parameters

def main():
    os.chdir(str(ROOT));torch.set_num_threads(1);cv2.setNumThreads(1);register_config()
    dest=ROOT/'change/rqc_dev_v3'/('center_tracker_check_'+time.strftime('%Y%m%d_%H%M%S'))
    dest.mkdir(exist_ok=False)
    source=ROOT/'checkpoints/train/uvltrack/baseline_base/UVLTrack_ep0300.pth.tar'
    identity=dict(config=NAME,config_sha256=file_hash(ROOT/'experiments/uvltrack'/(NAME+'.yaml')),
        source_checkpoint_sha256=file_hash(source),output_only=True,head_trainable=0)
    model=CenterSafeRefiner()
    manifest=load_role(ROOT/'change/rqc_dev_v3/locked_refiner_protocol_20260930','dev')
    chosen=[next(m for m in manifest if m['domain']==d) for d in ('lasot_train','tnl2k_train')]
    counts={};read=lambda f:cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB)
    for variant in ('zero','synthetic_nonzero'):
        if variant=='synthetic_nonzero':
            with torch.no_grad():model.net[-1].bias.fill_(.05)
        file=dest/(variant+'.pt')
        torch.save(dict(variant='center_safe',epoch=0,identity=identity,model=model.state_dict(),synthetic_check=True),file)
        native_params=parameters('baseline_base',epoch=300);native_params.debug=0
        native=UVLTrack(native_params,'tnl2k')
        wrapper=CenterEvaluationTracker('tnl2k',file,run_id=93000)
        candidate=wrapper.create_tracker(wrapper.params);frames_checked=0;changes=0
        try:
            for m in chosen:
                path=Path(m['path']);tnl=m['domain']=='tnl2k_train'
                frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
                gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2)
                language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
                info=dict(init_bbox=gt[0].tolist(),language=language)
                for tracker in (native,candidate):
                    tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
                    tracker.initialize(read(frames[0]),copy.deepcopy(info))
                for t in range(1,min(25,len(frames))):
                    image=read(frames[t]);a=native.track(image)['target_bbox'];b=candidate.track(image)['target_bbox']
                    np.testing.assert_array_equal(a,candidate.state)
                    np.testing.assert_array_equal(a,candidate.last_native_box)
                    assert torch.equal(native.prompt,candidate.prompt)
                    if variant=='zero':np.testing.assert_array_equal(a,b)
                    else:changes+=int(not np.array_equal(a,b))
                    assert np.isfinite(b).all() and min(b[2:])>0
                    frames_checked+=1
            if variant!='zero':assert changes>0
            counts[variant]=dict(frames=frames_checked,output_changes=changes,native_state_exact=True,prompt_exact=True)
        finally:candidate.close();del native,candidate;torch.cuda.empty_cache()
    print('CENTER_TRACKER_PASS',json.dumps(dict(checks=counts,optimizer_steps=0,
        test_weights='synthetic initialization only, not trained',directory=str(dest))),flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
