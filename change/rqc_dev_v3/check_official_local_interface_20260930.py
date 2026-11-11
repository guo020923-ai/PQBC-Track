"""Official E300 interface smoke: unchanged selection/state, no optimization."""
import os,sys,json,math
from pathlib import Path
import numpy as np
import torch,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prepare_locked_refiner_protocol_20260930 import load_role
from head_quality_features_20260930 import HeadQualityCapture
from local_box_refiner_20260930 import LocalBoxRefiner,decode_image_boxes
from lib.test.parameter.uvltrack import parameters
from lib.test.tracker.uvltrack import UVLTrack

def main():
    os.chdir(str(ROOT));torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(42)
    params=parameters('baseline_base',epoch=300);params.debug=0
    assert params.cfg.TEST.MODE=='NLBBOX'
    assert not params.cfg.MODEL.HEAD.ERC.ENABLED and not params.cfg.MODEL.RQC.ENABLED
    assert Path(params.checkpoint).name=='UVLTrack_ep0300.pth.tar'
    tracker=UVLTrack(params,'tnl2k')
    state=torch.load(params.checkpoint,map_location='cpu')['net']
    tracker.network.load_state_dict(state,strict=True);del state
    for p in tracker.network.parameters():p.requires_grad_(False)
    tracker.network.eval();refiner=LocalBoxRefiner().cuda().eval()
    manifest=load_role(ROOT/'change/rqc_dev_v3/locked_refiner_protocol_20260930','dev')
    selected=[next(m for m in manifest if m['domain']==d) for d in ('lasot_train','tnl2k_train')]
    capture={};original=tracker.network.forward_test
    with HeadQualityCapture(tracker.network.box_head) as cap:
        def observed(*args,**kwargs):
            out=original(*args,**kwargs)
            assert int(tracker.flag.item())==2
            cls=out['cls_score_test'].view(-1).detach().cpu()
            cont=out['cont_score'].softmax(-1)[:,:,0].view(-1).detach().cpu()
            idx=torch.argmax(cls*tracker.window*cont).reshape(1,1)
            box=out['bbox_map'].reshape(-1,4)[idx.item()].detach().reshape(1,4)
            feature=cap.gather(idx.to(box.device))[:,0]
            capture.update(box=box,refined=refiner(feature,box)['boxes'])
            assert torch.equal(capture['refined'],box)
            return out
        tracker.network.forward_test=observed
        try:
            read=lambda f:cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB)
            checked=0;max_error=0.
            for m in selected:
                path=Path(m['path']);tnl=m['domain']=='tnl2k_train'
                frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
                gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2)
                language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
                tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
                tracker.initialize(read(frames[0]),dict(init_bbox=gt[0].tolist(),language=language))
                for t in range(1,min(25,len(frames))):
                    image=read(frames[t]);previous=np.asarray(tracker.state)
                    side=math.ceil(math.sqrt(previous[2]*previous[3])*params.search_factor)
                    origin=previous[:2]+previous[2:]/2-side/2
                    # Observer returns the native forward outputs unchanged. This checks
                    # coordinate equivalence, not an independently executed second trajectory.
                    prediction=tracker.track(image)['target_bbox'];internal=list(tracker.state)
                    corrected=decode_image_boxes(capture['refined'].cpu(),torch.tensor(origin,dtype=torch.float32)[None],
                        torch.tensor([side]),torch.tensor([image.shape[:2]]))[0].numpy()
                    np.testing.assert_allclose(corrected,prediction,atol=.002,rtol=0)
                    assert tracker.state==internal
                    max_error=max(max_error,float(np.abs(corrected-np.asarray(prediction)).max()));checked+=1
            print('OFFICIAL_INTERFACE_PASS',json.dumps(dict(frames=checked,max_coordinate_error=max_error,
                checkpoint=params.checkpoint,strict_load=True,optimizer_steps=0,weights_saved=False,
                backbone_head_frozen=True,erc_rqc_disabled=True)))
        finally:tracker.network.forward_test=original

if __name__=='__main__':
    with torch.no_grad():main()
