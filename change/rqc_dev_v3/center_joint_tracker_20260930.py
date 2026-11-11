"""Native UVLTrack internal trajectory + independent center-corrected output."""
import sys,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from center_joint_contract_20260930 import NAME,register_config,select_native,file_hash
from center_safe_refiner_20260930 import CenterSafeRefiner
from head_quality_features_20260930 import HeadQualityCapture
from local_box_refiner_20260930 import decode_image_boxes
from lib.test.tracker.uvltrack import UVLTrack
from lib.test.evaluation.tracker import Tracker
from lib.test.parameter.uvltrack import parameters

def source_signature(path):
    path=Path(path).resolve();stat=path.stat()
    return (str(path),stat.st_size,stat.st_mtime_ns)

class JointCenterTracker(UVLTrack):
    def __init__(self,params,dataset_name):
        assert source_signature(params.checkpoint)==params.center_source_signature,'Official source file changed'
        super().__init__(params,dataset_name)
        assert self.cfg.TEST.MODE=='NLBBOX' and not self.erc_enabled
        # The native constructor has already loaded the immutable source. Exact
        # key/shape matching gives strict-loading compatibility without a second
        # large torch.load for every sequence. Schema/hash verified once in parent.
        schema={k:tuple(v.shape) for k,v in self.network.state_dict().items()}
        assert schema==params.center_source_schema,'Official checkpoint/model key or shape mismatch'
        self.refiner=CenterSafeRefiner(self.cfg.MODEL.CENTER_OUTPUT.CENTER_LIMIT).cuda().eval()
        saved=torch.load(str(params.center_checkpoint),map_location='cpu')
        assert saved['variant']=='center_safe_joint' and saved['identity']['config']==NAME
        assert saved['identity']['config_sha256']==file_hash(ROOT/'experiments/uvltrack'/(NAME+'.yaml'))
        assert saved['identity']['source_checkpoint_sha256']==params.center_source_sha256
        assert saved['identity']['output_only'] and saved['identity']['head_trainable']>0
        if 'head' not in saved:raise ValueError('Joint checkpoint must include the trained Head')
        self.network.box_head.load_state_dict(saved['head'],strict=True)
        self.refiner.load_state_dict(saved['model'],strict=True)
        for p in self.network.parameters():p.requires_grad_(False)
        for p in self.refiner.parameters():p.requires_grad_(False)
        self._capture=HeadQualityCapture(self.network.box_head);self._capture.__enter__()
        self._original_forward=self.network.forward_test;self._center_prediction=None
        def observed(*args,**kwargs):
            out=self._original_forward(*args,**kwargs)
            assert int(self.flag.item())==2
            indices=select_native(out).to(out['bbox_map'].device)
            boxes=out['bbox_map'].detach().gather(1,indices[...,None].expand(-1,-1,4))[:,0]
            features=self._capture.gather(indices)[:,0]
            self._center_prediction=self.refiner(features,boxes)
            assert torch.equal(self._center_prediction['boxes'][:,2:],boxes[:,2:])
            return out
        self.network.forward_test=observed
    @torch.no_grad()
    def track(self,image,info=None):
        previous=np.array(self.state,dtype=np.float64)
        side=math.ceil(math.sqrt(previous[2]*previous[3])*self.params.search_factor)
        origin=previous[:2]+previous[2:]/2-side/2
        out=super().track(image,info);native=list(self.state)
        assert self._center_prediction is not None
        self.last_native_box=native
        if torch.count_nonzero(self._center_prediction['delta'])==0:
            # Strict bitwise native equivalence at zero, no alternate rounding path.
            return out
        corrected=decode_image_boxes(self._center_prediction['boxes'].cpu(),
            torch.tensor(origin,dtype=torch.float32)[None],torch.tensor([side]),
            torch.tensor([image.shape[:2]]))[0].tolist()
        assert self.state==native
        result=dict(out);result['target_bbox']=corrected;return result
    def close(self):
        self.network.forward_test=self._original_forward
        self._capture.__exit__()

class JointCenterEvaluationTracker(Tracker):
    def __init__(self,dataset_name,checkpoint,run_id,epoch=30):
        self.center_checkpoint=Path(checkpoint)
        self.center_epoch=int(epoch)
        super().__init__('uvltrack',NAME,dataset_name,run_id=run_id,epoch=self.center_epoch)
        self.tracker_class=JointCenterTracker
    def get_parameters(self,epoch=None):
        register_config();p=parameters(NAME,epoch=self.center_epoch)
        # The immutable baseline stays E300; the independent small checkpoint is separate.
        p.checkpoint=p.cfg.TRAIN.INIT_CHECKPOINT;p.center_checkpoint=self.center_checkpoint
        p.center_source_signature=source_signature(p.checkpoint)
        p.center_source_sha256=file_hash(p.checkpoint)
        source=torch.load(p.checkpoint,map_location='cpu')['net']
        p.center_source_schema={k:tuple(v.shape) for k,v in source.items()}
        del source
        p.debug=0;return p
    def create_tracker(self,params):
        self._live_tracker=super().create_tracker(params)
        return self._live_tracker
    def run_sequence(self,seq,debug=None):
        self._live_tracker=None
        try:return super().run_sequence(seq,debug)
        finally:
            if self._live_tracker is not None:self._live_tracker.close()
            self._live_tracker=None
