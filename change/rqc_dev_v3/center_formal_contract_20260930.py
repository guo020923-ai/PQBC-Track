"""Isolated center-refinement contract; no modification of shared UVLTrack files."""
import math,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from easydict import EasyDict as edict
from lib.config.uvltrack.config import cfg
from lib.train.data.processing import TrackProcessing
from lib.train.data import transforms as tfm
from lib.train.actors.uvltrack import UVLTrackActor
from head_quality_features_20260930 import HeadQualityCapture
from center_safe_refiner_20260930 import CenterSafeRefiner,center_safe_loss
from local_box_refiner_20260930 import decode_image_boxes

NAME='baseline_base_frozen_head_output_center_safe_bs16_30ep_20260930'

def register_config():
    if not hasattr(cfg.MODEL,'CENTER_OUTPUT'):
        cfg.MODEL.CENTER_OUTPUT=edict(ENABLED=False,CENTER_LIMIT=.1,OUTPUT_ONLY=True)
        cfg.TRAIN.CENTER_OUTPUT_LR=1e-4

def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

class RecordingSearchTransform:
    def __init__(self,original,owner):self.original=original;self.owner=owner
    def __call__(self,**inputs):
        if not self.owner._center_recording:return self.original(**inputs)
        assert inputs.get('joint',True) is False
        names=[k for k in inputs if k in self.original._valid_inputs]
        outputs=[]
        # The native Transform uses exactly this frame-major splitting/order.
        for item in self.original._split_inputs(inputs):
            result=self.original(**item)
            flips=[t for t in self.original.transforms if isinstance(t,tfm.RandomHorizontalFlip_Norm)]
            assert len(flips)<=1
            flip=bool(flips[0]._rand_params[0]) if flips else False
            self.owner._center_flips.append(flip)
            outputs.append(result)
        return tuple([r[i] for r in outputs] for i in range(len(names)))

class MetadataTrackProcessing(TrackProcessing):
    def _get_jittered_box(self,box,mode):
        result=super()._get_jittered_box(box,mode)
        if self._center_recording and mode=='search':
            index=len(self._center_meta);image=self._center_data['search_images'][index]
            side=math.ceil(math.sqrt(float(result[2]*result[3]))*self.search_area_factor['search'])
            center=result[:2]+result[2:]/2
            size=self.output_sz['search']
            # Separate label inverse (half-pixel convention) from TEST decoder.
            # Supervision of predicted boxes must match native map_box_back.
            annotation_origin=center-side*(size-1)/(2*size)
            origin=center-side/2
            self._center_meta.append(dict(origin=origin.clone(),annotation_origin=annotation_origin.clone(),
                extract_box=result.clone(),side=float(side),
                hw=torch.tensor(image.shape[:2],dtype=torch.float32),gt=box.clone()))
        return result
    def track_process(self,data):
        self._center_recording=True;self._center_data=data;self._center_meta=[];self._center_flips=[]
        try:out=super().track_process(data)
        finally:self._center_recording=False;self._center_data=None
        if not out['valid']:return out
        assert len(self._center_meta)==len(self._center_flips)==out['search_images'].shape[0]
        out['center_origin']=torch.stack([r['origin'] for r in self._center_meta])
        out['center_side']=torch.tensor([r['side'] for r in self._center_meta])
        out['center_hw']=torch.stack([r['hw'] for r in self._center_meta])
        out['center_gt']=torch.stack([r['gt'] for r in self._center_meta])
        out['center_flip']=torch.tensor(self._center_flips,dtype=torch.bool)
        # Recover the GT from augmented crop coordinates; no extra random rolls.
        recovered=unflip_boxes(torch.cat((out['search_anno'][:,:2]+out['search_anno'][:,2:]/2,
            out['search_anno'][:,2:]),-1),out['center_flip'])
        annotation_origin=torch.stack([r['annotation_origin'] for r in self._center_meta])
        gt_image=torch.cat((annotation_origin+(recovered[:,:2]-recovered[:,2:]/2)*out['center_side'][:,None],
            recovered[:,2:]*out['center_side'][:,None]),-1)
        torch.testing.assert_close(gt_image,out['center_gt'],rtol=2e-4,atol=.004)
        return out
    def grounding_process(self,data):
        self._center_recording=False
        out=super().grounding_process(data)
        if out['valid']:
            n=out['search_images'].shape[0]
            out['center_origin']=torch.zeros(n,2);out['center_side']=torch.ones(n)
            out['center_hw']=torch.full((n,2),256.);out['center_gt']=torch.zeros(n,4)
            out['center_flip']=torch.zeros(n,dtype=torch.bool)
        return out

def install_metadata(processing):
    if isinstance(processing,MetadataTrackProcessing):return processing
    obj=copy.copy(processing);obj.__class__=MetadataTrackProcessing
    obj.transform=dict(processing.transform);obj._center_recording=False
    obj.transform['search']=RecordingSearchTransform(processing.transform['search'],obj)
    return obj

def unflip_boxes(boxes,flip):
    center_x=torch.where(flip.detach().bool(),1-boxes[:,0],boxes[:,0])
    return torch.cat((center_x[:,None],boxes[:,1:]),-1)

def decode_training_boxes(boxes,data,active):
    flip=data['center_flip'].reshape(-1)[active]
    return decode_image_boxes(unflip_boxes(boxes,flip),
        data['center_origin'].reshape(-1,2)[active],data['center_side'].reshape(-1)[active],
        data['center_hw'].reshape(-1,2)[active])

def select_native(out):
    cls=out['cls_score_test'].flatten(1).detach().cpu()
    cont=out['cont_score'].softmax(-1)[...,0].detach().cpu()
    side=int(math.sqrt(cls.shape[1]));assert side*side==cls.shape[1]
    window=np.outer(np.hanning(side),np.hanning(side)).reshape(-1)
    # Native tracker multiplies float32 responses by a numpy float64 Hann on CPU.
    score=cls*torch.from_numpy(window)*cont
    return score.argmax(-1,keepdim=True)

class CenterOutputActor(UVLTrackActor):
    def train(self,mode=True):
        self.net.eval()  # Freeze both parameters AND BN/dropout behavior.
        self.net.output_refiner.train(mode)
    def __call__(self,data):
        with HeadQualityCapture(self.net.box_head) as capture:
            with torch.no_grad():out=self.forward_pass(data)
            active=out['flag'].reshape(-1).eq(2)
            if not active.any():
                zero=self.net.output_refiner.net[-1].weight.sum()*0
                return zero,{'Loss/total':0.,'Center/active_samples':0.,'Center/iou_gain':0.}
            indices=select_native(out).to(out['bbox_map'].device)
            boxes=out['bbox_map'].detach().gather(1,indices[...,None].expand(-1,-1,4))[:,0]
            features=capture.gather(indices)[:,0]
            prediction=self.net.output_refiner(features[active],boxes[active])
            assert torch.equal(prediction['boxes'][:,2:],boxes[active,2:])
        base=decode_training_boxes(boxes[active],data,active)
        refined=decode_training_boxes(prediction['boxes'],data,active)
        truth=data['center_gt'].reshape(-1,4)[active]
        hw=data['center_hw'].reshape(-1,2)[active]
        valid=torch.isfinite(truth).all(-1)&(truth[:,2:]>0).all(-1)
        loss,stats=center_safe_loss(refined,base,truth,hw,valid)
        return loss,dict({'Loss/total':float(loss.detach()),'Center/active_samples':float(active.sum()),
            'Center/delta_abs':float(prediction['delta'].detach().abs().mean())},
            **{'Center/'+k:float(v) for k,v in stats.items()})

def frozen_buffer_snapshot(net):
    return {k:v.detach().cpu().clone() for k,v in net.named_buffers() if not k.startswith('output_refiner.')}

def verify_frozen(net,before=None):
    assert all(not p.requires_grad and p.grad is None for n,p in net.named_parameters() if not n.startswith('output_refiner.'))
    assert not net.backbone.training and not net.box_head.training
    if before is not None:
        for k,v in net.named_buffers():
            if k in before:assert torch.equal(v.detach().cpu(),before[k]),'Frozen buffer changed: '+k
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==19202
