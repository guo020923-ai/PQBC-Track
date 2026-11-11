"""Restricted simultaneous adaptation: original Head loss + detached refiner loss."""
import torch
from lib.train.actors.uvltrack import UVLTrackActor
from center_formal_contract_20260930 import (
    register_config as register_base_config,file_hash,select_native,decode_training_boxes)
from center_joint_metadata_exact_20260930 import install_metadata
from center_safe_refiner_20260930 import center_safe_loss
from directional_center_refiner_20261001 import DirectionalCenterRefiner,DirectionalCenterCapture,subset_features
from head_native_decision_guard_20261001 import NativeDecisionGuardMixin
from head_quality_features_20260930 import HeadQualityCapture

NAME='baseline_base_head_directional_center_guard_bs16_30ep_20261001'

def fixed_snapshot(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()
        if k.startswith('backbone.') or k.endswith(('running_mean','running_var','num_batches_tracked'))}

def verify_joint(net,before=None):
    assert all(not p.requires_grad and p.grad is None for p in net.backbone.parameters())
    assert not net.backbone.training
    assert all(p.requires_grad for p in net.box_head.parameters())
    assert all(p.requires_grad for p in net.output_refiner.parameters())
    assert all(not m.training for m in net.box_head.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
    if before is not None:
        state=net.state_dict()
        for name,value in before.items():assert torch.equal(value,state[name].detach().cpu()),'Fixed state changed: '+name

class DirectionalCenterCore(UVLTrackActor):
    check_gradients=False
    last_losses=None
    def train(self,mode=True):
        self.net.train(mode)
        self.net.backbone.eval()
        # Eval does not freeze BN affine parameters, only its running statistics.
        for module in self.net.box_head.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    def __call__(self,data):
        self.last_losses=None
        with DirectionalCenterCapture(self.net.box_head) as capture:
            out=self.forward_pass(data)  # Head gradient retained; no global no_grad.
            n,b,h,w=data['search_cls'].shape
            gt_bbox=data['search_anno'].reshape(n*b,4)
            gt_cls=data['search_cls'].reshape(n*b,h,w)
            head_loss,head_status=self.compute_losses(out,gt_bbox,gt_cls,
                self.cont_gt(gt_bbox,data['search_images'].shape[-1]//16),
                epoch=data.get('epoch',1),num_search_frames=n,
                dataset_ids=data['dataset_id'].repeat(n,1,1).reshape(n*b) if 'dataset_id' in data else None)
            active=out['flag'].reshape(-1).eq(2)
            if active.any():
                indices=select_native(out).to(out['bbox_map'].device)
                boxes=out['bbox_map'].detach().gather(1,indices[...,None].expand(-1,-1,4))[:,0]
                features=subset_features(capture.gather_directional(indices,boxes[:,None]),(slice(None),0))
                prediction=self.net.output_refiner(subset_features(features,active),boxes[active])
                assert torch.equal(prediction['boxes'][:,2:],boxes[active,2:])
                base=decode_training_boxes(boxes[active],data,active)
                refined=decode_training_boxes(prediction['boxes'],data,active)
                truth=data['center_gt'].reshape(-1,4)[active]
                hw=data['center_hw'].reshape(-1,2)[active]
                valid=torch.isfinite(truth).all(-1)&(truth[:,2:]>0).all(-1)
                center_loss,center_status=center_safe_loss(refined,base,truth,hw,valid)
                center_status={'Center/'+k:float(v) for k,v in center_status.items()}
                center_status['Center/delta_abs']=float(prediction['delta'].detach().abs().mean())
            else:
                # No refiner graph: AdamW must not advance refiner moments/decay
                # on a purely BBOX/NL batch.
                center_loss=head_loss.detach()*0
                center_status={'Center/iou_gain':0.,'Center/delta_abs':0.}
        loss=head_loss+center_loss
        stats={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in head_status.items()}
        stats.update(center_status)
        stats.update({'Loss/total':float(loss.detach()),'Loss/head_total':float(head_loss.detach()),
            'Loss/center_total':float(center_loss.detach()),'Head/samples':float(n*b),
            'Center/active_samples':float(active.sum())})
        if self.check_gradients:self.last_losses=(head_loss,center_loss)
        return loss,stats

class JointCenterActor(NativeDecisionGuardMixin,DirectionalCenterCore):
    pass



def register_config():
    from lib.config.uvltrack.config import cfg
    register_base_config()
    cfg.MODEL.CENTER_OUTPUT.DESIGN='directional_center_pair'
    cfg.TRAIN.NATIVE_DECISION_BOX_WEIGHT=.5
    cfg.TRAIN.NATIVE_DECISION_RANK_WEIGHT=.25
