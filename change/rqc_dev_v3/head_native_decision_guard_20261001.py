"""Training-only geometric/decision regret, not dense response distillation.

The frozen official Head runs on the SAME frozen-backbone inputs. GT is used
only during training/validation, never to choose the inference output. Improved
student decisions are not required to copy the teacher's scores or boxes.
"""
import math,copy
import numpy as np
import torch
from torch.nn import functional as F
from center_formal_contract_20260930 import select_native,decode_training_boxes
from relative_local_objective_20260930 import aligned_iou

def detach_tree(value):
    if torch.is_tensor(value):return value.detach()
    if isinstance(value,dict):return {k:detach_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [detach_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(detach_tree(v) for v in value)
    return value

def giou(box,gt):
    iou=aligned_iou(box,gt)
    enclosing=(torch.maximum(box[:,:2]+box[:,2:],gt[:,:2]+gt[:,2:])-
               torch.minimum(box[:,:2],gt[:,:2])).clamp_min(0).prod(-1).clamp_min(1e-8)
    inter=(torch.minimum(box[:,:2]+box[:,2:],gt[:,:2]+gt[:,2:])-
           torch.maximum(box[:,:2],gt[:,:2])).clamp_min(0).prod(-1)
    union=box[:,2:].prod(-1)+gt[:,2:].prod(-1)-inter
    return iou-(enclosing-union)/enclosing,iou

def score(out):
    c=out['cls_score_test'].flatten(1)
    t=out['cont_score'].softmax(-1)[...,0]
    side=math.isqrt(c.shape[-1]);assert side*side==c.shape[-1]
    hann=c.new_tensor(np.outer(np.hanning(side),np.hanning(side)).reshape(-1),dtype=torch.float64)
    return c.to(torch.float64)*hann*t.to(torch.float64)

def decision_guard(student,teacher,data,box_weight=.5,rank_weight=.25):
    active=student['flag'].reshape(-1).eq(2)
    zero=student['bbox_map'].sum()*0
    if not active.any():return zero,{'HeadDecision/active':0.}
    si=select_native(student).to(student['bbox_map'].device)
    ti=select_native(teacher).to(si.device)
    def selected(out,index):return out['bbox_map'].gather(1,index[...,None].expand(-1,-1,4))[:,0]
    s=decode_training_boxes(selected(student,si)[active],data,active)
    t=decode_training_boxes(selected(teacher,ti)[active].detach(),data,active)
    alternative=decode_training_boxes(selected(student,ti)[active],data,active)
    gt=data['center_gt'].reshape(-1,4)[active].detach()
    valid=torch.isfinite(gt).all(-1)&(gt[:,2:]>0).all(-1)
    if not valid.any():return zero,{'HeadDecision/active':0.}
    sg,sq=giou(s[valid],gt[valid]);tg,tq=giou(t[valid],gt[valid])
    aq=aligned_iou(alternative[valid].detach(),gt[valid])
    # Continuous teacher quality weights, no hard inference threshold. GIoU
    # retains a directional gradient even when student and GT do not overlap.
    quality=tq.detach().square()
    box_regret=(quality*F.relu(tg.detach()-sg)).mean()
    advantage=F.relu(aq-sq.detach())
    logs=score(student).clamp_min(1e-12).log()
    current=logs.gather(1,si)[:,0][active][valid]
    recovery=logs.gather(1,ti)[:,0][active][valid]
    rank_regret=(quality*advantage*F.relu(current-recovery+advantage)).mean()
    loss=box_weight*box_regret+rank_weight*rank_regret
    return loss,{'HeadDecision/active':float(valid.sum()),'HeadDecision/box_regret':float(box_regret.detach()),
        'HeadDecision/rank_regret':float(rank_regret.detach()),'HeadDecision/teacher_iou':float(tq.detach().mean()),
        'HeadDecision/student_iou':float(sq.detach().mean()),
        'HeadDecision/teacher_better_rate':float((tq.detach()>sq.detach()+.01).float().mean())}

class NativeDecisionGuardMixin:
    def __init__(self,net,cfg):
        super().__init__(net,cfg)
        self.decision_teacher=copy.deepcopy(net.box_head).eval()
        for p in self.decision_teacher.parameters():p.requires_grad_(False)
    def __call__(self,data):
        captured={}
        def before(module,inputs):
            with torch.no_grad():captured['teacher']=self.decision_teacher(detach_tree(inputs[0]))
        def after(module,inputs,out):captured['student']=out
        a=self.net.box_head.register_forward_pre_hook(before);b=self.net.box_head.register_forward_hook(after)
        try:loss,stats=super().__call__(data)
        finally:a.remove();b.remove()
        extra,guard_stats=decision_guard(captured['student'],captured['teacher'],data,
            self.cfg.TRAIN.NATIVE_DECISION_BOX_WEIGHT,self.cfg.TRAIN.NATIVE_DECISION_RANK_WEIGHT)
        if self.check_gradients:
            head,center=self.last_losses;self.last_losses=(head+extra,center)
        stats.update(guard_stats);stats['Loss/head_decision']=float(extra.detach())
        stats['Loss/head_total']+=float(extra.detach());stats['Loss/total']+=float(extra.detach())
        return loss+extra,stats
