"""One-factor objective variant for output-only local refinement.

Retain geometry loss; add direct realized-IoU deterioration cost. No learned
quality/gate, no GT inference input, and no assertion of guaranteed improvement.
"""
import torch
from local_box_refiner_20260930 import localization_loss


def aligned_iou(a,b):
    lo=torch.maximum(a[:,:2],b[:,:2])
    hi=torch.minimum(a[:,:2]+a[:,2:],b[:,:2]+b[:,2:])
    inter=(hi-lo).clamp_min(0).prod(-1)
    union=a[:,2:].prod(-1)+b[:,2:].prod(-1)-inter
    return inter/union.clamp_min(1e-8)


def relative_localization_loss(refined,baseline,gt_xywh,image_hw,valid,harm_weight=1.):
    if harm_weight<0:raise ValueError('Negative harm weight')
    geometry,stats=localization_loss(refined,baseline,gt_xywh,image_hw,valid)
    mask=valid.detach().bool()
    if not mask.any():return geometry,dict(stats,relative_harm=geometry.detach(),geometry=geometry.detach())
    current=aligned_iou(refined[mask],gt_xywh.detach()[mask])
    original=aligned_iou(baseline.detach()[mask],gt_xywh.detach()[mask])
    # Zero margin: no artificial positive-gain target and no penalty on harmless
    # unchanged outputs. ReLU subgradient at equality is zero; geometry still
    # supplies the initial learning signal from the zero-initialized refiner.
    harm=torch.relu(original-current).mean()
    loss=geometry+harm_weight*harm
    return loss,dict(stats,geometry=geometry.detach(),relative_harm=harm.detach())
