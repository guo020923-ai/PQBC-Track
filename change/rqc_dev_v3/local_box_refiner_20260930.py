"""Isolated prototype: refine the response-selected box, never select a candidate.

Not registered with the tracker. Inputs are normalized search-crop cx,cy,w,h
and the 288-dimensional passive HeadQualityCapture feature at that location.
Backbone/Head inputs are detached; the original tracker remains unchanged.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LocalBoxRefiner(nn.Module):
    def __init__(self, center_limit=0.1, log_size_limit=math.log(1.2)):
        super().__init__()
        if center_limit <= 0 or log_size_limit <= 0:
            raise ValueError('Residual limits must be positive')
        self.center_limit = float(center_limit)
        self.log_size_limit = float(log_size_limit)
        self.net = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, 64),
                                 nn.GELU(), nn.Linear(64, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, boxes):
        if features.shape[:-1] != boxes.shape[:-1] or features.shape[-1] != 288 or boxes.shape[-1] != 4:
            raise ValueError('Expected matching [...,288] features and [...,4] boxes')
        if not torch.isfinite(features).all() or not torch.isfinite(boxes).all() or (boxes[..., 2:] <= 0).any():
            raise ValueError('Nonfinite inputs or nonpositive box size')
        boxes = boxes.detach()
        raw = self.net(features.detach()).tanh()
        delta = torch.cat((raw[..., :2] * self.center_limit,
                           raw[..., 2:] * self.log_size_limit), -1)
        refined = torch.cat((boxes[..., :2] + delta[..., :2] * boxes[..., 2:],
                             boxes[..., 2:] * delta[..., 2:].exp()), -1)
        return {'boxes': refined, 'delta': delta}


def residual_target(boxes, gt):
    """Unclipped target: do NOT silently label unreachable corrections reachable."""
    boxes, gt = boxes.detach(), gt.detach()
    if boxes.shape != gt.shape or (boxes[..., 2:] <= 0).any() or (gt[..., 2:] <= 0).any():
        raise ValueError('Expected matching positive-size cxcywh boxes')
    return torch.cat(((gt[..., :2] - boxes[..., :2]) / boxes[..., 2:],
                       (gt[..., 2:] / boxes[..., 2:]).log()), -1)


def diagnostic_loss(pred, boxes, gt, center_limit=0.1, log_size_limit=math.log(1.2)):
    """Gradient check only, NOT a finalized training objective or sample policy."""
    target = residual_target(boxes, gt)
    limit = target.new_tensor([center_limit, center_limit, log_size_limit, log_size_limit])
    reachable = (target.abs() <= limit).all(-1)
    loss = F.smooth_l1_loss(pred['delta'] / limit, target.clamp(-limit, limit) / limit)
    return loss, {'reachable_rate': reachable.float().mean(),
                  'target_clipped_rate': (target.abs() > limit).float().mean()}


def decode_image_boxes(boxes, origin, side, image_hw, margin=10.):
    """Differentiable equivalent of crop mapping followed by official clip_box.

    Inputs [B,4] cxcywh, [B,2] crop origin, [B] crop side, [B,2] H,W.
    Outputs image xywh. No GT is involved in decoding.
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('Expected [B,4]')
    origin, side, image_hw = origin.detach(), side.detach(), image_hw.detach()
    if (image_hw < margin).any() or (side <= 0).any():
        raise ValueError('Image too small for tracker margin or invalid crop')
    size = boxes[:, 2:] * side[:, None]
    lo = origin + (boxes[:, :2] - boxes[:, 2:] / 2) * side[:, None]
    hi = lo + size
    wh = image_hw.flip(-1)
    lo = torch.minimum(lo.clamp_min(0), wh - margin)
    hi = torch.minimum(hi.clamp_min(margin), wh)
    return torch.cat((lo, (hi-lo).clamp_min(margin)), -1)


def localization_loss(refined, baseline, gt_xywh, image_hw, valid):
    """Same 2*GIoU + 5*L1 weights as baseline; full GT, no clipped residual labels.

    All boxes image xywh, [B,4]. Loss operates only on visible, annotated frames.
    baseline is diagnostic only: no learned gate or GT inference selection.
    L1 uses image-normalized xyxy (explicitly not the original crop L1 units).
    """
    gt = gt_xywh.detach(); baseline = baseline.detach()
    keep = valid.detach().bool()
    if keep.ndim != 1 or keep.shape[0] != refined.shape[0]:
        raise ValueError('Expected per-frame validity mask')
    if not keep.any():
        return refined.sum()*0, {'valid_count': 0}
    r, g, b = refined[keep], gt[keep], baseline[keep]
    if not torch.isfinite(g).all() or (g[:, 2:] <= 0).any():
        raise ValueError('Invalid visible GT')
    def corners(box): return torch.cat((box[:, :2], box[:, :2]+box[:, 2:]), -1)
    def overlap(a, target):
        x, y = corners(a), corners(target)
        inter = (torch.minimum(x[:,2:],y[:,2:])-torch.maximum(x[:,:2],y[:,:2])).clamp_min(0).prod(-1)
        union = a[:,2:].prod(-1)+target[:,2:].prod(-1)-inter
        iou = inter/union.clamp_min(1e-8)
        enclosing = (torch.maximum(x[:,2:],y[:,2:])-torch.minimum(x[:,:2],y[:,:2])).prod(-1).clamp_min(1e-8)
        return iou, iou-(enclosing-union)/enclosing
    iou, giou = overlap(r,g)
    base_iou, _ = overlap(b,g)
    norm = image_hw[keep].detach().flip(-1).repeat(1,2)
    l1 = F.l1_loss(corners(r)/norm, corners(g)/norm)
    loss = 2*(1-giou).mean()+5*l1
    gain = iou.detach()-base_iou
    return loss, {'valid_count': int(keep.sum()), 'iou_gain': gain.mean(),
                  'harmful_rate': (gain < -.01).float().mean(),
                  'severe_harmful_rate': (gain < -.2).float().mean(),
                  'beneficial_rate': (gain > .01).float().mean()}
