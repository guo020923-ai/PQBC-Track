"""Training-only, continuous spatial evidence targets and Head rank preservation.

Uses GT center/extent, never candidate IoU or predicted geometry. Native Head
classification labels remain unchanged. These targets are not IoU estimates.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def spatial_quality(gt_xywh, height, width, dtype=torch.float32):
    gt = gt_xywh.detach().to(dtype=dtype)
    valid = torch.isfinite(gt).all(-1) & gt[:, 2:].gt(0).all(-1)
    center = gt[:, :2] + .5 * gt[:, 2:]
    valid &= center.ge(0).all(-1) & center.lt(1).all(-1)
    gt = torch.where(torch.isfinite(gt), gt, torch.zeros_like(gt))
    center = gt[:, :2] + .5 * gt[:, 2:]
    # Cell centers, continuous GT (no floor/round); one-cell minimum scale.
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=gt.device, dtype=dtype)+.5)/height,
        (torch.arange(width, device=gt.device, dtype=dtype)+.5)/width, indexing='ij')
    scale = torch.maximum(.5 * gt[:, 2:], gt.new_tensor([1./width, 1./height]))
    distance = ((xx[None]-center[:,0,None,None])/scale[:,0,None,None]).square()
    distance += ((yy[None]-center[:,1,None,None])/scale[:,1,None,None]).square()
    quality = 1. / (1. + distance)
    quality = torch.where(valid[:,None,None], quality, torch.zeros_like(quality))
    return quality, valid


def teacher_peak_preservation(pred, gt_bbox, cfg):
    """Protect a better teacher location only when the student reverses its rank.

    Teacher and quality are detached. Gradient updates student Head through its
    PRE-ERC scores; ERC is not trained to imitate the teacher. Existing KL is kept.
    """
    student = pred['erc_windowed_base_score']
    zero = student.sum()*0
    required = 'head_teacher_windowed_base_score'
    if required not in pred:
        raise ValueError('DR4a Head peak preservation requires the official Head teacher')
    teacher = pred[required].detach().to(student)
    n = student.shape[-1]; size = int(n**.5)
    quality, valid = spatial_quality(gt_bbox, size, size, student.dtype)
    quality = quality.flatten(1)
    valid &= pred['erc_active_mask'].reshape(-1).bool()
    row = torch.arange(student.shape[0], device=student.device)
    ti, si = teacher.argmax(-1), student.detach().argmax(-1)
    gain = quality[row,ti]-quality[row,si]
    mask = valid & gain.gt(float(cfg.TRAIN.ERC_DR4_MIN_GAIN)) & ti.ne(si)
    tiny = torch.finfo(student.dtype).tiny
    slog, tlog = student.clamp_min(tiny).log(), teacher.clamp_min(tiny).log()
    margin = (tlog[row,ti]-tlog[row,si]).clamp(0,float(cfg.TRAIN.ERC_DR4_MARGIN))
    element = F.relu(margin + slog[row,si] - slog[row,ti])
    loss = element[mask].mean() if mask.any() else zero
    return loss, {'Loss/head_peak_preserve':loss.detach(),
                  'Head/peak_preserve_count':mask.sum().float(),
                  'Head/peak_preserve_valid_count':valid.sum().float()}
