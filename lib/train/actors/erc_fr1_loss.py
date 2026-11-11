"""GT-only foreground ranking; no IoU target or gate supervision."""
import torch
from torch.nn import functional as F


def center_ordered_rank(score, gt_bbox, active, margin):
    """Rank deployed response by GT-center proximity, NOT predicted box IoU.

    No ignored interior/boundary band. Equal-distance grid cells are tied;
    nearest supported cells are reachable even when Hann removes the border.
    This is a response-localization surrogate, not a guarantee of box quality.
    """
    gt = gt_bbox.detach().to(score)
    n = int(score.shape[-1] ** .5)
    if n*n != score.shape[-1]:
        raise ValueError('Expected a square response grid')
    center = gt[:, :2] + gt[:, 2:]/2
    valid = torch.isfinite(gt).all(-1) & gt[:, 2:].gt(0).all(-1)
    valid &= center.ge(0).all(-1) & center.lt(1).all(-1) & active.bool().reshape(-1)
    clean = torch.nan_to_num(gt, nan=0., posinf=0., neginf=0.)
    center = clean[:, :2] + clean[:, 2:]/2
    y, x = torch.meshgrid((torch.arange(n, device=score.device, dtype=score.dtype)+.5)/n,
                         (torch.arange(n, device=score.device, dtype=score.dtype)+.5)/n, indexing='ij')
    xy = torch.stack((x.flatten(), y.flatten()), -1)
    scale = (clean[:, 2:]/2).clamp_min(1/n)
    distance = ((xy[None]-center[:, None])/scale[:, None]).square().sum(-1)
    support = score.detach().gt(0) & torch.isfinite(score.detach())
    minimum = distance.masked_fill(~support, float('inf')).amin(-1, keepdim=True)
    anchor = support & torch.isclose(distance, minimum, rtol=1e-6, atol=1e-10)
    valid &= anchor.any(-1)
    quality = torch.exp(-.5*distance)
    best_quality = quality.masked_fill(~anchor, 0).amax(-1, keepdim=True)
    gaps = (best_quality-quality).clamp_min(0)
    logs = score.clamp_min(torch.finfo(score.dtype).tiny).log()
    low = torch.finfo(score.dtype).min
    anchor_score = logs.masked_fill(~anchor, low).amax(-1)
    competitor = (logs+float(margin)*gaps).masked_fill(~support, low).amax(-1)
    per_sample = F.relu(competitor-anchor_score)
    loss = per_sample[valid].mean() if valid.any() else score.sum()*0
    return loss, valid


def compute_fr1_loss(pred, gt_bbox, cfg):
    score = pred['erc_windowed_score']
    base = pred['erc_windowed_base_score'].detach()
    gt = gt_bbox.detach().to(score)
    n = int(score.shape[-1]**.5)
    y,x = torch.meshgrid((torch.arange(n,device=score.device,dtype=score.dtype)+.5)/n,
                         (torch.arange(n,device=score.device,dtype=score.dtype)+.5)/n,indexing='ij')
    x=x.flatten()[None]; y=y.flatten()[None]
    center=gt[:,:2]+gt[:,2:]/2
    valid=torch.isfinite(gt).all(-1)&gt[:,2:].gt(0).all(-1)&center.ge(0).all(-1)&center.lt(1).all(-1)
    valid &= pred['erc_active_mask'].bool()
    # Inner 75% foreground, exterior beyond bbox + one grid cell; ignore boundary band.
    half=.375*gt[:,2:]
    positive=(x-center[:,0:1]).abs().le(half[:,0:1]) & (y-center[:,1:2]).abs().le(half[:,1:2])
    # Tiny targets may contain no grid center: include nearest in-crop center.
    index=((center[:,1].nan_to_num().clamp(0,1-1/n)*n).long()*n + (center[:,0].nan_to_num().clamp(0,1-1/n)*n).long())
    positive.scatter_(1,index[:,None],True)
    outer=gt[:,2:]/2+1/n
    negative=((x-center[:,0:1]).abs()>outer[:,0:1]) | ((y-center[:,1:2]).abs()>outer[:,1:2])
    support=base.gt(0)
    positive &= support; negative &= support
    mask=valid & positive.any(-1) & negative.any(-1)
    logs=score.clamp_min(torch.finfo(score.dtype).tiny).log()
    lo=torch.finfo(logs.dtype).min
    pos=logs.masked_fill(~positive,lo).amax(-1)
    neg=logs.masked_fill(~negative,lo).amax(-1)
    loss=F.relu(float(cfg.TRAIN.ERC_FR1_MARGIN)+neg-pos)[mask].mean() if mask.any() else score.sum()*0
    foreground_loss=loss
    if getattr(cfg.TRAIN, 'ERC_FR1_CENTER_ORDERED', False):
        loss,mask=center_ordered_rank(score,gt,pred['erc_active_mask'],cfg.TRAIN.ERC_FR1_MARGIN)
    total=float(cfg.TRAIN.ERC_FR1_RANK_WEIGHT)*loss
    with torch.no_grad():
        bi=base.argmax(-1); si=score.argmax(-1); changed=valid & bi.ne(si)
        bb=pred['bbox_map'].detach(); row=torch.arange(bb.shape[0],device=bb.device)
        def iou(b):
            a=b[:,:2]-b[:,2:]/2; z=b[:,:2]+b[:,2:]/2
            inter=(torch.minimum(z,gt[:,:2]+gt[:,2:])-torch.maximum(a,gt[:,:2])).clamp_min(0).prod(-1)
            return inter/(b[:,2:].prod(-1)+gt[:,2:].prod(-1)-inter).clamp_min(1e-8)
        gain=iou(bb[row,si])-iou(bb[row,bi]); denom=valid.sum().clamp_min(1)
        info=pred['erc_fr1']
        status={'Loss/erc_total':total.detach(),'Loss/fr1_rank':loss.detach(),
                'Loss/fr1_foreground_diagnostic':foreground_loss.detach(),
                'ERC/fr1_valid_count':valid.sum().float(),'ERC/fr1_rank_count':mask.sum().float(),
                'ERC/fr1_changed_count':changed.sum().float(),'ERC/fr1_changed_rate':changed.sum()/denom,
                'ERC/fr1_box_gain_sum':gain[valid].sum(),'ERC/fr1_box_gain':gain[valid].sum()/denom,
                'ERC/fr1_beneficial_rate':(changed & gain.gt(1e-6)).sum()/denom,
                'ERC/fr1_harmful_rate':(changed & gain.lt(-1e-6)).sum()/denom,
                'ERC/fr1_relative_delta':info['relative_norm'][valid].mean() if valid.any() else total.detach()*0,
                'ERC/fr1_foreground_tokens':info['foreground_count'].float().mean()}
    return total,status
