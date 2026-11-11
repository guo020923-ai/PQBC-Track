"""Detached source evidence. No GT, tracker, weights or dataset imports."""
import torch
from torch.nn import functional as F

@torch.no_grad()
def pixel_evidence(search_rgb,template_rgb,foreground):
    """High-resolution local evidence, not an additional trainable Backbone.

    Use the SAME normalized source patches and known initialization mask.
    Preserve 256x256 resolution until point sampling; do not upsample 16x16
    features and call them high-resolution observations.
    """
    b=len(search_rgb)
    if search_rgb.shape!=(b,3,256,256) or template_rgb.shape!=(b,3,128,128):raise ValueError('Wrong RGB patch schema')
    q=search_rgb.detach().float();r=template_rgb.detach().float()
    mask=F.interpolate(foreground.detach().float().reshape(b,1,8,8),size=(128,128),mode='nearest')
    if not mask.sum((2,3)).gt(0).all() or not (1-mask).sum((2,3)).gt(0).all():raise ValueError('Missing RGB reference support')
    fg=(r*mask).sum((2,3))/mask.sum((2,3));bg=(r*(1-mask)).sum((2,3))/(1-mask).sum((2,3))
    mean=F.avg_pool2d(q,5,stride=1,padding=2)
    std=(F.avg_pool2d(q.square(),5,stride=1,padding=2)-mean.square()).clamp_min(0).sqrt()
    kernel=q.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/8.
    dx=F.conv2d(q,kernel[None,None].expand(3,1,3,3),padding=1,groups=3).abs()
    dy=F.conv2d(q,kernel.t()[None,None].expand(3,1,3,3),padding=1,groups=3).abs()
    df=(q-fg[:,:,None,None]).square().mean(1,keepdim=True)
    db=(q-bg[:,:,None,None]).square().mean(1,keepdim=True)
    return torch.cat((q,mean,std,dx,dy,df,db,db-df),1).detach()

@torch.no_grad()
def crop_geometry(fixed_xywh, origin, side, image_hw, flip=None):
    """Convert unchanged output to augmented crop coordinates, without GT."""
    p, origin, side, hw = (x.detach().float() for x in (fixed_xywh, origin, side, image_hw))
    if p.shape != (len(p), 4) or origin.shape != (len(p), 2) or side.shape != (len(p),) or hw.shape != (len(p), 2):
        raise ValueError("Invalid crop geometry schema")
    if (side <= 0).any() or (p[:, 2:] <= 0).any(): raise ValueError("Nonpositive geometry")
    box = torch.cat(((p[:, :2] - origin) / side[:, None], p[:, 2:] / side[:, None]), 1)
    real_lo = torch.maximum(-origin / side[:, None], torch.zeros_like(origin))
    real_hi = torch.minimum((hw.flip(-1) - origin) / side[:, None], torch.ones_like(origin))
    bounds = torch.cat((real_lo, real_hi), 1)
    if flip is not None:
        flip = flip.detach().bool()
        if flip.shape != (len(p),): raise ValueError("Wrong reflection mask")
        box[:, 0] = torch.where(flip, 1. - box[:, 0] - box[:, 2], box[:, 0])
        xlo, xhi = bounds[:, 0].clone(), bounds[:, 2].clone()
        bounds[:, 0] = torch.where(flip, 1. - xhi, xlo)
        bounds[:, 2] = torch.where(flip, 1. - xlo, xhi)
    return box, bounds

@torch.no_grad()
def template_association_map(search,template,foreground):
    if search.ndim!=3 or search.shape[1]!=256 or template.shape!=(len(search),64,search.shape[2]):
        raise ValueError('Expected search 16x16 and ordered template 8x8 tokens')
    if foreground.shape!=(len(search),64):
        raise ValueError('Wrong ordered initialization FG mask')
    mask=foreground.detach().bool()
    if not mask.any(1).all() or not (~mask).any(1).all():
        raise ValueError('Both template roles require support')
    if not torch.isfinite(search).all() or not torch.isfinite(template).all():
        raise ValueError('Nonfinite template/search token')
    q=F.normalize(search.detach().float(),dim=-1)
    t=F.normalize(template.detach().float(),dim=-1)
    similarity=q@t.transpose(1,2)
    # Channel j retains template row-major location j. FG and BG are distinct
    # banks; empty role/location pairs are exactly zero. No current-frame GT.
    bank=torch.cat((similarity*mask[:,None],similarity*(~mask)[:,None]),-1)
    return bank.transpose(1,2).reshape(len(search),128,16,16).detach()
