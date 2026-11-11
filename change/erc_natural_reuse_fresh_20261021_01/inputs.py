"""Reuse already-computed source tensors. No RGB image/crop/Backbone call."""
import torch
from change.erc_rqc30_dense_curve_20261020_01.evidence import crop_geometry
from change.erc_rqc30_dense_curve_20261020_01.inputs import map_mask,boundary_support
from .geometry import action_lattice,candidates

INPUT_KEYS={'search','head','template_roles','native_extent','source_mask','source_context',
            'candidate_boxes','crop_boxes','saved_crop_boxes','valid'}


@torch.no_grad()
def template_roles(template,foreground):
    if template.ndim!=3 or template.shape[1:]!=(64,768) or foreground.shape!=(len(template),64):
        raise ValueError('Expected existing 8x8 initialization template tokens/mask')
    if foreground.dtype!=torch.bool or not torch.isfinite(template).all():
        raise ValueError('Invalid template tokens/support')
    if not foreground.any(1).all() or not (~foreground).any(1).all():
        raise ValueError('Both legal initial FG and BG are required')
    values=[]
    for mask in (foreground,~foreground):
        values.append((template.detach()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None])
    return torch.stack(values,1).detach()


@torch.no_grad()
def build_inputs(search,head,roles,bbox_map,fixed,origin,side,hw,context):
    b=len(fixed)
    if search.shape!=(b,256,768) or head.shape!=(b,32,16,16) or roles.shape!=(b,2,768):
        raise ValueError('Wrong captured search/Head/template schema')
    if bbox_map.shape!=(b,256,4) or context.shape!=(b,64):
        raise ValueError('Wrong native extent/context schema')
    fixed,origin,side,hw=(v.detach().double() for v in (fixed,origin,side,hw))
    if not torch.stack([torch.isfinite(v).all() for v in (fixed,origin,side,hw,roles,context)]).all():
        raise ValueError('Nonfinite source geometry/context')
    if not fixed[:,2:].gt(0).all() or not side.gt(0).all():
        raise ValueError('Nonpositive source geometry')
    keep,bounds=crop_geometry(fixed,origin,side,hw)
    mask=map_mask(bounds,16,16)
    sm=mask.flatten(2).transpose(1,2)
    search=torch.where(sm,search.detach().float(),torch.zeros_like(search,dtype=torch.float))
    head=torch.where(mask,head.detach().float(),torch.zeros_like(head,dtype=torch.float))
    wh=bbox_map.detach().float()[...,2:].transpose(1,2).reshape(b,2,16,16)
    wh=torch.where(mask,wh,torch.ones_like(wh))
    if not torch.isfinite(wh).all() or not wh.gt(0).all():
        raise ValueError('Invalid visible native widths/heights')
    # Already-predicted native extents relative to the fixed RQC width/height.
    extent=torch.where(mask,wh.log()-keep[:,2:,None,None].log(),torch.zeros_like(wh))
    boxes=candidates(fixed,hw,action_lattice().to(fixed.device))
    repeat=lambda v:v[:,None].expand(-1,25,*v.shape[1:]).reshape(b*25,*v.shape[1:])
    cb,bb=crop_geometry(boxes.reshape(-1,4),repeat(origin),repeat(side),repeat(hw))
    cb,bb=cb.reshape(b,25,4),bb.reshape(b,25,4)
    # Do not round in crop coordinates: preserve the official image .long() writer first.
    saved,_=crop_geometry(boxes.long().double().reshape(-1,4),repeat(origin),repeat(side),repeat(hw))
    valid=boundary_support(cb,bb)
    result=dict(search=search.transpose(1,2).reshape(b,768,16,16),head=head,
        template_roles=roles.detach(),native_extent=extent,source_mask=mask,
        source_context=context.detach(),candidate_boxes=boxes,crop_boxes=cb,
        saved_crop_boxes=saved.reshape(b,25,4),valid=valid)
    if set(result)!=INPUT_KEYS or not torch.stack([torch.isfinite(v).all() for v in result.values()]).all():
        raise ValueError('Nonfinite/missing lightweight input')
    return {k:v.detach() for k,v in result.items()}
