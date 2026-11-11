"""Dense ordered ROI from captured tensors only. Current GT has no entry."""
import math
import torch
from torch.nn import functional as F
from .geometry import action_lattice,candidates
from .evidence import crop_geometry,pixel_evidence,template_association_map

ROI_CHANNELS=951
ROI_SIDE=16
ROI_SPAN=1.5

@torch.no_grad()
def boundary_support(boxes,bounds):
    # Identical old boundary locations; no boundary feature encoding is used.
    b,k=boxes.shape[:2]
    offsets=boxes.new_zeros(2,2,2,5,2)
    tangent=boxes.new_tensor([-.4,-.2,0.,.2,.4])
    for axis in range(2):
        for side,sign in enumerate((-1.,1.)):
            for region,radius in enumerate((.5*math.exp(-.05),.5*math.exp(.05))):
                offsets[axis,side,region,:,axis]=sign*radius
                offsets[axis,side,region,:,1-axis]=tangent
    points=(boxes[...,:2]+boxes[...,2:]/2.)[:,:,None,None,None,None]+offsets[None,None]*boxes[:,:,None,None,None,None,2:]
    return ((points>=bounds[:,:,None,None,None,None,:2])&(points<=bounds[:,:,None,None,None,None,2:])&
            (points>=0)&(points<=1)).all(-1)

@torch.no_grad()
def map_mask(bounds,h,w):
    y=(torch.arange(h,device=bounds.device,dtype=bounds.dtype)+.5)/h
    x=(torch.arange(w,device=bounds.device,dtype=bounds.dtype)+.5)/w
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    points=torch.stack((xx,yy),-1)[None]
    return ((points>=bounds[:,None,None,:2])&(points<=bounds[:,None,None,2:])).all(-1)[:,None]

@torch.no_grad()
def sample_map(field,grid,bounds):
    # Never multiply NaN by zero: remove invalid map cells before interpolation.
    m=map_mask(bounds,*field.shape[-2:])
    safe=torch.where(m,field.detach().float(),torch.zeros_like(field,dtype=torch.float))
    if not torch.isfinite(safe).all():raise ValueError('Nonfinite real source map')
    return F.grid_sample(safe,grid,align_corners=False,padding_mode='zeros')

@torch.no_grad()
def build_inputs(search,template,foreground,bbox_features,fixed_xywh,origin,side,image_hw,
                 source_context,search_rgb,template_rgb,bbox_map,actions,
                 inherited_valid=None,inherited_executed_delta=None):
    b=len(fixed_xywh)
    if search.shape!=(b,256,768) or template.shape!=(b,64,768) or bbox_features.shape!=(b,32,16,16):
        raise ValueError('Wrong frozen source token/Head schema')
    if bbox_map.shape!=(b,256,4) or source_context.shape!=(b,64):
        raise ValueError('Wrong native extent/context schema')
    if foreground.shape!=(b,64) or foreground.dtype!=torch.bool:
        raise ValueError('Expected legal initialization FG mask')
    if not torch.equal(actions,action_lattice().to(actions.device)):raise ValueError('Wrong action order')
    if not all(torch.isfinite(v).all() for v in (fixed_xywh,origin,side,image_hw,source_context,template,template_rgb)):
        raise ValueError('Nonfinite fixed geometry/context/template')
    box,bounds=crop_geometry(fixed_xywh,origin,side,image_hw)
    axis=((torch.arange(16,device=search.device,dtype=torch.float)+.5)/16-.5)*ROI_SPAN
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    relative=torch.stack((xx,yy),0)[None].expand(b,-1,-1,-1)
    points=box[:,None,None,:2]+box[:,None,None,2:]/2+relative.permute(0,2,3,1)*box[:,None,None,2:]
    mask=((points>=bounds[:,None,None,:2])&(points<=bounds[:,None,None,2:])&
          (points>=0)&(points<=1)).all(-1)[:,None]
    grid=points*2-1
    source_mask=map_mask(bounds,16,16)
    safe_search=torch.where(source_mask.flatten(2).transpose(1,2),search.detach().float(),torch.zeros_like(search,dtype=torch.float))
    association=template_association_map(safe_search,template,foreground)
    # Pixel neighborhoods must not read NaNs in padded pixels.
    rgb_mask=map_mask(bounds,256,256)
    safe_rgb=torch.where(rgb_mask,search_rgb.detach().float(),torch.zeros_like(search_rgb,dtype=torch.float))
    if not torch.isfinite(safe_rgb).all():raise ValueError('Nonfinite real source RGB')
    pixels=pixel_evidence(safe_rgb,template_rgb,foreground)
    native=bbox_map.detach().float()[...,2:].transpose(1,2).reshape(b,2,16,16)
    native=torch.where(source_mask,native,torch.ones_like(native))
    if not torch.isfinite(native).all() or not native.gt(0).all():raise ValueError('Invalid real native extent')
    log_native=sample_map(native.log(),grid,bounds)
    support=sample_map(torch.ones_like(native[:,:1]),grid,bounds)
    relative_native=log_native-support*box[:,2:,None,None].log()
    fields=(sample_map(safe_search.transpose(1,2).reshape(b,768,16,16),grid,bounds),
            sample_map(association,grid,bounds),sample_map(bbox_features,grid,bounds),
            sample_map(pixels,grid,bounds),relative_native,relative,mask.float())
    evidence=torch.cat(fields,1)
    evidence=torch.where(mask,evidence,torch.zeros_like(evidence))
    if evidence.shape!=(b,951,16,16) or not torch.isfinite(evidence).all():raise ValueError('Invalid ROI evidence')
    executed=candidates(fixed_xywh,image_hw,actions)
    delta=(executed[...,2:]/fixed_xywh[:,None,2:]).log().float()
    crop_boxes,crop_bounds=crop_geometry(executed.reshape(-1,4),origin[:,None].expand(-1,25,-1).reshape(-1,2),
        side[:,None].expand(-1,25).reshape(-1),image_hw[:,None].expand(-1,25,-1).reshape(-1,2))
    valid=boundary_support(crop_boxes.reshape(b,25,4),crop_bounds.reshape(b,25,4))
    if inherited_valid is not None and (inherited_valid.dtype!=torch.bool or not torch.equal(inherited_valid,valid)):
        raise ValueError('Original collector action support mismatch')
    if inherited_executed_delta is not None and not torch.allclose(inherited_executed_delta.detach().double(),delta.double(),rtol=1e-6,atol=1e-7):
        raise ValueError('Original collector executed delta mismatch')
    return dict(roi_evidence=evidence.detach(),roi_valid=mask.detach(),valid=valid.detach(),
                source_context=source_context.detach(),executed_delta=delta.detach())
