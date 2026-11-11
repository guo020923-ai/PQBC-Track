"""Independent boundary-contrast RQC prototype; not enabled by any YAML.

Same candidate boxes and gain target as PairGain. Adds explicit inside/edge/
outside feature evidence. Coarse search tokens may still lack localization detail.
"""
import torch
from torch import nn
from torch.nn import functional as F


def boundary_samples(search, boxes):
    search, boxes = search.detach(), boxes.detach()
    b,n,d=search.shape;side=int(n**.5)
    if side*side!=n:raise ValueError('Search tokens must form a square grid')
    # Four sides, three radial bands, three tangential samples per band.
    offsets=[]
    for edge in range(4):
        for radius in (.4,.5,.6):
            for tangent in (-1/3,0.,1/3):
                offsets.append((-radius,tangent) if edge==0 else
                               (radius,tangent) if edge==1 else
                               (tangent,-radius) if edge==2 else (tangent,radius))
    off=boxes.new_tensor(offsets)
    grid=boxes[:,:,None,:2]+off[None,None]*boxes[:,:,None,2:]
    # Only support inside actual token-center extent; no border replication.
    valid=((grid>=.5/side)&(grid<=1-.5/side)).all(-1)
    fmap=search.transpose(1,2).reshape(b,d,side,side)
    values=F.grid_sample(fmap,grid.reshape(b,-1,1,2)*2-1,
                         mode='bilinear',padding_mode='zeros',align_corners=False)
    values=values[...,0].transpose(1,2).reshape(b,boxes.shape[1],4,3,3,d)
    valid=valid.reshape(b,boxes.shape[1],4,3,3)
    values=values*valid[...,None].to(values)
    return values,valid


def candidates(out, template_mask, k=10):
    from .candidate_quality_v2 import candidates as original_candidates
    batch=original_candidates(out,template_mask,k)
    batch['boundary'],batch['boundary_valid']=boundary_samples(out['search'],batch['boxes'])
    return batch


class BoundaryPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(768,8,bias=False)
        self.compare=nn.Sequential(nn.Linear(198,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.compare[-1].weight);nn.init.zeros_(self.compare[-1].bias)

    def forward(self,batch,use_reliability=False):
        if use_reliability:raise ValueError('No ERC in boundary probe')
        region=self.project(F.normalize(batch['spatial'].detach(),dim=-1)).flatten(-2)
        v=batch['boundary_valid'].detach().bool()
        f=self.project(F.normalize(batch['boundary'].detach(),dim=-1))
        bands=(f*v[...,None]).sum(-2)/v.sum(-1,keepdim=True).clamp_min(1)
        support=v.any(-1)
        inside,edge,outside=bands.unbind(-2)
        # Differences are masked when either side lacks support.
        edge_in=(edge-inside)*(support[...,1]&support[...,0])[...,None]
        out_edge=(outside-edge)*(support[...,2]&support[...,1])[...,None]
        boundary=torch.cat((edge,edge_in,out_edge),-1).flatten(-2)
        coverage=v.float().mean(-1).flatten(-2)
        box=batch['boxes'].detach();top=box[:,:1]
        geom=torch.cat(((box[...,:2]-top[...,:2])/((box[...,2:]+top[...,2:])*.5).clamp_min(1e-4),
                        (box[...,2:].clamp_min(1e-4)/top[...,2:].clamp_min(1e-4)).log()),-1).clamp(-10,10)
        s,e=batch['score_features'].detach(),batch['evidence'].detach()
        x=torch.cat((region-region[:,:1],boundary-boundary[:,:1],coverage-coverage[:,:1],
                     geom,s-s[:,:1],e-e[:,:1]),-1)
        assert x.shape[-1]==198
        gain=((self.compare(x)-self.compare(-x)).squeeze(-1)*.5).tanh()
        gain=torch.cat((torch.zeros_like(gain[:,:1]),gain[:,1:]),-1)
        return dict(decision_score=gain,predicted_gain=gain)


def training_loss(pred,iou_target):
    target=iou_target.detach().clamp(0,1);delta=target-target[:,:1]
    loss=F.mse_loss(pred['predicted_gain'][:,1:],delta[:,1:])
    return loss,dict(gain_mae=(pred['predicted_gain'][:,1:].detach()-delta[:,1:]).abs().mean())
