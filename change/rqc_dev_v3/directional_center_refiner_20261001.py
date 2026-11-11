"""Spatially ordered, antisymmetric center correction; no identity/ranking gate.

This is an independent proposal, not a replacement for the saved E30 model.
Positive/negative side evidence is compared with a shared function, preventing
an unconditional center translation when opposite-side observations are equal.
"""
import torch
from torch import nn
from torch.nn import functional as F
from head_quality_features_20260930 import HeadQualityCapture

class DirectionalCenterCapture(HeadQualityCapture):
    def gather_directional(self,indices,boxes):
        boxes=boxes.detach()
        if boxes.shape!=(*indices.shape,4):raise ValueError('Matching [B,K,4] boxes required')
        context=self.gather(indices).detach()
        # Crop-normalized cxcywh -> symmetric object-relative side locations.
        # Ordering is (+x,-x,+y,-y). grid_sample align_corners=False corresponds
        # to pixel centers (j+.5)/W, just as normalized tracking crop coordinates.
        offsets=boxes.new_tensor([[.25,0],[-.25,0],[0,.25],[0,-.25]])
        points=boxes[...,:2].unsqueeze(-2)
        points=points+offsets*boxes[...,2:].unsqueeze(-2)
        grid=2*points-1;parts=[]
        for name in ('conv_bbox','conv_offset','conv_cls'):
            value=self.maps[name].detach()
            sampled=F.grid_sample(value,grid,mode='bilinear',padding_mode='border',align_corners=False)
            parts.append(sampled.permute(0,2,3,1))
        sides=torch.cat(parts,-1).reshape(*indices.shape,2,2,96)
        return dict(context=context,pairs=sides.detach())

class DirectionalCenterRefiner(nn.Module):
    def __init__(self,center_limit=.1):
        super().__init__()
        if center_limit<=0:raise ValueError('Positive center limit required')
        self.center_limit=float(center_limit)
        self.net=nn.Sequential(nn.LayerNorm(480),nn.Linear(480,64),nn.GELU(),nn.Linear(64,1,bias=False))
        nn.init.zeros_(self.net[-1].weight)
    def forward(self,features,boxes):
        context=features['context'].detach();pairs=features['pairs'].detach();boxes=boxes.detach()
        shape=boxes.shape[:-1]
        if context.shape!=(*shape,288) or pairs.shape!=(*shape,2,2,96):raise ValueError('Wrong directional features')
        if not all(torch.isfinite(x).all() for x in (boxes,context,pairs)) or (boxes[...,2:]<=0).any():
            raise ValueError('Invalid inputs')
        c=context.unsqueeze(-2).expand(*shape,2,288)
        pos,neg=pairs[...,0,:],pairs[...,1,:]
        plus=self.net(torch.cat((c,pos,neg),-1)).squeeze(-1)
        minus=self.net(torch.cat((c,neg,pos),-1)).squeeze(-1)
        delta=((plus-minus)*.5).tanh()*self.center_limit
        return dict(boxes=torch.cat((boxes[...,:2]+delta*boxes[...,2:],boxes[...,2:]),-1),delta=delta)

def subset_features(features,index):return {k:v[index] for k,v in features.items()}
