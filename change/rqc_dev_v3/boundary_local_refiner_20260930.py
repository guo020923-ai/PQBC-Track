"""Independent box-aligned input variant; no tracker/actor registration.

Keep the original refiner's action space to isolate input representation changes.
Four sides x three inside/edge/outside bands x three tangent samples preserve
spatial order. Inputs are detached; sampling uses token-center coordinates.
"""
import torch
from torch import nn
from torch.nn import functional as F
from local_box_refiner_20260930 import LocalBoxRefiner


def sample_box_boundary(maps, boxes):
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('Expected [B,4] normalized cxcywh')
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError('Invalid boxes')
    ordered = [maps[k].detach() for k in ('conv_bbox','conv_offset','conv_cls')]
    shape = ordered[0].shape
    if len(shape) != 4 or shape[1] != 32 or shape[0] != boxes.shape[0] or any(v.shape != shape for v in ordered):
        raise ValueError('Expected three matching [B,32,H,W] maps')
    feature = torch.cat(ordered, 1)
    if not torch.isfinite(feature).all(): raise ValueError('Nonfinite maps')
    offsets = []
    for edge in range(4):
        for radius in (.4,.5,.6):
            for tangent in (-1/3,0.,1/3):
                offsets.append((-radius,tangent) if edge == 0 else
                               (radius,tangent) if edge == 1 else
                               (tangent,-radius) if edge == 2 else (tangent,radius))
    box = boxes.detach()
    grid = box[:,None,:2] + box[:,None,2:] * box.new_tensor(offsets)[None]
    h,w = shape[-2:]
    lo = box.new_tensor([.5/w,.5/h]); hi = 1-lo
    valid = ((grid >= lo) & (grid <= hi)).all(-1)
    samples = F.grid_sample(feature,grid[:,:,None,:]*2-1,mode='bilinear',
                            padding_mode='zeros',align_corners=False)
    samples = samples[:,:,:,0].transpose(1,2) * valid[...,None]
    return dict(samples=samples.reshape(-1,4,3,3,96),valid=valid.reshape(-1,4,3,3),
                coordinates=grid.reshape(-1,4,3,3,2),boxes=box)


class BoundaryLocalRefiner(LocalBoxRefiner):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(96,8,bias=False)
        self.net = nn.Sequential(nn.LayerNorm(328),nn.Linear(328,56),nn.GELU(),nn.Linear(56,4))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self, batch):
        samples=batch['samples'].detach(); valid=batch['valid'].detach().bool(); boxes=batch['boxes'].detach()
        if samples.shape != (len(boxes),4,3,3,96) or valid.shape != (len(boxes),4,3,3):
            raise ValueError('Invalid boundary feature shape')
        projected=self.project(samples)*valid[...,None]
        features=torch.cat((projected.flatten(1),valid.float().flatten(1),boxes),-1)
        raw=self.net(features).tanh()
        delta=torch.cat((raw[:,:2]*self.center_limit,raw[:,2:]*self.log_size_limit),-1)
        refined=torch.cat((boxes[:,:2]+delta[:,:2]*boxes[:,2:],boxes[:,2:]*delta[:,2:].exp()),-1)
        return dict(boxes=refined,delta=delta,boundary_valid_rate=valid.float().mean())
