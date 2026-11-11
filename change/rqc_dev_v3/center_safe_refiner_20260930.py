"""Center-only output refinement with relative geometric preservation supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from relative_local_objective_20260930 import relative_localization_loss

class CenterSafeRefiner(nn.Module):
    def __init__(self,center_limit=.1):
        super().__init__()
        if center_limit<=0:raise ValueError('Positive center limit required')
        self.center_limit=float(center_limit)
        self.net=nn.Sequential(nn.LayerNorm(288),nn.Linear(288,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features,boxes):
        if features.shape[:-1]!=boxes.shape[:-1] or features.shape[-1]!=288 or boxes.shape[-1]!=4:
            raise ValueError('Matching [...,288] and [...,4] required')
        if not torch.isfinite(features).all() or not torch.isfinite(boxes).all() or (boxes[...,2:]<=0).any():
            raise ValueError('Invalid inputs')
        boxes=boxes.detach();delta=self.net(features.detach()).tanh()*self.center_limit
        return dict(boxes=torch.cat((boxes[...,:2]+delta*boxes[...,2:],boxes[...,2:]),-1),delta=delta)

def center_safe_loss(refined,baseline,gt_xywh,image_hw,valid):
    loss,stats=relative_localization_loss(refined,baseline,gt_xywh,image_hw,valid)
    keep=valid.detach().bool()
    if not keep.any():return loss,stats
    r=refined[keep];b=baseline[keep].detach();g=gt_xywh[keep].detach()
    # Dimensionless Euclidean center error in GT-box-diagonal units. log1p
    # prevents a few already-lost targets from dominating this preservation term.
    scale=g[:,2:].square().sum(-1).sqrt().clamp_min(1.)
    target=g[:,:2]+g[:,2:]/2
    def error(box):
        squared=((box[:,:2]+box[:,2:]/2-target)/scale[:,None]).square().sum(-1)
        return torch.log1p(squared)
    penalty=F.relu(error(r)-error(b).detach()).mean()
    stats=dict(stats,center_harm_penalty=penalty.detach())
    return loss+penalty,stats
