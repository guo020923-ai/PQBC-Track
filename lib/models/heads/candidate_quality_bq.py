"""Boundary/region IoU quality regression; native response only forms TopK/ties."""
import torch
from torch import nn
from torch.nn import functional as F
from .candidate_quality_boundary_pair import candidates
from .candidate_quality_jc import select_candidate

class BoundaryQuality(nn.Module):
    def __init__(self,feature_dim=768,hidden=64,reference=True,scale=1.0):
        super().__init__()
        if feature_dim!=768 or scale!=1.0:raise ValueError('BQ fixes feature size and scale')
        self.project=nn.Linear(768,8,bias=False)
        self.quality=nn.Sequential(nn.Linear(198,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.quality[-1].weight);nn.init.zeros_(self.quality[-1].bias)

    def forward(self,batch,use_reliability=False):
        if use_reliability:raise ValueError('BQ does not enable ERC')
        region=self.project(F.normalize(batch['spatial'].detach(),dim=-1)).flatten(-2)
        valid=batch['boundary_valid'].detach().bool()
        feature=self.project(F.normalize(batch['boundary'].detach(),dim=-1))
        bands=(feature*valid[...,None]).sum(-2)/valid.sum(-1,keepdim=True).clamp_min(1)
        support=valid.any(-1)
        inside,edge,outside=bands.unbind(-2)
        ei=(edge-inside)*(support[...,1]&support[...,0])[...,None]
        oe=(outside-edge)*(support[...,2]&support[...,1])[...,None]
        boundary=torch.cat((edge,ei,oe),-1).flatten(-2)
        coverage=valid.float().mean(-1).flatten(-2)
        x=torch.cat((region,boundary,coverage,batch['boxes'].detach(),
                     batch['score_features'].detach(),batch['evidence'].detach()),-1)
        assert x.shape[-1]==198
        logits=self.quality(x).squeeze(-1)
        return dict(decision_score=logits,quality=logits.sigmoid(),boundary_valid_rate=valid.float().mean())

def training_loss(pred,iou_target,temperature=.1):
    target=iou_target.detach().clamp(0,1)
    loss=F.mse_loss(pred['quality'],target)
    winner=pred['decision_score'].detach().argmax(-1,keepdim=True)
    gain=target.gather(1,winner).flatten()-target[:,0]
    return loss,dict(quality=loss.detach(),decision=loss.detach()*0,
        quality_mae=(pred['quality'].detach()-target).abs().mean(),
        selected_iou_gain=gain.mean(),harmful_change_rate=gain.lt(-.01).float().mean(),
        beneficial_change_rate=gain.gt(.01).float().mean(),severe_harm_rate=gain.lt(-.2).float().mean(),
        boundary_valid_rate=pred['boundary_valid_rate'])
