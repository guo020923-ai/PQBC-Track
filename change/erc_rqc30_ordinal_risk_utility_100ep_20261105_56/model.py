"""One ordinal head jointly selects actions and controls deployment risk."""
import torch
from torch import nn
VERSION='rqc30_unified_ordinal_risk_utility_100ep_v1'
class OrdinalRiskUtilityERC(nn.Module):
 def __init__(self,evidence_dim=48,hidden_dim=64):
  super().__init__()
  if evidence_dim!=48 or hidden_dim!=64:raise ValueError('fixed dimensions required')
  self.encoder=nn.Sequential(nn.LayerNorm(evidence_dim),nn.Linear(evidence_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim),nn.GELU())
  self.ordinal_readout=nn.Sequential(nn.LayerNorm(hidden_dim*3),nn.Linear(hidden_dim*3,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,3))
  nn.init.zeros_(self.ordinal_readout[-1].weight);nn.init.zeros_(self.ordinal_readout[-1].bias);self.version=VERSION
 @staticmethod
 def _validate(e,x,a):
  if e.ndim!=3 or e.shape[1:]!=(25,48):raise ValueError('evidence shape')
  if x.shape!=(len(e),25) or x.dtype!=torch.bool or a.shape!=(len(e),) or a.dtype!=torch.long:raise ValueError('mask/anchor shape')
  r=torch.arange(len(e),device=e.device)
  if not x[r,a].all() or not torch.isfinite(e).all():raise ValueError('invalid evidence')
 @staticmethod
 def choose(score,executable,anchor,threshold):
  if not -1<=float(threshold)<=1:raise ValueError('threshold range')
  r=torch.arange(len(score),device=score.device);allowed=executable&score.ge(float(threshold));allowed[r,anchor]=False;c=score.masked_fill(~allowed,-torch.inf).argmax(-1);return torch.where(allowed.any(-1),c,anchor)
 def forward(self,evidence,executable,anchor_index):
  self._validate(evidence,executable,anchor_index);r=torch.arange(len(evidence),device=evidence.device);z=self.encoder(evidence);a=z[r,anchor_index];c=a[:,None].expand(-1,25,-1);h=torch.cat((z,c,z-c),-1);logits=self.ordinal_readout(h);prob=logits.softmax(-1);score=prob[...,2]-prob[...,0];score=score.scatter(1,anchor_index[:,None],0.0)
  if not torch.isfinite(logits).all():raise FloatingPointError('non-finite output')
  return {'ordinal_logits':logits,'ordinal_probability':prob,'risk_score':score}
