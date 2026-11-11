import torch
from torch import nn
VERSION='rqc30_pairwise_utility_100ep_v1'
class PairwiseUtilityERC(nn.Module):
 def __init__(self,evidence_dim=48,hidden_dim=64):
  super().__init__();self.encoder=nn.Sequential(nn.LayerNorm(evidence_dim),nn.Linear(evidence_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim),nn.GELU());self.readout=nn.Sequential(nn.LayerNorm(hidden_dim*3),nn.Linear(hidden_dim*3,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1));nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias);self.version=VERSION
 @staticmethod
 def choose(score,executable,anchor,threshold):
  r=torch.arange(len(score),device=score.device);allowed=executable&score.ge(float(threshold));allowed[r,anchor]=False;c=score.masked_fill(~allowed,-torch.inf).argmax(-1);return torch.where(allowed.any(-1),c,anchor)
 def forward(self,evidence,executable,anchor):
  if evidence.shape[1:]!=(25,48) or executable.shape!=(len(evidence),25) or anchor.shape!=(len(evidence),):raise ValueError('schema')
  r=torch.arange(len(evidence),device=evidence.device);z=self.encoder(evidence);a=z[r,anchor];c=a[:,None].expand(-1,25,-1);raw=self.readout(torch.cat((z,c,z-c),-1)).squeeze(-1);score=raw-raw[r,anchor,None];score=score.scatter(1,anchor[:,None],0.)
  if not torch.isfinite(score).all():raise FloatingPointError('score')
  return {'utility_score':score}
