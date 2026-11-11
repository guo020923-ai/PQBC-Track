"""Class-balanced ordinal action-risk objective."""
import torch
from torch.nn import functional as F
from .config import PROTOCOL
def ordinal_target(utility):
 target=torch.ones_like(utility,dtype=torch.long);target=target.masked_fill(utility.le(PROTOCOL['harmful_upper_utility']),0);target=target.masked_fill(utility.ge(PROTOCOL['actionable_lower_utility']),2);return target
def _mean(v,m):
 x=v.masked_select(m);return x.mean() if x.numel() else v.sum()*0
def ordinal_risk_loss(prediction,executable,anchor,utility):
 if utility.shape!=(len(utility),25):raise ValueError('utility shape')
 r=torch.arange(len(utility),device=utility.device);valid=executable.clone();valid[r,anchor]=False;target=ordinal_target(utility);element=F.cross_entropy(prediction['ordinal_logits'].flatten(0,1),target.flatten(),reduction='none').reshape_as(utility);parts=[_mean(element,valid&target.eq(k)) for k in range(3) if (valid&target.eq(k)).any()];loss=torch.stack(parts).mean() if parts else element.sum()*0
 if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
 with torch.no_grad():
  pred=prediction['ordinal_probability'].argmax(-1);score=prediction['risk_score'];chosen=prediction['risk_score'].masked_fill(~valid,-torch.inf).argmax(-1);active=valid.any(-1)&score[r,chosen].gt(0);selected=torch.where(active,chosen,anchor);su=utility[r,selected]
  stats={'loss':loss.detach(),'harmful_accuracy':_mean(pred.eq(0).float(),valid&target.eq(0)),'marginal_accuracy':_mean(pred.eq(1).float(),valid&target.eq(1)),'actionable_accuracy':_mean(pred.eq(2).float(),valid&target.eq(2)),'predicted_keep_rate':selected.eq(anchor).double().mean(),'harmful_deploy_rate':((su<=0)&active).double().sum()/active.double().sum().clamp_min(1),'mean_deployed_utility':su.masked_select(active).double().sum()/active.double().sum().clamp_min(1),'mean_risk_score':score.masked_select(valid).double().mean()}
 return loss,stats
