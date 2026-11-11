import torch
from torch.nn import functional as F
from .config import PROTOCOL
def pairwise_utility_loss(prediction,executable,anchor,utility):
 score=prediction['utility_score'];b,a=score.shape;r=torch.arange(b,device=score.device);valid=executable.clone();valid[r,anchor]=True;delta_u=utility[:,:,None]-utility[:,None,:];delta_s=score[:,:,None]-score[:,None,:];pair=valid[:,:,None]&valid[:,None,:];eye=torch.eye(a,dtype=torch.bool,device=score.device)[None];pair=pair&~eye;target=torch.sigmoid(delta_u/PROTOCOL['pair_temperature']);weight=(delta_u.abs()/PROTOCOL['pair_temperature']).clamp(.25,4.);element=F.binary_cross_entropy_with_logits(delta_s,target,reduction='none')*weight;loss=element[pair].mean()
 if not torch.isfinite(loss):raise FloatingPointError('loss')
 with torch.no_grad():
  alternatives=valid.clone();alternatives[r,anchor]=False;c=score.masked_fill(~alternatives,-torch.inf).argmax(-1);active=alternatives.any(-1)&score[r,c].gt(0);chosen=torch.where(active,c,anchor);cu=utility[r,chosen];oracle=utility.masked_fill(~alternatives,-torch.inf).argmax(-1);ou=utility[r,oracle].clamp_min(0);count=active.double().sum().clamp_min(1)
  stats={'loss':loss.detach(),'predicted_keep_rate':chosen.eq(anchor).double().mean(),'harmful_deploy_rate':((cu<=0)&active).double().sum()/count,'mean_deployed_utility':cu.masked_select(active).double().sum()/count,'chosen_regret':(ou-cu).clamp_min(0).double().mean(),'pair_accuracy':((delta_s.sign()==delta_u.sign())&pair).double().sum()/pair.double().sum().clamp_min(1),'mean_score':score.masked_select(alternatives).double().mean()}
 return loss,stats
