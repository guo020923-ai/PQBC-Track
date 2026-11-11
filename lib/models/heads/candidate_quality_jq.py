"""JQ prototype: native prior times quality, with explicit Head score gradients.

Indices/box geometry/quality inputs and IoU targets stay detached. Only native
classification/contrastive scores carry ranking gradients into the Head.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .candidate_quality_bq import BoundaryQuality
from .candidate_quality_boundary_pair import candidates as frozen_candidates
from .candidate_quality_jc import select_candidate

def candidates(out, template_mask, k=10):
    batch=frozen_candidates(out,template_mask,k)
    cls=out['cls_score_test'].flatten(1)
    logits=out['cont_score']
    if logits.shape[-1]==2:
        logits=torch.cat((logits,torch.zeros_like(logits[...,:1])),-1)
    if logits.shape[-1]!=3:raise ValueError('JQ requires SOFTMAX_ONE')
    side=int(cls.shape[-1]**.5)
    window=torch.as_tensor(np.outer(np.hanning(side),np.hanning(side)).flatten(),device=cls.device)
    native=cls.double()*logits.softmax(-1)[...,0].double()*window
    batch['live_prior']=native.gather(1,batch['indices'])
    return batch

def decision_scores(prior, quality_logits):
    # Product of positive evidence; log form avoids underflow. Equal initial
    # quality preserves native ordering. Neither term is interpreted as IoU gain.
    p=prior.double().clamp_min(1e-30)
    return p.log()-p[:,:1].log()+F.logsigmoid(quality_logits.double())

class JointQuality(BoundaryQuality):
    def forward(self,batch,use_reliability=False):
        pred=super().forward(batch,use_reliability)
        pred['quality_logit']=pred['decision_score']
        pred['decision_score']=decision_scores(batch['live_prior'],pred['quality_logit'])
        return pred

def training_loss(pred,iou_target,temperature=.1):
    target=iou_target.detach().clamp(0,1)
    quality=F.mse_loss(pred['quality'],target)
    # Every unordered pair is supervised on the deployed final score. Near-ties
    # have proportionally small influence. No threshold or auxiliary Gate.
    scores=pred['decision_score']
    i,j=torch.triu_indices(scores.shape[1],scores.shape[1],offset=1,device=scores.device)
    advantage=target[:,i]-target[:,j]
    pair=F.softplus(-advantage.sign()*(scores[:,i]-scores[:,j]))
    rank=(pair*advantage.abs()).mean().float()
    winner=scores.detach().argmax(-1,keepdim=True)
    gain=target.gather(1,winner).flatten()-target[:,0]
    return quality+rank,dict(quality=quality.detach(),decision=rank.detach(),
        quality_mae=(pred['quality'].detach()-target).abs().mean(),
        selected_iou_gain=gain.mean(),harmful_change_rate=gain.lt(-.01).float().mean(),
        beneficial_change_rate=gain.gt(.01).float().mean(),
        severe_harm_rate=gain.lt(-.2).float().mean(),boundary_valid_rate=pred['boundary_valid_rate'])
