"""Target-conditioned joint candidate scoring, independent of legacy RQC.

The final quality probability is supervised directly. No utility, permission,
dense Head loss or test-time GT. All candidate/reference inputs are detached.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .candidate_quality_v2 import candidates as native_candidates


def candidates(out, template_mask, k=10):
    batch = native_candidates(out, template_mask, k)
    batch['template_tokens'] = out['template'].detach()
    batch['template_foreground'] = template_mask.detach().reshape(out['template'].shape[:2]).bool()
    return batch


def select_candidate(batch, prediction):
    return batch['indices'].gather(1, prediction['decision_score'].argmax(-1, keepdim=True))


def joint_scores(prior, residual):
    # Double precision preserves ordering of very close response scores.
    # Clipped ties retain native Top-K order (argmax takes the first entry).
    p = prior.detach().double().clamp(1e-12, 1-1e-12)
    logit = torch.log(p)-torch.log1p(-p)+residual.double()
    return logit, logit.sigmoid()


class CandidateQualityJC(nn.Module):
    def __init__(self, feature_dim=768, hidden=64, reference=False, scale=1.0):
        super().__init__()
        if scale != 1.0:
            raise ValueError('JC fixes residual scale to 1; no utility scale tuning')
        self.token = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 32), nn.GELU())
        self.reference_token = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 32), nn.GELU())
        self.visual_match = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        self.language = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 64))
        self.condition_norm = nn.LayerNorm(32)
        self.region = nn.Sequential(nn.Linear(9*32, hidden), nn.GELU())
        self.score = nn.Sequential(nn.Linear(2*hidden+26, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('JC has no independent ERC')
        spatial = batch['spatial'].detach()
        b,k,n,_ = spatial.shape
        query = self.token(spatial).reshape(b,k*n,32)
        foreground = batch['template_foreground'].bool()
        if not foreground.any(-1).all():
            raise ValueError('Empty target template mask')
        reference = self.reference_token(batch['template_tokens'].detach())
        matched, _ = self.visual_match(query,reference,reference,
                                      key_padding_mask=~foreground,need_weights=False)
        gamma,beta = self.language(batch['language_reference'].detach()).chunk(2,-1)
        conditioned = self.condition_norm((query+matched)*(1+gamma.tanh()[:,None])+beta.tanh()[:,None])
        region = self.region(conditioned.reshape(b,k,9*32))
        box = batch['boxes'].detach()
        top = box[:,:1]
        relative = torch.cat(((box[...,:2]-top[...,:2])/top[...,2:].clamp_min(1e-4),
            (box[...,2:].clamp_min(1e-4)/top[...,2:].clamp_min(1e-4)).log()),-1).clamp(-10,10)
        s = batch['score_features'].detach()
        x = torch.cat((region,region-region[:,:1],box,relative,s,s-s[:,:1],batch['evidence'].detach()),-1)
        residual = self.score(x).squeeze(-1)
        logits,quality = joint_scores(batch['scores'],residual)
        return dict(decision_score=logits,quality_logit=logits,quality=quality,residual=residual)


def training_loss(pred, iou_target, temperature=.1):
    # One final-score continuous quality loss, not extra intermediate objectives.
    target = iou_target.detach().clamp(0,1).to(pred['quality_logit'])
    error = (pred['quality']-target).abs().square()
    loss = (error*F.binary_cross_entropy_with_logits(pred['quality_logit'],target,reduction='none')).mean().float()
    winner = pred['decision_score'].detach().argmax(-1,keepdim=True)
    gain = target.gather(1,winner).squeeze(1)-target[:,0]
    changed = winner.squeeze(1).ne(0)
    return loss, dict(quality=loss.detach(),decision=loss.detach()*0,
        selected_iou_gain=gain.mean().float(),
        changed_iou_gain=(gain.sum()/changed.sum().clamp_min(1)).float(),
        harmful_change_rate=gain.lt(-.01).float().mean(),
        beneficial_change_rate=gain.gt(.01).float().mean(),
        severe_harm_rate=gain.lt(-.2).float().mean(),
        quality_mae=(pred['quality'].detach()-target).abs().mean().float(),
        residual_abs=pred['residual'].detach().abs().mean())
