"""RQC-R1: replace binary permission with downside-aware relative utility.

Not stacked on v5. C/T, proposals and geometry are unchanged. The original
quality estimator is retained as an auxiliary task; only the relative utility
residual drives reordering. No dataset identifiers or test GT are inputs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .candidate_quality_v3 import CandidateQualityV3
from .candidate_quality_v2 import candidates


def decision_scores(prior, utility, scale=1.0):
    return prior.detach() + scale * utility


def select_candidate(batch, prediction):
    return batch['indices'].gather(1, prediction['decision_score'].argmax(-1, keepdim=True))


def risk_regression(predicted, target):
    # Overestimating action value (including underestimating damage) is costlier.
    error = predicted - target.detach()
    weight = torch.where(error.detach() > 0, 2., 1.)
    return (weight * F.smooth_l1_loss(predicted, target.detach(), reduction='none', beta=.1)).mean()


class CandidateQualityR1(CandidateQualityV3):
    def __init__(self, feature_dim=768, hidden=64, reference=False, scale=1.0):
        super().__init__(feature_dim, hidden, reference, scale)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Invalid RQC scale')
        self.utility = nn.Sequential(nn.Linear(2*hidden+56, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.utility[-1].weight)
        nn.init.zeros_(self.utility[-1].bias)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('R1 is RQC only, not ERC')
        region = self.region(self.token(batch['spatial']).flatten(-2))
        box, score = batch['boxes'], batch['score_features']
        top = box[:, :1]
        relative = torch.cat(((box[..., :2]-top[..., :2])/top[..., 2:].clamp_min(1e-4),
                              (box[..., 2:].clamp_min(1e-4)/top[..., 2:].clamp_min(1e-4)).log()), -1).clamp(-10, 10)
        evidence = batch['evidence'].detach()
        context = torch.cat((evidence, evidence-evidence[:, :1]), -1)
        inputs = torch.cat((region, region-region[:, :1], box, relative, score, score-score[:, :1],
                            context if self.reference else torch.zeros_like(context)), -1)
        qlogit = self.quality(inputs).squeeze(-1)
        raw = self.utility(torch.cat((inputs, context), -1)).squeeze(-1).tanh()
        # The reference action has exactly zero utility; other actions are
        # directly regressed against their Top1-relative labels, not q_i-q_0.
        utility = torch.cat((torch.zeros_like(raw[:, :1]), raw[:, 1:]), -1)
        return dict(quality_logit=qlogit, quality=qlogit.sigmoid(), utility=utility,
                    decision_score=decision_scores(score[..., 3], utility, self.scale))


def training_loss(pred, iou_target, temperature=.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid temperature')
    target = iou_target.detach().clamp(0, 1)
    advantage = target-target[:, :1]
    quality = F.smooth_l1_loss(pred['quality'], target)
    utility_loss = risk_regression(pred['utility'][:, 1:], advantage[:, 1:])
    scores = pred['decision_score']
    regret = target.max(-1, keepdim=True).values-target
    decision = (scores.div(temperature).softmax(-1)*regret).sum(-1).mean()
    winner = scores.detach().argmax(-1, keepdim=True)
    gain = target.gather(1, winner).squeeze(1)-target[:, 0]
    margin = scores.gather(1, winner).squeeze(1)-scores[:, 0]
    hard_harm = ((-gain).clamp_min(0)*F.softplus(margin/temperature)).mean()
    changed = winner.squeeze(1).ne(0)
    total = quality+utility_loss+decision+.5*hard_harm
    return total, dict(quality=quality.detach(), decision=decision.detach(),
        utility=utility_loss.detach(), hard_harm=hard_harm.detach(),
        selected_iou_gain=gain.mean(),
        changed_iou_gain=(gain*changed).sum()/changed.sum().clamp_min(1),
        harmful_change_rate=gain.lt(-.01).float().mean(),
        beneficial_change_rate=gain.gt(.01).float().mean(),
        severe_harm_rate=gain.lt(-.2).float().mean(),
        utility_mae=(pred['utility'].detach()-advantage).abs().mean(),
        utility_abs=pred['utility'].detach().abs().mean())
