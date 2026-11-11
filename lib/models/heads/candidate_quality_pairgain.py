"""Pairwise localization gain RQC.

Antisymmetric comparator, single IoU-difference regression, no log-prior sum.
Native response defines candidate membership/order only (Top1 wins zero ties).
"""
import torch
from torch import nn
from torch.nn import functional as F


from .candidate_quality_jc import candidates, select_candidate


class PairGain(nn.Module):
    def __init__(self, feature_dim=768, hidden=64, reference=True, scale=1.0):
        super().__init__()
        if feature_dim != 768 or scale != 1.0:
            raise ValueError("PairGain requires 768 features and unit scale")
        # Shared compression before taking differences; no absolute identity input.
        self.project = nn.Linear(768, 8, bias=False)
        # 9*8 spatial delta, four box relations, four score and ten evidence deltas.
        self.compare = nn.Sequential(nn.Linear(90, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.compare[-1].weight)
        nn.init.zeros_(self.compare[-1].bias)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('PairGain has no ERC dependency')
        spatial = F.normalize(batch['spatial'].detach(), dim=-1)
        region = self.project(spatial).flatten(-2)
        box = batch['boxes'].detach()
        top = box[:, :1]
        # Symmetric denominator makes these relations change sign under swap.
        geometry = torch.cat(((box[..., :2]-top[..., :2]) /
                              ((box[..., 2:]+top[..., 2:])*.5).clamp_min(1e-4),
                              (box[..., 2:].clamp_min(1e-4)/top[..., 2:].clamp_min(1e-4)).log()), -1).clamp(-10, 10)
        scores = batch['score_features'].detach()
        evidence = batch['evidence'].detach()
        x = torch.cat((region-region[:, :1], geometry, scores-scores[:, :1],
                       evidence-evidence[:, :1]), -1)
        # Structural anti-symmetry: exchanging a pair negates its prediction.
        raw = (self.compare(x)-self.compare(-x)).squeeze(-1)*.5
        gain = raw.tanh()
        gain = torch.cat((torch.zeros_like(gain[:, :1]), gain[:, 1:]), -1)
        return dict(decision_score=gain, predicted_gain=gain)


def training_loss(pred, iou_target, temperature=.1):
    target = iou_target.detach().clamp(0, 1)
    delta = target-target[:, :1]
    # Uniform squared loss estimates conditional mean gain, unlike reweighted
    # regression whose numeric expectation is shifted. No positive oversampling.
    loss = F.mse_loss(pred['predicted_gain'][:, 1:], delta[:, 1:])
    winner = pred['decision_score'].detach().argmax(-1, keepdim=True)
    realized = target.gather(1, winner).flatten()-target[:, 0]
    return loss, dict(quality=loss.detach(), decision=loss.detach()*0,
                      harmful_change_rate=realized.lt(-.01).float().mean(),
                      beneficial_change_rate=realized.gt(.01).float().mean(),
                      severe_harm_rate=realized.lt(-.2).float().mean(),
                      predicted_gain_abs=pred['predicted_gain'].detach().abs().mean(),
                      gain_mae=(pred['predicted_gain'][:, 1:].detach()-delta[:, 1:]).abs().mean(),
                      selected_iou_gain=realized.mean())
