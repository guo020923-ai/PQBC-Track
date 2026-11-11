"""PC: frozen-tracker, Top1-anchored candidate comparison.

Reuse JC features, NOT its prior-logit score or absolute-quality objective.
No test GT, extra proposals, geometry refinement or new identity gate.
"""
import math
import torch
from torch.nn import functional as F
from .candidate_quality_jc import CandidateQualityJC, candidates, select_candidate


class CandidateQualityPC(CandidateQualityJC):
    def forward(self, batch, use_reliability=False):
        raw = super().forward(batch, use_reliability)['residual']
        # Fixed retain action = 0; bounded relative scores avoid logit explosion.
        advantage = torch.tanh(raw - raw[:, :1])
        return dict(decision_score=advantage, advantage=advantage)


def training_loss(pred, iou_target, temperature=.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('PC requires positive finite comparison temperature')
    score = pred['decision_score']
    target = iou_target.detach().to(score).clamp(0, 1)
    # Ordered pairs include Top1 vs replacements AND replacement vs replacement.
    # Fixed denominator preserves small-gap downweighting (do not divide by gap sum).
    gap = target[:, :, None] - target[:, None, :]
    diff = (score[:, :, None] - score[:, None, :]) / temperature
    k = score.shape[-1]
    loss = (gap.clamp_min(0) * F.softplus(-diff)).sum() / max(len(score)*k*(k-1)/2, 1)
    winner = score.detach().argmax(-1, keepdim=True)
    gain = (target.gather(1, winner) - target[:, :1]).flatten()
    changed = winner.flatten().ne(0)
    return loss, dict(quality=loss.detach()*0, decision=loss.detach(),
        selected_iou_gain=gain.mean(),
        changed_iou_gain=gain.sum()/changed.sum().clamp_min(1),
        harmful_change_rate=gain.lt(-.01).float().mean(),
        beneficial_change_rate=gain.gt(.01).float().mean(),
        severe_harm_rate=gain.lt(-.2).float().mean(),
        positive_gain=gain.clamp_min(0).mean(),
        negative_gain=(-gain).clamp_min(0).mean(),
        score_abs=score.detach().abs().mean(),
        saturation_rate=score.detach().abs().gt(.95).float().mean())
