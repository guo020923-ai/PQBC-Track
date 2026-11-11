"""PD: decision-focused structured regret margin; PC inference unchanged.

This objective addresses all-pairs dilution, not identity or trajectory modeling.
No claim that lower loss guarantees closed-loop AUC gains.
"""
import torch
from .candidate_quality_pc import CandidateQualityPC, candidates, select_candidate

CandidateQualityPD = CandidateQualityPC


def decision_margin_loss(score, target):
    target = target.detach().to(score).clamp(0, 1)
    best = target.max(-1, keepdim=True).values
    regret = best - target
    # Among equal-quality oracle winners, favor the already highest-scoring one.
    # This avoids imposing a spurious order on equally good candidates.
    oracle_score = score.masked_fill(target.ne(best), -torch.inf).max(-1).values
    competitor_score = (score + regret).max(-1).values
    loss = (competitor_score - oracle_score).clamp_min(0).mean()
    return loss, regret


def training_loss(pred, iou_target, temperature=.1):
    # Compatibility argument only: IoU regret directly sets the margin.
    # No temperature-scaled softmax or hand-tuned test threshold.
    score = pred['decision_score']
    target = iou_target.detach().to(score).clamp(0, 1)
    loss, regret = decision_margin_loss(score, target)
    winner = score.detach().argmax(-1, keepdim=True)
    gain = (target.gather(1, winner) - target[:, :1]).flatten()
    changed = winner.flatten().ne(0)
    return loss, dict(quality=loss.detach()*0, decision=loss.detach(),
        selected_iou_gain=gain.mean(),
        changed_iou_gain=gain.sum()/changed.sum().clamp_min(1),
        harmful_change_rate=gain.lt(-.01).float().mean(),
        beneficial_change_rate=gain.gt(.01).float().mean(),
        severe_harm_rate=gain.lt(-.2).float().mean(),
        positive_gain=gain.clamp_min(0).mean(), negative_gain=(-gain).clamp_min(0).mean(),
        score_abs=score.detach().abs().mean(),
        saturation_rate=score.detach().abs().gt(.95).float().mean(),
        selected_regret=regret.gather(1,winner).mean(),
        oracle_gain=(target.max(-1).values-target[:,0]).mean(),
        margin_violation_rate=((score.detach()+regret).max(-1).values >
            score.detach().masked_fill(regret.ne(0),-torch.inf).max(-1).values).float().mean())
