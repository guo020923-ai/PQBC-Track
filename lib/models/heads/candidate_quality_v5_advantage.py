"""RQC-v5: response-prior quality and continuous relative advantage.

RQC-v4 learns candidate IoU and a binary permission for changing the original
Top1.  This version retains both validated components and additionally predicts
the signed IoU advantage of every candidate over the deployed Top1.  The new
residual is zero initialized, so official initialization remains exactly the
original UVLTrack decision.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_quality_v4 import CandidateQualityV4
from .candidate_quality_v4 import training_loss as v4_training_loss
from .candidate_quality_v2 import candidates


class CandidateQualityV5Advantage(CandidateQualityV4):
    def __init__(self, feature_dim=768, hidden=64, reference=False, scale=1.0,
                 advantage_scale=0.50):
        super().__init__(feature_dim=feature_dim, hidden=hidden,
                         reference=reference, scale=scale)
        if not math.isfinite(advantage_scale) or advantage_scale < 0:
            raise ValueError('RQC advantage scale must be finite and non-negative')
        self.advantage_scale = float(advantage_scale)

        # Reuse the full RQC quality context (2*hidden+36), append explicit
        # reference evidence (20), quality/permission and their Top1 deltas (4).
        self.advantage = nn.Sequential(
            nn.Linear(2 * hidden + 60, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.advantage[-1].weight)
        nn.init.zeros_(self.advantage[-1].bias)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('RQC-v5 advantage does not expose an ERC path')

        base = super().forward(batch, use_reliability=False)
        region = self.region(self.token(batch['spatial']).flatten(-2))
        box = batch['boxes']
        top = box[:, :1]
        score = batch['score_features']
        relative_box = torch.cat((
            (box[..., :2] - top[..., :2]) / top[..., 2:].clamp_min(1e-4),
            (box[..., 2:].clamp_min(1e-4)
             / top[..., 2:].clamp_min(1e-4)).log(),
        ), dim=-1).clamp(-10, 10)
        evidence = batch['evidence'].detach()
        evidence_context = torch.cat(
            (evidence, evidence - evidence[:, :1]), dim=-1
        )
        quality_context = (
            evidence_context if self.reference
            else torch.zeros_like(evidence_context)
        )
        quality_input = torch.cat((
            region, region - region[:, :1], box, relative_box,
            score, score - score[:, :1], quality_context,
        ), dim=-1)
        quality = base['quality']
        permission = base['permission']
        advantage_input = torch.cat((
            quality_input,
            evidence_context,
            quality[..., None], (quality - quality[:, :1])[..., None],
            permission[..., None],
            (permission - permission[:, :1])[..., None],
        ), dim=-1)

        advantage_logit = self.advantage(advantage_input).squeeze(-1)
        relative_advantage_logit = advantage_logit - advantage_logit[:, :1]
        predicted_advantage = relative_advantage_logit.tanh()
        advantage_residual = (
            self.advantage_scale * permission * predicted_advantage
        )
        v4_decision_score = base['decision_score']
        decision_score = v4_decision_score + advantage_residual

        base.update({
            'v4_decision_score': v4_decision_score,
            'advantage_logit': advantage_logit,
            'predicted_advantage': predicted_advantage,
            'advantage_residual': advantage_residual,
            'decision_score': decision_score,
        })
        return base


def training_loss(pred, iou_target, temperature=0.1,
                  advantage_weight=0.50, final_decision_weight=0.50,
                  final_harm_weight=1.0):
    """Align continuous advantage learning with the deployed hard decision."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('RQC-v5 temperature must be positive')
    target = iou_target.detach().clamp(0, 1)

    v4_pred = dict(pred)
    v4_pred['decision_score'] = pred['v4_decision_score']
    v4_loss, parts = v4_training_loss(
        v4_pred, target, temperature=temperature
    )

    advantage_target = target - target[:, :1]
    # Negative actions are costlier because one harmful switch can propagate
    # through all following frames of a tracking trajectory.
    advantage_weights = 0.25 + advantage_target.abs() * torch.where(
        advantage_target < 0,
        torch.full_like(advantage_target, 2.0),
        torch.ones_like(advantage_target),
    )
    advantage_regression = F.smooth_l1_loss(
        pred['predicted_advantage'], advantage_target,
        reduction='none', beta=0.1,
    )
    advantage_regression = (
        advantage_regression * advantage_weights
    ).sum() / advantage_weights.sum().clamp_min(1e-6)

    signed = advantage_target[:, 1:]
    direction_weight = signed.abs()
    direction = F.softplus(
        -signed.sign()
        * (pred['advantage_logit'][:, 1:]
           - pred['advantage_logit'][:, :1])
    )
    advantage_direction = (
        direction * direction_weight
    ).sum() / direction_weight.sum().clamp_min(1e-6)
    advantage_loss = advantage_regression + 0.25 * advantage_direction

    final_scores = pred['decision_score']
    regret = target.max(-1, keepdim=True).values - target
    final_decision = (
        F.softmax(final_scores / temperature, dim=-1) * regret
    ).sum(-1).mean()

    winner = final_scores.detach().argmax(-1, keepdim=True)
    selected_iou = target.gather(1, winner).squeeze(1)
    baseline_iou = target[:, 0]
    selected_margin = (
        final_scores.gather(1, winner).squeeze(1) - final_scores[:, 0]
    )
    hard_harm = (
        (baseline_iou - selected_iou).clamp_min(0)
        * F.softplus(selected_margin / temperature)
    ).mean()

    total = (
        v4_loss
        + advantage_weight * advantage_loss
        + final_decision_weight * final_decision
        + final_harm_weight * hard_harm
    )
    gain = selected_iou - baseline_iou
    changed = winner.squeeze(1).ne(0)
    changed_count = changed.sum().clamp_min(1)
    parts.update({
        'advantage': advantage_loss.detach(),
        'advantage_regression': advantage_regression.detach(),
        'advantage_direction': advantage_direction.detach(),
        'advantage_mae': (
            pred['predicted_advantage'].detach() - advantage_target
        ).abs().mean(),
        'advantage_target_abs': advantage_target.abs().mean(),
        'advantage_residual_abs': pred['advantage_residual'].abs().mean().detach(),
        'final_decision': final_decision.detach(),
        'final_hard_harm': hard_harm.detach(),
        'final_selected_iou_gain': gain.mean().detach(),
        'final_changed_iou_gain': (
            gain * changed.to(gain)
        ).sum().detach() / changed_count.to(gain),
        'final_beneficial_change_rate': gain.gt(0.01).float().mean().detach(),
        'final_harmful_change_rate': gain.lt(-0.01).float().mean().detach(),
    })
    return total, parts
