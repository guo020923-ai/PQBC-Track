"""RQC-v6: benefit-risk decomposed candidate action value.

The original response remains the decision prior.  RQC predicts localization
quality, but candidate replacement is controlled by two different questions:
how likely/how large is a benefit, and how likely/how large is a harm.  This
avoids treating a slightly positive mean advantage as a safe action.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_quality_v3 import CandidateQualityV3
from .candidate_quality_v2 import candidates


def _focal_bce(logits, targets, positive_alpha=0.75, gamma=2.0,
               sample_weight=None):
    probability = logits.sigmoid()
    probability_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = positive_alpha * targets + (1 - positive_alpha) * (1 - targets)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    ) * alpha_t * (1 - probability_t).pow(gamma)
    if sample_weight is not None:
        loss = loss * sample_weight
        return loss.sum() / sample_weight.sum().clamp_min(1e-6)
    return loss.mean()


class CandidateQualityV6RiskValue(CandidateQualityV3):
    def __init__(self, feature_dim=768, hidden=64, reference=False, scale=1.0,
                 risk_weight=1.00):
        super().__init__(feature_dim=feature_dim, hidden=hidden,
                         reference=reference, scale=scale)
        if not math.isfinite(risk_weight) or risk_weight < 1:
            raise ValueError('RQC risk weight must be finite and at least one')
        self.risk_weight = float(risk_weight)

        # Base quality input (2*hidden+36), explicit multimodal evidence (20),
        # quality and Top1-relative quality (2).
        action_dim = 2 * hidden + 58
        self.benefit = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )
        self.risk = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )
        for head in (self.benefit, self.risk):
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, -1.5)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('RQC-v6 has no ERC inference path')
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
        quality_logit = self.quality(quality_input).squeeze(-1)
        quality = quality_logit.sigmoid()
        action_input = torch.cat((
            quality_input, evidence_context,
            quality[..., None], (quality - quality[:, :1])[..., None],
        ), dim=-1)

        benefit_output = self.benefit(action_input)
        risk_output = self.risk(action_input)
        benefit_logit = benefit_output[..., 0]
        risk_logit = risk_output[..., 0]
        benefit_probability = benefit_logit.sigmoid()
        risk_probability = risk_logit.sigmoid()
        benefit_magnitude = benefit_output[..., 1].sigmoid()
        risk_magnitude = risk_output[..., 1].sigmoid()

        net_action_value = (
            benefit_probability * benefit_magnitude
            - self.risk_weight * risk_probability * risk_magnitude
        )
        # Top1 is the explicit abstention/default action.  A candidate may only
        # receive a positive residual when estimated benefit exceeds estimated
        # risk; negative values never suppress Top1 and cannot turn into a
        # positive residual through subtraction.
        candidate_mask = torch.ones_like(net_action_value)
        candidate_mask[:, 0] = 0
        action_value = F.relu(net_action_value) * candidate_mask
        action_residual = self.scale * action_value
        decision_score = score[..., 3].detach() + action_residual
        return {
            'quality_logit': quality_logit,
            'quality': quality,
            'benefit_logit': benefit_logit,
            'benefit_probability': benefit_probability,
            'benefit_magnitude': benefit_magnitude,
            'risk_logit': risk_logit,
            'risk_probability': risk_probability,
            'risk_magnitude': risk_magnitude,
            'action_value': action_value,
            'action_residual': action_residual,
            'decision_score': decision_score,
        }


def training_loss(pred, iou_target, temperature=0.1, action_margin=0.01):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('RQC temperature must be positive')
    if not 0 <= action_margin < 1:
        raise ValueError('RQC action margin must be in [0, 1)')
    target = iou_target.detach().clamp(0, 1)
    delta = target - target[:, :1]
    magnitude = delta.abs()
    sample_weight = 0.1 + magnitude

    quality = F.smooth_l1_loss(pred['quality'], target)
    benefit_target = delta.gt(action_margin).to(target)
    risk_target = delta.lt(-action_margin).to(target)
    benefit_classification = _focal_bce(
        pred['benefit_logit'], benefit_target,
        positive_alpha=0.80, sample_weight=sample_weight,
    )
    risk_classification = _focal_bce(
        pred['risk_logit'], risk_target,
        positive_alpha=0.80, sample_weight=sample_weight,
    )

    benefit_mask = benefit_target.bool()
    risk_mask = risk_target.bool()
    zero = target.sum() * 0.0
    benefit_magnitude = (
        F.smooth_l1_loss(
            pred['benefit_magnitude'][benefit_mask],
            delta.clamp_min(0)[benefit_mask], beta=0.1,
        ) if benefit_mask.any() else zero
    )
    risk_magnitude = (
        F.smooth_l1_loss(
            pred['risk_magnitude'][risk_mask],
            (-delta).clamp_min(0)[risk_mask], beta=0.1,
        ) if risk_mask.any() else zero
    )
    magnitude_loss = benefit_magnitude + risk_magnitude

    # Only positive improvement is actionable.  Harm is modeled separately;
    # forcing a signed mean into the decision was the v5 failure mode.
    positive_delta = delta.clamp_min(0)
    value_weight = 0.25 + magnitude * torch.where(
        delta < 0, torch.full_like(delta, 2.0), torch.ones_like(delta)
    )
    value_regression = (
        F.smooth_l1_loss(
            pred['action_value'], positive_delta, reduction='none', beta=0.1
        ) * value_weight
    ).sum() / value_weight.sum().clamp_min(1e-6)
    harmful_overestimate = (
        pred['action_value'] * (-delta).clamp_min(0)
    ).sum() / (-delta).clamp_min(0).sum().clamp_min(1e-6)

    scores = pred['decision_score']
    regret = target.max(-1, keepdim=True).values - target
    decision = (
        F.softmax(scores / temperature, dim=-1) * regret
    ).sum(-1).mean()

    harmful = (-delta[:, 1:]).clamp_min(0)
    harmful_margin = (
        harmful
        * F.softplus((scores[:, 1:] - scores[:, :1]) / temperature)
    ).sum() / harmful.sum().clamp_min(1e-6)

    winner = scores.detach().argmax(-1, keepdim=True)
    selected_iou = target.gather(1, winner).squeeze(1)
    selected_margin = scores.gather(1, winner).squeeze(1) - scores[:, 0]
    hard_harm = (
        (target[:, 0] - selected_iou).clamp_min(0)
        * F.softplus(selected_margin / temperature)
    ).mean()

    total = (
        quality
        + 0.25 * benefit_classification
        + 0.75 * risk_classification
        + 0.25 * magnitude_loss
        + 0.50 * value_regression
        + harmful_overestimate
        + decision
        + harmful_margin
        + hard_harm
    )
    gain = selected_iou - target[:, 0]
    changed = winner.squeeze(1).ne(0)
    changed_count = changed.sum().clamp_min(1).to(gain)
    return total, {
        'quality': quality.detach(),
        'decision': decision.detach(),
        'benefit_classification': benefit_classification.detach(),
        'risk_classification': risk_classification.detach(),
        'magnitude': magnitude_loss.detach(),
        'value_regression': value_regression.detach(),
        'harmful_overestimate': harmful_overestimate.detach(),
        'harmful_margin': harmful_margin.detach(),
        'hard_harm': hard_harm.detach(),
        'selected_iou_gain': gain.mean().detach(),
        'changed_iou_gain': (
            gain * changed.to(gain)
        ).sum().detach() / changed_count,
        'harmful_change_rate': gain.lt(-0.01).float().mean().detach(),
        'beneficial_change_rate': gain.gt(0.01).float().mean().detach(),
        'benefit_probability_mean':
            pred['benefit_probability'][:, 1:].mean().detach(),
        'risk_probability_mean':
            pred['risk_probability'][:, 1:].mean().detach(),
        'action_value_abs': pred['action_value'].abs().mean().detach(),
    }
