"""Risk-controlled version of listwise RQC-v7.

The listwise quality model supplies the correction direction.  A contextual
benefit/risk estimator decides how much of that correction is safe to deploy.
This is an abstaining quality calibrator, not an ERC identity module: its
targets are candidate-box IoU differences only.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_quality_v2 import candidates
from .candidate_quality_v7_listwise import CandidateQualityV7Listwise


def _weighted_bce(logits, targets, weights):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


class CandidateQualityV7Risk(CandidateQualityV7Listwise):
    def __init__(self, feature_dim=768, hidden=128, k=10, scale=2.0):
        super().__init__(feature_dim=feature_dim, hidden=hidden, k=k, scale=scale)
        self.action = nn.Sequential(
            nn.Linear(2 * hidden + 14, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.action[-1].weight)
        nn.init.constant_(self.action[-1].bias, -1.0)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError("RQC-v7-risk is RQC-only; ERC is disabled")
        head_input, score = self.encode(batch)
        raw_quality_logit = self.quality(head_input).squeeze(-1)
        quality_logit = 2.0 * torch.tanh(raw_quality_logit / 2.0)
        quality = quality_logit.sigmoid()
        action = self.action(head_input)
        benefit_logit = action[..., 0]
        risk_logit = action[..., 1]
        benefit_probability = benefit_logit.sigmoid()
        risk_probability = risk_logit.sigmoid()
        permission = benefit_probability * (1.0 - risk_probability)
        quality_delta = quality_logit - quality_logit[:, :1]
        residual = self.scale * permission * quality_delta
        residual = residual - residual[:, :1]
        decision_score = score[..., 3].detach() + residual
        return {
            "quality_logit": quality_logit,
            "quality": quality,
            "benefit_logit": benefit_logit,
            "risk_logit": risk_logit,
            "permission": permission,
            "decision_residual": residual,
            "decision_score": decision_score,
        }


def training_loss(pred, iou_target, temperature=0.10, margin=0.01):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    target = iou_target.detach().clamp(0, 1)
    logits = pred["quality_logit"]
    delta_top = target - target[:, :1]
    weights = 0.10 + delta_top.abs()

    quality = F.binary_cross_entropy_with_logits(logits, target)
    pair_delta = target[:, :, None] - target[:, None, :]
    logit_delta = logits[:, :, None] - logits[:, None, :]
    pair_mask = pair_delta.abs().gt(0.02)
    pairwise = (
        F.softplus(-pair_delta.sign() * logit_delta) * pair_delta.abs()
    )[pair_mask].mean() if pair_mask.any() else logits.sum() * 0.0
    listwise = F.kl_div(
        F.log_softmax(logits / temperature, dim=-1),
        F.softmax(target / temperature, dim=-1),
        reduction="batchmean",
    )
    benefit = _weighted_bce(
        pred["benefit_logit"], delta_top.gt(margin).to(target), weights
    )
    # Harm errors are more costly in a closed-loop tracker.
    risk_weights = weights * torch.where(
        delta_top.lt(-margin), torch.full_like(weights, 2.0),
        torch.ones_like(weights),
    )
    risk = _weighted_bce(
        pred["risk_logit"], delta_top.lt(-margin).to(target), risk_weights
    )

    scores = pred["decision_score"]
    regret = target.max(-1, keepdim=True).values - target
    oracle_gain = (target.max(-1).values - target[:, 0]).clamp_min(0)
    decision_per_frame = (
        F.softmax(scores / temperature, dim=-1) * regret
    ).sum(-1)
    decision = (
        decision_per_frame * (0.25 + oracle_gain)
    ).sum() / (0.25 + oracle_gain).sum().clamp_min(1e-6)

    harmful = (target[:, :1] - target[:, 1:]).clamp_min(0)
    harmful_margin = (
        harmful * F.softplus(
            (scores[:, 1:] - scores[:, :1]) / temperature
        )
    ).sum() / harmful.sum().clamp_min(1e-6)
    winner = scores.detach().argmax(-1, keepdim=True)
    selected_iou = target.gather(1, winner).squeeze(1)
    selected_margin = scores.gather(1, winner).squeeze(1) - scores[:, 0]
    hard_harm = (
        (target[:, 0] - selected_iou).clamp_min(0)
        * F.softplus(selected_margin / temperature)
    ).mean()
    total = (
        quality + 0.25 * pairwise + 0.10 * listwise
        + 0.20 * benefit + 0.50 * risk
        + decision + harmful_margin + hard_harm
    )
    gain = selected_iou - target[:, 0]
    return total, {
        "quality": quality.detach(),
        "pairwise": pairwise.detach(),
        "listwise": listwise.detach(),
        "benefit": benefit.detach(),
        "risk": risk.detach(),
        "decision": decision.detach(),
        "harmful_margin": harmful_margin.detach(),
        "hard_harm": hard_harm.detach(),
        "selected_iou_gain": gain.mean().detach(),
        "beneficial_change_rate": gain.gt(0.01).float().mean().detach(),
        "harmful_change_rate": gain.lt(-0.01).float().mean().detach(),
        "permission_mean": pred["permission"][:, 1:].mean().detach(),
    }

