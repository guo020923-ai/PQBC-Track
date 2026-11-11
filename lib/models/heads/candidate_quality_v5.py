"""Progressive candidate calibration: RQC followed by quality-conditioned ERC.

The inherited RQC-v4 path estimates localization quality and decides when its
quality residual is safe to apply.  The new ERC path receives detached RQC
quality together with visual/language/competition evidence and adds a
zero-initialized identity residual.  Consequently a v4 checkpoint loaded into
this module is exactly v4-equivalent before ERC training.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_quality_v4 import CandidateQualityV4
from .candidate_quality_v4 import training_loss as rqc_training_loss
from .candidate_quality_v2 import candidates


class CandidateQualityV5(CandidateQualityV4):
    """RQC-v4 plus an explicit, quality-conditioned identity ERC."""

    def __init__(self, feature_dim=768, hidden=64, reference=False, scale=1.0,
                 erc_scale=0.35):
        super().__init__(feature_dim=feature_dim, hidden=hidden,
                         reference=reference, scale=scale)
        if not math.isfinite(erc_scale) or erc_scale < 0:
            raise ValueError('RQC-v5 ERC scale must be finite and non-negative')
        self.erc_scale = float(erc_scale)

        # Evidence and Top1-relative evidence: 20
        # RQC quality and Top1-relative quality: 2
        # RQC permission and Top1-relative permission: 2
        # Response prior and Top1-relative response prior: 8
        self.identity = nn.Sequential(
            nn.Linear(32, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        # Exact v4 decision at initialization, independent of the hidden layer.
        nn.init.zeros_(self.identity[-1].weight)
        nn.init.zeros_(self.identity[-1].bias)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('RQC-v5 exposes ERC through its final decision score')

        base = super().forward(batch, use_reliability=False)
        evidence = batch['evidence'].detach()
        evidence_context = torch.cat(
            (evidence, evidence - evidence[:, :1]), dim=-1
        )
        score = batch['score_features'].detach()
        score_context = torch.cat((score, score - score[:, :1]), dim=-1)

        # ERC is conditioned on RQC outputs but cannot redefine their meaning.
        quality = base['quality'].detach()
        permission = base['permission'].detach()
        condition = torch.cat((
            evidence_context,
            quality[..., None], (quality - quality[:, :1])[..., None],
            permission[..., None],
            (permission - permission[:, :1])[..., None],
            score_context,
        ), dim=-1)
        identity_logit = self.identity(condition).squeeze(-1)
        identity = identity_logit.sigmoid()

        # Candidate-specific RQC quality gates an identity residual.  The
        # Top1-relative form preserves ordering exactly when ERC is initialized.
        erc_residual = self.erc_scale * quality * (
            identity - identity[:, :1]
        )
        rqc_decision_score = base['decision_score']
        decision_score = rqc_decision_score + erc_residual

        base.update({
            'rqc_decision_score': rqc_decision_score,
            'identity_logit': identity_logit,
            'identity_reliability': identity,
            'erc_residual': erc_residual,
            'decision_score': decision_score,
        })
        return base


def _hard_selection_gain(scores, target, reference_scores):
    winner = scores.detach().argmax(-1, keepdim=True)
    reference_winner = reference_scores.detach().argmax(-1, keepdim=True)
    selected = target.gather(1, winner).squeeze(1)
    reference = target.gather(1, reference_winner).squeeze(1)
    return selected - reference, winner, reference_winner


def training_loss(pred, iou_target, identity_target, temperature=0.1,
                  identity_weight=0.10, erc_decision_weight=0.50,
                  erc_harm_weight=0.50):
    """Train RQC localization quality and ERC target identity separately."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('RQC-v5 temperature must be positive')
    target = iou_target.detach().clamp(0, 1)
    identity_target = identity_target.detach().clamp(0, 1)

    # Preserve the complete, already validated RQC-v4 objective.
    rqc_pred = dict(pred)
    rqc_pred['decision_score'] = pred['rqc_decision_score']
    rqc_loss, parts = rqc_training_loss(
        rqc_pred, target, temperature=temperature
    )

    # Identity is learned from the official target-center supervision, with
    # greater emphasis on candidates that RQC considers geometrically plausible.
    identity_weights = (0.25 + pred['quality'].detach()) * (
        1.0 + 2.0 * identity_target
    )
    identity_bce = F.binary_cross_entropy_with_logits(
        pred['identity_logit'], identity_target, reduction='none'
    )
    identity_bce = (
        identity_bce * identity_weights
    ).sum() / identity_weights.sum().clamp_min(1e-6)

    identity_advantage = (
        identity_target[:, :, None] - identity_target[:, None, :]
    )
    identity_pairs = identity_advantage.gt(1e-4)
    identity_difference = (
        pred['identity_logit'][:, :, None]
        - pred['identity_logit'][:, None, :]
    )
    if identity_pairs.any():
        identity_rank = (
            F.softplus(-identity_difference)
            * identity_advantage.clamp_min(0)
        )[identity_pairs].mean()
    else:
        identity_rank = identity_difference.sum() * 0.0
    identity_loss = identity_bce + 0.25 * identity_rank

    # Align the deployed hard-ranking path, while measuring ERC increment over
    # the RQC decision rather than falsely attributing Head/RQC gains to ERC.
    final_scores = pred['decision_score']
    regret = target.max(-1, keepdim=True).values - target
    erc_decision = (
        F.softmax(final_scores / temperature, dim=-1) * regret
    ).sum(-1).mean()

    incremental_gain, winner, rqc_winner = _hard_selection_gain(
        final_scores, target, pred['rqc_decision_score']
    )
    selected_margin = (
        final_scores.gather(1, winner).squeeze(1)
        - final_scores.gather(1, rqc_winner).squeeze(1)
    )
    incremental_harm = (-incremental_gain).clamp_min(0)
    erc_hard_harm = (
        incremental_harm * F.softplus(selected_margin / temperature)
    ).mean()

    total = (
        rqc_loss
        + identity_weight * identity_loss
        + erc_decision_weight * erc_decision
        + erc_harm_weight * erc_hard_harm
    )

    baseline_gain, _, _ = _hard_selection_gain(
        final_scores, target,
        torch.cat((torch.zeros_like(final_scores[:, :1]),
                   torch.full_like(final_scores[:, 1:], -1e9)), dim=1),
    )
    parts.update({
        'identity': identity_loss.detach(),
        'identity_bce': identity_bce.detach(),
        'identity_rank': identity_rank.detach(),
        'erc_decision': erc_decision.detach(),
        'erc_hard_harm': erc_hard_harm.detach(),
        'erc_change_rate': winner.ne(rqc_winner).float().mean().detach(),
        'erc_incremental_iou_gain': incremental_gain.mean().detach(),
        'erc_beneficial_change_rate':
            incremental_gain.gt(0.01).float().mean().detach(),
        'erc_harmful_change_rate':
            incremental_gain.lt(-0.01).float().mean().detach(),
        'final_selected_iou_gain': baseline_gain.mean().detach(),
        'identity_reliability_mean':
            pred['identity_reliability'].mean().detach(),
        'erc_residual_abs': pred['erc_residual'].abs().mean().detach(),
    })
    return total, parts
