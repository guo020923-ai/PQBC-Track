"""RQC with a learned, reference-aware permission for changing Top1.

The quality predictor remains an IoU estimator. A separate permission predicts
whether each candidate's localization advantage is trustworthy enough to act
on. The score uses the same formula during training and tracking.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_quality_v3 import CandidateQualityV3
from .candidate_quality_v2 import candidates


class CandidateQualityV4(CandidateQualityV3):
    def __init__(self, feature_dim=768, hidden=64, reference=False, scale=1.0):
        super().__init__(feature_dim=feature_dim, hidden=hidden,
                         reference=reference, scale=scale)
        # 2*hidden + box/response (16) + optional quality context (20)
        # + raw target/reference/distractor evidence and its Top1 difference (20).
        self.permission = nn.Sequential(
            nn.Linear(2 * hidden + 56, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.permission[-1].weight)
        nn.init.constant_(self.permission[-1].bias, -1.0)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('RQC-v4 permission is not an independent ERC module')
        region = self.region(self.token(batch['spatial']).flatten(-2))
        box = batch['boxes']
        top = box[:, :1]
        score = batch['score_features']
        relative_box = torch.cat((
            (box[..., :2] - top[..., :2]) / top[..., 2:].clamp_min(1e-4),
            (box[..., 2:].clamp_min(1e-4) / top[..., 2:].clamp_min(1e-4)).log(),
        ), -1).clamp(-10, 10)
        evidence = batch['evidence'].detach()
        reference_context = torch.cat((evidence, evidence - evidence[:, :1]), -1)
        quality_context = reference_context if self.reference else torch.zeros_like(reference_context)
        quality_input = torch.cat((
            region, region - region[:, :1], box, relative_box,
            score, score - score[:, :1], quality_context,
        ), -1)
        quality_logit = self.quality(quality_input).squeeze(-1)
        quality = quality_logit.sigmoid()
        permission_logit = self.permission(
            torch.cat((quality_input, reference_context), -1)
        ).squeeze(-1)
        permission = permission_logit.sigmoid()
        # Baseline Top1 has a zero residual; zero quality initialization gives
        # exact deployed baseline ordering without requiring a test threshold.
        decision_score = score[..., 3].detach() + self.scale * permission * (
            quality - quality[:, :1]
        )
        return {
            'quality_logit': quality_logit,
            'quality': quality,
            'permission_logit': permission_logit,
            'permission': permission,
            'decision_score': decision_score,
        }


def training_loss(pred, iou_target, temperature=0.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('RQC temperature must be positive')
    target = iou_target.detach().clamp(0, 1)
    quality = F.smooth_l1_loss(pred['quality'], target)

    scores = pred['decision_score']
    regret = target.max(-1, keepdim=True).values - target
    decision = (F.softmax(scores / temperature, dim=-1) * regret).sum(-1).mean()

    # The permission learns the *signed* advantage over the original Top1.
    # Wrong switches receive greater weight, especially when the loss is large.
    advantage = target[:, 1:] - target[:, :1]
    weights = advantage.abs() * torch.where(
        advantage < 0, 2.0, 1.0
    )
    permission_bce = F.binary_cross_entropy_with_logits(
        pred['permission_logit'][:, 1:], (advantage > 0).to(target),
        reduction='none',
    )
    permission = (permission_bce * weights).sum() / weights.sum().clamp_min(1e-6)

    # The expectation above is smooth but test time takes a hard argmax.
    # Penalize the actual winning action if it has worse IoU than Top1.
    winner = scores.detach().argmax(-1, keepdim=True)
    selected_iou = target.gather(1, winner).squeeze(1)
    selected_margin = scores.gather(1, winner).squeeze(1) - scores[:, 0]
    harm = (target[:, 0] - selected_iou).clamp_min(0)
    hard_harm = (harm * F.softplus(selected_margin / temperature)).mean()

    total = quality + decision + 0.05 * permission + 0.5 * hard_harm
    gain = selected_iou - target[:, 0]
    return total, {
        'quality': quality.detach(),
        'decision': decision.detach(),
        'permission': permission.detach(),
        'hard_harm': hard_harm.detach(),
        'selected_iou_gain': gain.mean().detach(),
        'harmful_change_rate': gain.lt(-0.01).float().mean().detach(),
        'beneficial_change_rate': gain.gt(0.01).float().mean().detach(),
        'permission_mean': pred['permission'][:, 1:].mean().detach(),
    }
