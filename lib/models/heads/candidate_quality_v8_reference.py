"""Multimodal target-conditioned listwise candidate quality predictor.

This follows the useful principle behind target-conditioned IoU prediction,
but adapts it to UVLTrack: each existing Top-K candidate is conditioned by the
initial visual target, language description, and target/distractor/background
prototypes before the candidate set is compared jointly.  It predicts box IoU
only; candidate identity reliability remains a separate future ERC task.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_quality_v2 import candidates


class CandidateQualityV8Reference(nn.Module):
    def __init__(self, feature_dim=768, hidden=128, k=10, scale=1.0):
        super().__init__()
        self.k = int(k)
        self.scale = float(scale)
        self.spatial_token = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 16), nn.GELU()
        )
        self.spatial_region = nn.Sequential(
            nn.Linear(9 * 16, hidden // 2), nn.GELU()
        )
        self.candidate = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.GELU()
        )
        self.reference = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.GELU()
        )
        # candidate, multiplicative and difference interactions with visual and
        # language references, plus target/distractor/background interactions.
        self.reference_fuse = nn.Sequential(
            nn.Linear(9 * hidden, 2 * hidden), nn.GELU(),
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        self.scalar = nn.Sequential(
            nn.LayerNorm(36), nn.Linear(36, hidden // 2), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.rank_embedding = nn.Parameter(torch.zeros(1, k, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=3 * hidden,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=2)
        self.quality = nn.Sequential(
            nn.Linear(2 * hidden + 14, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    @staticmethod
    def _relative_box(box):
        top = box[:, :1]
        return torch.cat((
            (box[..., :2] - top[..., :2]) / top[..., 2:].clamp_min(1e-4),
            (box[..., 2:].clamp_min(1e-4)
             / top[..., 2:].clamp_min(1e-4)).log(),
        ), -1).clamp(-10, 10)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError("RQC-v8 is quality-only; ERC is disabled")
        required = ("visual_reference", "language_reference", "prompt_reference")
        if any(key not in batch for key in required):
            raise KeyError("RQC-v8 requires raw multimodal reference embeddings")
        if batch["boxes"].shape[1] != self.k:
            raise ValueError("candidate count differs from configured Top-K")

        candidate = self.candidate(batch["pooled"])
        visual = self.reference(batch["visual_reference"])[:, None]
        language = self.reference(batch["language_reference"])[:, None]
        prompts = self.reference(batch["prompt_reference"])
        target, distractor, background = prompts.split(1, dim=1)
        reference_conditioned = self.reference_fuse(torch.cat((
            candidate,
            candidate * visual, (candidate - visual).abs(),
            candidate * language, (candidate - language).abs(),
            candidate * target, candidate * distractor, candidate * background,
            candidate * (target - distractor),
        ), -1))
        spatial = self.spatial_region(
            self.spatial_token(batch["spatial"]).flatten(-2)
        )
        box = batch["boxes"]
        score = batch["score_features"]
        evidence = batch["evidence"].detach()
        scalar = torch.cat((
            box, self._relative_box(box), score, score - score[:, :1],
            evidence, evidence - evidence[:, :1],
        ), -1)
        token = self.fuse(torch.cat((
            reference_conditioned,
            spatial,
            self.scalar(scalar),
        ), -1))
        contextual = self.context(token + self.rank_embedding)
        head_input = torch.cat((
            contextual, contextual - contextual[:, :1],
            score - score[:, :1], evidence - evidence[:, :1],
        ), -1)
        raw = self.quality(head_input).squeeze(-1)
        # Learn a correction to the exact deployed response score.  Unlike the
        # earlier two-score formulation, this joint score is itself supervised
        # by localization quality, so training and test-time ranking coincide.
        residual = self.scale * 2.0 * torch.tanh(raw / 2.0)
        decision_score = score[..., 3].detach() + residual
        quality = decision_score.sigmoid()
        return {
            "quality_logit": decision_score,
            "quality": quality,
            "decision_residual": residual,
            "decision_score": decision_score,
        }


def training_loss(pred, iou_target, temperature=0.10):
    """IoU-aware joint-score learning on the deployed ranking score.

    This is a Top-K tracking adaptation of the quality-score alignment idea in
    GFL/VarifocalNet and the localization-score alignment used by SiamRCR.  It
    does not refine boxes and does not use a separate test-time threshold.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    target = iou_target.detach().clamp(0, 1)
    scores = pred["decision_score"]
    probability = scores.sigmoid()
    error = (probability - target).abs()
    # High-quality candidates supervise the joint score strongly; near-zero
    # candidates are retained as hard negatives in a focal manner.
    weight = torch.where(
        target.gt(0.05), target.clamp_min(0.10), 0.75 * error.square()
    )
    quality = (
        F.binary_cross_entropy_with_logits(scores, target, reduction="none")
        * weight
    ).sum() / weight.sum().clamp_min(1e-6)

    pair_delta = target[:, :, None] - target[:, None, :]
    score_delta = scores[:, :, None] - scores[:, None, :]
    pair_mask = pair_delta.abs().gt(0.02)
    pairwise = (
        F.softplus(-pair_delta.sign() * score_delta) * pair_delta.abs()
    )[pair_mask].mean() if pair_mask.any() else scores.sum() * 0.0
    listwise = F.kl_div(
        F.log_softmax(scores / temperature, dim=-1),
        F.softmax(target / temperature, dim=-1),
        reduction="batchmean",
    )

    regret = target.max(-1, keepdim=True).values - target
    oracle_gain = (target.max(-1).values - target[:, 0]).clamp_min(0)
    soft_regret = (F.softmax(scores / temperature, -1) * regret).sum(-1)
    decision = (
        soft_regret * (0.25 + oracle_gain)
    ).sum() / (0.25 + oracle_gain).sum().clamp_min(1e-6)

    harmful = (target[:, :1] - target[:, 1:]).clamp_min(0)
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
        quality + 0.25 * pairwise + 0.10 * listwise
        + decision + 0.75 * harmful_margin + hard_harm
    )
    gain = selected_iou - target[:, 0]
    changed = winner.squeeze(1).ne(0)
    return total, {
        "quality": quality.detach(),
        "pairwise": pairwise.detach(),
        "listwise": listwise.detach(),
        "decision": decision.detach(),
        "harmful_margin": harmful_margin.detach(),
        "hard_harm": hard_harm.detach(),
        "selected_iou_gain": gain.mean().detach(),
        "changed_rate": changed.float().mean().detach(),
        "beneficial_change_rate": gain.gt(0.01).float().mean().detach(),
        "harmful_change_rate": gain.lt(-0.01).float().mean().detach(),
    }
