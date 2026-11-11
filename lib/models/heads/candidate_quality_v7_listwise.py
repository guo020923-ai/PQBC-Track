"""Target-evidence-conditioned listwise candidate quality calibration.

Earlier RQC variants scored candidates independently.  That is a poor match
for the deployed decision, which is inherently relative: a candidate is useful
only when it is better than the response Top1 and sufficiently better to
overcome its response-score deficit.  V7 therefore encodes the whole Top-K set
jointly and predicts a bounded, Top1-relative quality residual.

The module does not generate or move boxes.  The original Hann*C*T response is
kept as the decision prior, and zero initialization exactly recovers its Top1.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_quality_v2 import candidates


class CandidateQualityV7Listwise(nn.Module):
    def __init__(self, feature_dim=768, hidden=128, k=10, scale=1.0):
        super().__init__()
        if hidden % 4:
            raise ValueError("hidden must be divisible by four")
        if k < 2 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid Top-K or decision scale")
        self.k = int(k)
        self.scale = float(scale)

        # Preserve the 3x3 candidate layout instead of reducing it to one mean.
        self.spatial_token = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 16), nn.GELU()
        )
        self.spatial_region = nn.Sequential(
            nn.Linear(9 * 16, hidden // 2), nn.GELU()
        )
        # The pooled search embedding carries appearance information that the
        # earlier small heads discarded after scalar cosine similarities.
        self.appearance = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden // 2),
            nn.GELU(),
        )
        # box + box relative to Top1 + response + response relative to Top1 +
        # visual/language/target-distractor evidence and its Top1 difference.
        self.scalar = nn.Sequential(
            nn.LayerNorm(36), nn.Linear(36, hidden // 2), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(3 * (hidden // 2), hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.rank_embedding = nn.Parameter(torch.zeros(1, k, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=3 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=2)
        self.quality = nn.Sequential(
            nn.Linear(2 * hidden + 14, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Exact baseline identity before learning: every candidate receives the
        # same zero residual and the original response ordering is unchanged.
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    @staticmethod
    def _relative_box(box):
        top = box[:, :1]
        return torch.cat((
            (box[..., :2] - top[..., :2]) / top[..., 2:].clamp_min(1e-4),
            (box[..., 2:].clamp_min(1e-4)
             / top[..., 2:].clamp_min(1e-4)).log(),
        ), dim=-1).clamp(-10, 10)

    def encode(self, batch):
        if batch["boxes"].shape[1] != self.k:
            raise ValueError("RQC-v7 candidate count differs from configured K")

        spatial = self.spatial_region(
            self.spatial_token(batch["spatial"]).flatten(-2)
        )
        appearance = self.appearance(batch["pooled"])
        box = batch["boxes"]
        relative_box = self._relative_box(box)
        score = batch["score_features"]
        evidence = batch["evidence"].detach()
        scalar = torch.cat((
            box,
            relative_box,
            score,
            score - score[:, :1],
            evidence,
            evidence - evidence[:, :1],
        ), dim=-1)
        token = self.fuse(torch.cat((spatial, appearance, self.scalar(scalar)), -1))
        contextual = self.context(token + self.rank_embedding)
        relative_context = contextual - contextual[:, :1]
        head_input = torch.cat((
            contextual,
            relative_context,
            score - score[:, :1],
            evidence - evidence[:, :1],
        ), dim=-1)
        return head_input, score

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError("RQC-v7 is RQC-only; ERC is deliberately disabled")
        head_input, score = self.encode(batch)
        raw_quality_logit = self.quality(head_input).squeeze(-1)
        # Bounded logits prevent a small development set from overwhelming the
        # original response prior, while still allowing a two-logit correction.
        quality_logit = 2.0 * torch.tanh(raw_quality_logit / 2.0)
        quality = quality_logit.sigmoid()
        residual = self.scale * (quality_logit - quality_logit[:, :1])
        decision_score = score[..., 3].detach() + residual
        return {
            "quality_logit": quality_logit,
            "quality": quality,
            "decision_residual": residual,
            "decision_score": decision_score,
        }


def training_loss(pred, iou_target, temperature=0.10):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    target = iou_target.detach().clamp(0, 1)
    logits = pred["quality_logit"]

    # Soft-label BCE gives a calibrated localization quality estimate; listwise
    # and pairwise terms train the relative ordering actually used at test time.
    quality = F.binary_cross_entropy_with_logits(logits, target)
    delta = target[:, :, None] - target[:, None, :]
    logit_delta = logits[:, :, None] - logits[:, None, :]
    pair_mask = delta.abs().gt(0.02)
    pairwise = (
        F.softplus(-delta.sign() * logit_delta) * delta.abs()
    )[pair_mask].mean() if pair_mask.any() else logits.sum() * 0.0
    oracle_distribution = F.softmax(target / temperature, dim=-1)
    listwise = F.kl_div(
        F.log_softmax(logits / temperature, dim=-1),
        oracle_distribution,
        reduction="batchmean",
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

    # Explicitly discourage any deployed score margin for candidates whose box
    # is worse than Top1.  This uses the exact inference score, not a proxy.
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
        quality
        + 0.25 * pairwise
        + 0.10 * listwise
        + decision
        + 0.75 * harmful_margin
        + hard_harm
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
