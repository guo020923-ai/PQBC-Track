"""Joint KEEP/action utility prediction with conservative deployment.

The previous factorized policy answered two different questions with two
independent heads: whether any resize was useful, and which resize to use.
Those answers could disagree.  This module instead predicts the relative
utility of every executable action on one common scale.  KEEP is fixed at
zero and competes directly with all 24 resize actions.
"""

import torch
from torch import nn


VERSION = "rqc30_harm_aware_joint_utility_100ep_v1"
MIN_DEPLOY_GAIN = 1.0 / 21.0


class HarmAwareJointUtilityERC(nn.Module):
    def __init__(self, evidence_dim=48, hidden_dim=64):
        super().__init__()
        if evidence_dim != 48 or hidden_dim != 64:
            raise ValueError("fixed evidence/hidden dimensions required")
        self.encoder = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Action, RQC anchor, and their difference make the comparison
        # explicit while retaining a single shared action-value function.
        self.utility_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Exact zero initialization is a safety invariant: an untrained model
        # must reproduce RQC rather than resize every box.
        nn.init.zeros_(self.utility_readout[-1].weight)
        nn.init.zeros_(self.utility_readout[-1].bias)
        self.version = VERSION

    @staticmethod
    def _validate(evidence, executable, anchor_index):
        if evidence.ndim != 3 or evidence.shape[1:] != (25, 48):
            raise ValueError("evidence must be [batch,25,48]")
        batch = len(evidence)
        if executable.shape != (batch, 25) or executable.dtype != torch.bool:
            raise ValueError("executable mask mismatch")
        if anchor_index.shape != (batch,) or anchor_index.dtype != torch.long:
            raise ValueError("anchor index mismatch")
        if not torch.isfinite(evidence).all():
            raise ValueError("non-finite evidence")
        rows = torch.arange(batch, device=evidence.device)
        if (anchor_index < 0).any() or (anchor_index >= 25).any():
            raise ValueError("anchor outside action set")
        if not executable[rows, anchor_index].all():
            raise ValueError("RQC anchor must remain executable")

    @staticmethod
    def choose(predicted_gain, executable, anchor_index, threshold=MIN_DEPLOY_GAIN):
        """Choose jointly, then abstain unless predicted gain clears one AUC step."""
        if threshold < 0:
            raise ValueError("deployment threshold must be nonnegative")
        rows = torch.arange(len(predicted_gain), device=predicted_gain.device)
        alternatives = executable.clone()
        alternatives[rows, anchor_index] = False
        candidate = predicted_gain.masked_fill(~alternatives, -torch.inf).argmax(-1)
        candidate_gain = predicted_gain[rows, candidate]
        active = alternatives.any(-1) & candidate_gain.ge(threshold)
        return torch.where(active, candidate, anchor_index)

    def forward(self, evidence, executable, anchor_index):
        self._validate(evidence, executable, anchor_index)
        rows = torch.arange(len(evidence), device=evidence.device)
        encoded = self.encoder(evidence)
        anchor = encoded[rows, anchor_index]
        anchor_context = anchor[:, None].expand(-1, 25, -1)
        comparison = torch.cat((encoded, anchor_context, encoded - anchor_context), -1)
        raw = self.utility_readout(comparison).squeeze(-1)
        predicted_gain = raw - raw[rows, anchor_index, None]
        predicted_gain = predicted_gain.scatter(1, anchor_index[:, None], 0.0)
        if not torch.isfinite(predicted_gain).all():
            raise FloatingPointError("non-finite predicted utility")
        decision_score = predicted_gain.masked_fill(~executable, -torch.inf)
        index = self.choose(predicted_gain, executable, anchor_index)
        return {
            "predicted_gain": predicted_gain,
            "decision_score": decision_score,
            "index": index,
        }

