"""Stateless ERC scorer over the fixed 25-action RQC30 candidate set.

Unlike the stopped paired-rollout policy, this module has no recurrent state
and no stochastic action sampler.  It is therefore trainable from shuffled
RQC-style supervised batches and has identical train/inference inputs.
"""

import torch
from torch import nn


VERSION = "rqc30_epoch_supervised_25action_stateless_v1"


class RQC30EpochERC(nn.Module):
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
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
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
    def strict_positive_choice(relative_score, executable, anchor_index):
        rows = torch.arange(len(relative_score), device=relative_score.device)
        alternatives = executable.clone()
        alternatives[rows, anchor_index] = False
        masked = relative_score.masked_fill(~alternatives, -torch.inf)
        candidate = masked.argmax(-1)
        positive = alternatives.any(-1) & relative_score[rows, candidate].gt(0)
        return torch.where(positive, candidate, anchor_index)

    def forward(self, evidence, executable, anchor_index):
        self._validate(evidence, executable, anchor_index)
        rows = torch.arange(len(evidence), device=evidence.device)
        encoded = self.encoder(evidence)
        anchor = encoded[rows, anchor_index]
        context = anchor[:, None].expand(-1, 25, -1)
        raw = self.readout(torch.cat((encoded, context), -1)).squeeze(-1)
        relative = raw - raw[rows, anchor_index, None]
        relative = relative.scatter(1, anchor_index[:, None], 0.0)
        relative = relative.masked_fill(~executable, -torch.inf)
        index = self.strict_positive_choice(relative, executable, anchor_index)
        return {"relative_score": relative, "index": index}
