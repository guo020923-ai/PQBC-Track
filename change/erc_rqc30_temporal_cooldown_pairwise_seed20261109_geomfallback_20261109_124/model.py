import torch
from torch import nn


VERSION = "rqc30_execution_aligned_pairwise_cooldown8_100ep_v1"


class ExecutionAlignedPairwiseERC(nn.Module):
    def __init__(self, evidence_dim=48, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)
        self.version = VERSION

    @staticmethod
    def choose(score, executable, anchor, threshold):
        rows = torch.arange(len(score), device=score.device)
        allowed = executable & score.ge(float(threshold)); allowed[rows, anchor] = False
        candidate = score.masked_fill(~allowed, -torch.inf).argmax(-1)
        return torch.where(allowed.any(-1), candidate, anchor)

    def forward(self, evidence, executable, anchor):
        if evidence.shape[1:] != (25, 48) or executable.shape != (len(evidence), 25):
            raise ValueError("schema")
        rows = torch.arange(len(evidence), device=evidence.device)
        encoded = self.encoder(evidence); anchor_encoded = encoded[rows, anchor]
        context = anchor_encoded[:, None].expand(-1, 25, -1)
        raw = self.readout(torch.cat((encoded, context, encoded - context), -1)).squeeze(-1)
        score = raw - raw[rows, anchor, None]
        score = score.scatter(1, anchor[:, None], 0.0)
        if not torch.isfinite(score).all():
            raise FloatingPointError("score")
        return {"utility_score": score}
