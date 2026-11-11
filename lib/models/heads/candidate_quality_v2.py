"""Continuous spatial candidate features and Top1-relative RQC. No legacy edits."""
import torch
from torch import nn
from torch.nn import functional as F
from .candidate_reliability_quality import candidates as baseline_candidates


def candidates(out, template_mask, k=10):
    batch = baseline_candidates(out, template_mask, k)
    search = out['search'].detach()
    b, n, d = search.shape
    side = int(n ** .5)
    boxes = batch['boxes']
    # Nine bin centers, continuous normalized box coordinates; align_corners=False
    # matches token centers (i+.5)/side. Preserve spatial bins, not their mean.
    axis = (torch.arange(3, device=search.device, dtype=search.dtype)+.5)/3-.5
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    offsets = torch.stack((xx, yy), -1).reshape(1, 1, 9, 2)
    grid = boxes[..., None, :2] + offsets*boxes[..., None, 2:]
    feature_map = search.transpose(1, 2).reshape(b, d, side, side)
    sampled = F.grid_sample(feature_map, grid.reshape(b, k*9, 1, 2)*2-1,
                            mode='bilinear', padding_mode='border', align_corners=False)
    batch['spatial'] = sampled[..., 0].transpose(1, 2).reshape(b, k, 9, d)
    return batch


class CandidateQualityV2(nn.Module):
    def __init__(self, feature_dim=768, hidden=64):
        super().__init__()
        self.token = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 16), nn.GELU())
        self.region = nn.Sequential(nn.Linear(9*16, hidden), nn.GELU())
        # Absolute and Top1-relative region, geometry, response. No ERC in v2 pilot.
        self.quality = nn.Sequential(nn.Linear(2*hidden+16, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('v2 is RQC-only; ERC conditioning requires a separate experiment')
        region = self.region(self.token(batch['spatial']).flatten(-2))
        box = batch['boxes']
        top = box[:, :1]
        relative_box = torch.cat(((box[..., :2]-top[..., :2])/top[..., 2:].clamp_min(1e-4),
                                  (box[..., 2:].clamp_min(1e-4)/top[..., 2:].clamp_min(1e-4)).log()), -1).clamp(-10, 10)
        score = batch['score_features']
        inputs = torch.cat((region, region-region[:, :1], box, relative_box, score, score-score[:, :1]), -1)
        logit = self.quality(inputs).squeeze(-1)
        return dict(quality_logit=logit, quality=logit.sigmoid())


def training_loss(pred, iou_target, *unused):
    target = iou_target.detach().clamp(0, 1)
    quality = F.smooth_l1_loss(pred['quality'], target)
    advantage = target[:, 1:]-target[:, :1]
    delta = pred['quality_logit'][:, 1:]-pred['quality_logit'][:, :1]
    # Tiny IoU differences contribute proportionally less; no test threshold.
    weight = advantage.abs()
    rank = (F.softplus(-advantage.sign()*delta)*weight).mean()
    return quality+.25*rank, dict(quality=quality.detach(), rank=rank.detach())
