"""Response-prior quality calibration; shared train/inference decision scores."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .candidate_quality_v2 import CandidateQualityV2, candidates


def decision_scores(quality, score_features, scale=1.0):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('Invalid quality fusion scale')
    # Cached feature 3 is log(S_i/S_top1), clipped at -30.
    # Subtracting the common Top1 score does not affect the argmax.
    return score_features[..., 3].detach() + scale*(quality-quality[:, :1])


class CandidateQualityV3(CandidateQualityV2):
    def __init__(self, feature_dim=768, hidden=64, reference=False, scale=1.0):
        super().__init__(feature_dim, hidden)
        self.reference = bool(reference)
        self.scale = float(scale)
        # Same parameterization in both variants; reference-off inputs are zero.
        self.quality = nn.Sequential(nn.Linear(2*hidden+16+20, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    def forward(self, batch, use_reliability=False):
        if use_reliability:
            raise ValueError('Reference evidence is not a trained ERC reliability module')
        region = self.region(self.token(batch['spatial']).flatten(-2))
        box = batch['boxes'];top = box[:, :1];score = batch['score_features']
        relative_box = torch.cat(((box[..., :2]-top[..., :2])/top[..., 2:].clamp_min(1e-4),
                                  (box[..., 2:].clamp_min(1e-4)/top[..., 2:].clamp_min(1e-4)).log()), -1).clamp(-10,10)
        evidence = batch['evidence'].detach()
        context = torch.cat((evidence, evidence-evidence[:, :1]), -1)
        if not self.reference:context = torch.zeros_like(context)
        x = torch.cat((region,region-region[:, :1],box,relative_box,score,score-score[:, :1],context), -1)
        logit = self.quality(x).squeeze(-1);q = logit.sigmoid()
        return dict(quality_logit=logit,quality=q,decision_score=decision_scores(q,score,self.scale))


def training_loss(pred, iou_target, temperature=.1, decision_weight=1.0):
    if temperature <= 0 or decision_weight < 0:raise ValueError('Invalid loss settings')
    target = iou_target.detach().clamp(0,1)
    quality = F.smooth_l1_loss(pred['quality'],target)
    regret = target.max(-1,keepdim=True).values-target
    decision = (F.softmax(pred['decision_score']/temperature,-1)*regret).sum(-1).mean()
    return quality+decision_weight*decision, dict(quality=quality.detach(),decision=decision.detach())
