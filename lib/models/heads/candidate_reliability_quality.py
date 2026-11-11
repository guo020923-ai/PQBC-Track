"""Independent ERC evidence estimator + RQC quality head; no legacy ERC changes.

Candidate selection uses deployed Hann*C*T, never GT. ERC estimates target
support (weak foreground supervision), not IoU. RQC estimates box IoU.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def candidates(out, template_mask, k=10):
    search = out['search'].detach()
    b, n, d = search.shape
    side = int(n ** .5)
    if side*side != n or not 1 <= k <= n:
        raise ValueError('Invalid grid or candidate count')
    cls = out['cls_score_test'].detach().reshape(b, n)
    logits = out['cont_score'].detach()
    # Native training is 2-logit; deployed SOFTMAX_ONE is 3-logit.
    if logits.shape[-1] == 2:
        logits = torch.cat((logits, torch.zeros_like(logits[..., :1])), -1)
    cont = logits.softmax(-1)[..., 0]
    hann = torch.as_tensor(np.outer(np.hanning(side), np.hanning(side)).flatten(), device=search.device)
    score = cls.double()*hann*cont.double()
    indices = torch.argsort(score, dim=-1, descending=True, stable=True)[:, :k]
    row = torch.arange(b, device=search.device)[:, None]
    boxes = out['bbox_map'].detach()[row, indices]
    point_features = search[row, indices]
    y, x = torch.meshgrid((torch.arange(side, device=search.device)+.5)/side,
                         (torch.arange(side, device=search.device)+.5)/side, indexing='ij')
    xy = torch.stack((x.flatten(), y.flatten()), -1)
    lo, hi = boxes[..., :2]-boxes[..., 2:]/2, boxes[..., :2]+boxes[..., 2:]/2
    region = ((xy[None, None] >= lo[:, :, None]) & (xy[None, None] <= hi[:, :, None])).all(-1)
    count = region.sum(-1, keepdim=True)
    pooled = torch.einsum('bkn,bnd->bkd', region.to(search), search)/count.clamp_min(1)
    pooled = torch.where(count > 0, pooled, point_features)
    mask = template_mask.reshape(b, -1).to(search)
    if not mask.bool().any(-1).all():
        raise ValueError('ERC reliability requires initial foreground reference')
    reference = (out['template'].detach()*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
    language = out['txt_token'].detach().reshape(b, -1, d).mean(1)
    prompts = out['prompts'].detach()
    normalized = F.normalize(pooled, dim=-1)
    visual = (normalized*F.normalize(reference, dim=-1)[:, None]).sum(-1)
    text = (normalized*F.normalize(language, dim=-1)[:, None]).sum(-1)
    competition = normalized @ F.normalize(prompts, dim=-1).transpose(-1, -2)
    c, t, s = cls[row, indices], cont[row, indices], score[row, indices]
    # No bbox coordinates or IoU in ERC inputs.
    evidence = torch.stack((visual, text, competition[..., 0], competition[..., 1],
                            competition[..., 2], competition[..., 0]-competition[..., 1],
                            competition[..., 0]-competition[..., 2], c, t, c-t), -1)
    score_features = torch.stack((c, t, hann[indices].to(c),
                                 (s.clamp_min(1e-30).log()-score.amax(-1,keepdim=True).clamp_min(1e-30).log()).clamp(-30,0).to(c)), -1)
    return dict(indices=indices, boxes=boxes, pooled=pooled, evidence=evidence,
                score_features=score_features, scores=s, dense_score=score,
                visual_reference=reference, language_reference=language,
                prompt_reference=prompts)


class EvidenceReliabilityQuality(nn.Module):
    def __init__(self, feature_dim=768, hidden=64):
        super().__init__()
        self.evidence_encoder = nn.Sequential(nn.Linear(10,32), nn.GELU(), nn.Linear(32,16), nn.GELU())
        self.reliability = nn.Linear(16,1)
        self.region = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim,hidden), nn.GELU())
        self.quality = nn.Sequential(nn.Linear(hidden+4+4+17,hidden), nn.GELU(), nn.Linear(hidden,1))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    def forward(self, batch, use_reliability=True):
        embedding = self.evidence_encoder(batch['evidence'])
        reliability_logit = self.reliability(embedding).squeeze(-1)
        conditioning = torch.cat((embedding, reliability_logit.sigmoid()[...,None]), -1)
        if not use_reliability:
            conditioning = torch.zeros_like(conditioning)
        q = self.quality(torch.cat((self.region(batch['pooled']), batch['boxes'],
                                    batch['score_features'], conditioning), -1)).squeeze(-1)
        return dict(reliability_logit=reliability_logit, quality_logit=q, quality=q.sigmoid())


def training_loss(pred, iou_target, foreground_target, foreground_valid):
    """IoU trains RQC only; foreground BCE trains reliability directly.

    Joint quality gradients may also adapt reliability embeddings. This does
    not make r an IoU prediction. OFF ablation requires separately fitted RQC.
    """
    target = iou_target.detach().clamp(0,1)
    valid = foreground_valid.detach().bool()
    r = pred['reliability_logit']
    rel = F.binary_cross_entropy_with_logits(r[valid], foreground_target.detach()[valid]) if valid.any() else r.sum()*0
    quality = F.smooth_l1_loss(pred['quality'], target)
    advantage = target[..., :,None]-target[...,None,:]
    pairs = advantage.gt(1e-6)
    diff = pred['quality_logit'][..., :,None]-pred['quality_logit'][...,None,:]
    rank = (F.softplus(-diff)*advantage.clamp_min(0))[pairs].mean() if pairs.any() else diff.sum()*0
    return quality+.25*rank+.1*rel, dict(quality=quality.detach(), rank=rank.detach(), reliability=rel.detach())
