"""Training-only v4 extension: dense C alignment and quality-only extra boxes.

Candidate selection and IoU labels are detached. Only the dense auxiliary
classification loss trains the original Head; v4 inference stays unchanged.
"""
import math
import torch
from torch.nn import functional as F
from lib.models.heads.candidate_quality_v2 import candidates


def dense_alignment(probability, iou, foreground):
    p = probability.flatten(1).float().clamp(1e-6, 1 - 1e-6)
    with torch.no_grad():
        support = foreground.flatten(1).float().clamp(0, 1)
        target = support * iou.detach().float().clamp(0, 1)
        positive = support > 0
    # Retain the official center loss separately. Balance foreground/background
    # so the many empty cells cannot solve this task by suppressing all scores.
    loss = F.binary_cross_entropy(p, target, reduction='none')
    loss = loss * (p.detach() - target).abs().square()
    terms = []
    for mask in (positive, ~positive):
        count = mask.sum(-1)
        valid = count > 0
        if valid.any():
            terms.append(((loss * mask).sum(-1) / count.clamp_min(1))[valid].mean())
    return sum(terms) / len(terms), target


def deployed_pair_loss(scores, target, temperature):
    delta = (target.detach()[:, 1:] - target.detach()[:, :1])
    margin = scores[:, 1:] - scores[:, :1]
    weight = delta.abs()
    return (F.softplus(-delta.sign() * margin / temperature) * weight).sum() / weight.sum().clamp_min(1e-6)


def expanded_candidates(out, template_mask, k, extra):
    """Top-K first, then unique neighbors of Top1 and next high responses.

    No GT is used to select extra candidates. All dense ROI features are
    computed only for the optional auxiliary quality branch.
    """
    n = out['search'].shape[1]
    side = math.isqrt(n)
    if side * side != n or not 0 < k <= n or not 0 <= extra <= n-k:
        raise ValueError('Invalid candidate expansion')
    dense = candidates(out, template_mask, k=n)
    rows = []
    for ordered in dense['indices'].detach().cpu().tolist():
        chosen = list(range(k))
        y, x = divmod(ordered[0], side)
        inverse = {index: rank for rank, index in enumerate(ordered)}
        for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            if 0 <= y+dy < side and 0 <= x+dx < side:
                rank = inverse[(y+dy)*side+x+dx]
                if rank not in chosen and len(chosen) < k+extra:
                    chosen.append(rank)
        for rank in range(k, n):
            if len(chosen) == k+extra:
                break
            if rank not in chosen:
                chosen.append(rank)
        rows.append(chosen)
    selected = torch.tensor(rows, device=out['search'].device)
    row = torch.arange(len(rows), device=selected.device)[:, None]
    keys = ('indices', 'boxes', 'pooled', 'evidence', 'score_features', 'spatial', 'scores')
    return {key: dense[key][row, selected] for key in keys}


def auxiliary_loss(model, subset, mask, foreground, gt, native_prediction,
                   native_target, iou_function, cfg, expand):
    dense_iou = iou_function(subset['bbox_map'].detach(), gt.detach()).detach()
    alignment, targets = dense_alignment(subset['cls_score_test'], dense_iou, foreground)
    rank = deployed_pair_loss(native_prediction['decision_score'], native_target,
                              cfg.TRAIN.RQC_TEMPERATURE)
    extra_loss = alignment * 0
    if expand and cfg.TRAIN.RQC_EXTRA_CANDIDATES > 0:
        k = cfg.MODEL.RQC.TOPK
        batch = expanded_candidates(subset, mask, k, cfg.TRAIN.RQC_EXTRA_CANDIDATES)
        prediction = model.rqc(batch)
        with torch.no_grad():
            extra_target = iou_function(batch['boxes'], gt)
        extra_loss = F.smooth_l1_loss(prediction['quality'][:, k:], extra_target[:, k:])
    total = (cfg.TRAIN.RQC_HEAD_QUALITY_WEIGHT * alignment
             + cfg.TRAIN.RQC_DEPLOYED_RANK_WEIGHT * rank
             + cfg.TRAIN.RQC_EXTRA_QUALITY_WEIGHT * extra_loss)
    return total, {
        'Loss/rqc_head_alignment': alignment.detach(),
        'Loss/rqc_deployed_rank': rank.detach(),
        'Loss/rqc_extra_quality': extra_loss.detach(),
        'RQC/dense_quality_target_mean': targets.mean().detach(),
    }
