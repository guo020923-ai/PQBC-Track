"""Complete-action supervised structured regret for the fixed ERC lattice."""

import torch


def box_iou_xywh(box, truth):
    box, truth = box.double(), truth.double()
    lo = torch.maximum(box[..., :2], truth[..., :2])
    hi = torch.minimum(box[..., :2] + box[..., 2:], truth[..., :2] + truth[..., 2:])
    intersection = (hi - lo).clamp_min(0).prod(-1)
    union = box[..., 2:].prod(-1) + truth[..., 2:].prod(-1) - intersection
    return intersection / union.clamp_min(torch.finfo(torch.double).tiny)


def success_auc21(iou):
    thresholds = torch.linspace(0, 1, 21, dtype=iou.dtype, device=iou.device)
    return iou[..., None].gt(thresholds).double().mean(-1)


def action_utility(boxes, truth):
    """GT-only label creation; no label is accepted by the model forward."""
    if boxes.ndim != 3 or boxes.shape[1:] != (25, 4):
        raise ValueError("boxes must be [batch,25,4]")
    if truth.shape != (len(boxes), 4):
        raise ValueError("truth shape mismatch")
    iou = box_iou_xywh(boxes, truth[:, None])
    utility = success_auc21(iou) + 1e-3 * iou
    return utility - utility[:, :1]


def structured_regret_loss(relative_score, executable, anchor_index, utility):
    """Margin-rescaled hinge upper-bounding chosen-action label regret."""
    batch, actions = relative_score.shape
    if actions != 25 or executable.shape != (batch, 25):
        raise ValueError("action schema mismatch")
    if utility.shape != (batch, 25) or not torch.isfinite(utility).all():
        raise ValueError("utility schema mismatch")
    rows = torch.arange(batch, device=relative_score.device)
    safe_score = relative_score.masked_fill(~executable, -torch.inf)
    safe_utility = utility.masked_fill(~executable, -torch.inf)
    best_gain, argbest = safe_utility.max(-1)
    # If no action is better than KEEP, teach exact abstention even when another
    # action ties at zero utility.
    best = torch.where(best_gain.gt(0), argbest, anchor_index)
    best_utility = utility[rows, best]
    cost = (best_utility[:, None] - utility).clamp_min(0)
    violator = (safe_score + cost).max(-1).values
    loss_per_sample = (violator - relative_score[rows, best]).clamp_min(0)
    if not torch.isfinite(loss_per_sample).all():
        raise FloatingPointError("non-finite structured regret")
    chosen = safe_score.argmax(-1)
    chosen_regret = best_utility - utility[rows, chosen]
    stats = {
        "loss": loss_per_sample.mean().detach(),
        "chosen_regret": chosen_regret.mean().detach(),
        "oracle_positive_rate": best_gain.gt(0).double().mean().detach(),
        "label_keep_rate": best.eq(anchor_index).double().mean().detach(),
        "predicted_keep_rate": chosen.eq(anchor_index).double().mean().detach(),
    }
    return loss_per_sample.mean(), stats
