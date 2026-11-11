"""Decision-aligned response supervision for direct dense ERC DR1."""

import torch
from torch.nn import functional as F


def _domain_mean(values, mask, dataset_ids, worst_weight, zero):
    if not mask.any():
        return zero
    if dataset_ids is None:
        return values[mask].float().mean()
    ids = dataset_ids.reshape(-1).to(values.device)
    groups = []
    for dataset_id in ids[mask].unique(sorted=True):
        selected = mask & ids.eq(dataset_id)
        if selected.any():
            groups.append(values[selected].float().mean())
    if not groups:
        return zero
    groups = torch.stack(groups)
    weight = max(float(worst_weight), 0.0)
    return (groups.mean() + weight * groups.max()) / (1.0 + weight)


def compute_target_reference_direct_dense_loss_and_metrics(
        pred, gt_heatmap, cfg, dataset_ids=None):
    correction = pred['erc_dr1_raw_correction']
    zero = correction.sum() * 0.0
    active = pred['erc_active_mask'].reshape(-1).bool()
    support = pred['erc_dr1_support_mask'].bool()
    target = gt_heatmap.detach().to(correction).clamp_min(0.0)
    target = target / target.flatten(1).amax(-1).clamp_min(1e-6)[:, None, None]
    batch = target.shape[0]
    rows = torch.arange(batch, device=target.device)
    target_index = target.flatten(1).argmax(-1)

    tiny = torch.finfo(correction.dtype).tiny
    base_log = pred['erc_windowed_base_score'].detach().clamp_min(tiny).log()
    baseline_index = base_log.argmax(-1)
    target_flat = target.flatten(1)
    base_quality = target_flat[rows, baseline_index]
    target_quality = target_flat[rows, target_index]
    response_gain = target_quality - base_quality
    required_span = base_log[rows, baseline_index] - base_log[rows, target_index]
    valid = active & target_flat.amax(-1).gt(0.0)
    correctable = (
        valid & target_index.ne(baseline_index)
        & response_gain.gt(float(cfg.TRAIN.ERC_DR1_MIN_HARD_GAIN)))
    reachable = correctable & required_span.lt(
        2.0 * float(cfg.MODEL.HEAD.ERC.MAX_ALPHA))
    preserve = valid & ~reachable
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/dr1_valid_fraction': valid.float().mean(),
        }

    predicted_log = base_log + correction.flatten(1)
    minimum = torch.finfo(predicted_log.dtype).min
    predicted_log = predicted_log.masked_fill(~support.flatten(1), minimum)
    decision_element = F.cross_entropy(
        predicted_log, target_index, reduction='none')
    target_score = predicted_log[rows, target_index]
    baseline_score = predicted_log[rows, baseline_index]
    rank_element = F.relu(
        float(cfg.TRAIN.ERC_DR1_RANK_MARGIN)
        - target_score + baseline_score)
    target_residual = correction.flatten(1)[rows, target_index]
    baseline_residual = correction.flatten(1)[rows, baseline_index]
    max_alpha = float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
    pair_element = (
        F.smooth_l1_loss(
            target_residual, torch.full_like(target_residual, max_alpha),
            reduction='none')
        + F.smooth_l1_loss(
            baseline_residual, torch.full_like(baseline_residual, -max_alpha),
            reduction='none'))
    preserve_element = correction.abs().flatten(1).mean(-1)

    worst = float(cfg.TRAIN.ERC_DR1_DOMAIN_WORST_WEIGHT)
    reduce = lambda values, mask: _domain_mean(
        values, mask, dataset_ids, worst, zero)
    decision_loss = reduce(decision_element, reachable)
    rank_loss = reduce(rank_element, reachable)
    pair_loss = reduce(pair_element, reachable)
    preserve_loss = reduce(preserve_element, preserve)
    total = (
        float(cfg.TRAIN.ERC_DR1_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_DR1_RANK_WEIGHT) * rank_loss
        + float(cfg.TRAIN.ERC_DR1_PAIR_WEIGHT) * pair_loss
        + float(cfg.TRAIN.ERC_DR1_PRESERVE_WEIGHT) * preserve_loss)

    with torch.no_grad():
        predicted_index = predicted_log.argmax(-1)
        changed = valid & predicted_index.ne(baseline_index)
        predicted_quality = target_flat[rows, predicted_index]
        hard_gain = predicted_quality - base_quality
        beneficial = changed & hard_gain.gt(1e-8)
        harmful = changed & hard_gain.lt(-1e-8)
        valid_count = valid.float().sum().clamp_min(1.0)
        changed_count = changed.float().sum().clamp_min(1.0)
        reachable_count = reachable.float().sum().clamp_min(1.0)
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_dr1_decision': decision_loss.detach(),
            'Loss/erc_dr1_rank': rank_loss.detach(),
            'Loss/erc_dr1_pair': pair_loss.detach(),
            'Loss/erc_dr1_preserve': preserve_loss.detach(),
            'ERC/dr1_valid_fraction': valid.float().mean(),
            'ERC/dr1_correctable_rate': correctable.float().sum() / valid_count,
            'ERC/dr1_reachable_rate': reachable.float().sum() / valid_count,
            'ERC/dr1_required_span_mean': required_span[reachable].mean()
            if reachable.any() else zero.detach(),
            'ERC/dr1_correction_abs': correction[valid].abs().mean(),
            'ERC/dr1_correction_saturation': (
                correction[valid].abs().gt(0.95 * max_alpha).float().mean()),
            'ERC/dr1_selection_change_rate': changed.float().sum() / valid_count,
            'ERC/dr1_response_gain_per_valid': (hard_gain * valid).sum() / valid_count,
            'ERC/dr1_changed_gain': (hard_gain * changed).sum() / changed_count,
            'ERC/dr1_beneficial_change_rate': beneficial.float().sum() / valid_count,
            'ERC/dr1_harmful_change_rate': harmful.float().sum() / valid_count,
            'ERC/dr1_reachable_hit_rate': (
                (predicted_index.eq(target_index) & reachable).float().sum()
                / reachable_count),
        }
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for dataset_id in ids[valid].unique(sorted=True):
                mask = valid & ids.eq(dataset_id)
                count = mask.float().sum().clamp_min(1.0)
                suffix = str(int(dataset_id.item()))
                status['ERC/dr1_domain_' + suffix + '_response_gain'] = (
                    (hard_gain * mask).sum() / count)
                status['ERC/dr1_domain_' + suffix + '_harmful_rate'] = (
                    harmful[mask].float().mean())
    return total, status
