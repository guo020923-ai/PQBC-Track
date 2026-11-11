"""Hard-decision-aligned supervision for independent-branch PQ-ERC v3."""

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


@torch.no_grad()
def _hard_oracle(pred, target, support, valid, cfg):
    maximum = float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
    values = torch.tensor(
        [float(value) for value in cfg.TRAIN.ERC_PQ3_DELTA_GRID],
        device=target.device, dtype=target.dtype)
    if not bool(values.eq(0.0).any()):
        raise ValueError('ERC_PQ3_DELTA_GRID must contain zero')
    if float(values.abs().max()) > maximum + 1e-8:
        raise ValueError('PQ3 oracle residual exceeds MAX_ALPHA')
    delta_c, delta_t = torch.meshgrid(values, values, indexing='ij')
    actions = torch.stack([delta_c.flatten(), delta_t.flatten()], dim=-1)
    keep = actions.abs().sum(-1).eq(0.0).nonzero(as_tuple=False)[0, 0]

    # Preserve the deployed score ordering.  The Hann support excludes its
    # exact-zero boundary, so only a dtype-tiny numerical floor is required.
    base = pred['erc_windowed_base_score'].detach()
    tiny = torch.finfo(base.dtype).tiny
    log_base = base.clamp_min(tiny).log()
    log_c = pred['erc_pq3_centered_log_c'].detach().flatten(1)
    log_t = pred['erc_pq3_centered_log_t'].detach().flatten(1)
    candidate_log = (
        log_base[:, None, :]
        + actions[None, :, 0, None] * log_c[:, None, :]
        + actions[None, :, 1, None] * log_t[:, None, :]
    )
    expanded_support = support[:, None, :].expand_as(candidate_log)
    minimum = torch.finfo(candidate_log.dtype).min
    index = candidate_log.masked_fill(~expanded_support, minimum).argmax(-1)
    expanded_target = target[:, None, :].expand_as(candidate_log)
    quality = expanded_target.gather(2, index[:, :, None]).squeeze(-1)
    base_quality = quality[:, keep]
    gain = quality - base_quality[:, None]
    cost = float(cfg.TRAIN.ERC_PQ3_ACTION_COST) * (
        actions.abs().sum(-1) / max(2.0 * maximum, 1e-6))
    utility = gain - cost[None, :]
    utility[:, keep] = 0.0
    best_utility, best_action = utility.max(-1)
    rows = torch.arange(target.shape[0], device=target.device)
    best_index = index[rows, best_action]
    best_gain = gain[rows, best_action]
    actionable = (
        valid
        & best_action.ne(keep)
        & best_index.ne(index[:, keep])
        & best_gain.gt(float(cfg.TRAIN.ERC_PQ3_MIN_HARD_GAIN))
        & best_utility.gt(0.0)
    )
    oracle_delta = torch.where(
        actionable[:, None], actions[best_action],
        torch.zeros_like(actions[best_action]))
    return {
        'log_base': log_base,
        'base_index': index[:, keep],
        'oracle_index': best_index,
        'oracle_delta': oracle_delta,
        'oracle_gain': torch.where(
            actionable, best_gain, torch.zeros_like(best_gain)),
        'actionable': actionable,
    }


def compute_peak_quality_independent_branch_loss_and_metrics(
        pred, gt_heatmap, cfg, dataset_ids=None):
    control = pred['erc_pq3_raw_delta']
    zero = control.sum() * 0.0
    active = pred['erc_active_mask'].reshape(-1).bool()
    support = pred['erc_pq_support_mask'].detach().flatten(1).bool()
    target = gt_heatmap.detach().flatten(1).to(control).clamp_min(0.0)
    target = target / target.amax(-1, keepdim=True).clamp_min(1e-6)
    valid = active & support.any(-1) & target.amax(-1).gt(0.0)
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/pq3_valid_fraction': valid.float().mean(),
        }

    oracle = _hard_oracle(pred, target, support, valid, cfg)
    actionable = oracle['actionable']
    preserve = valid & ~actionable
    maximum = float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
    normalized = control / max(maximum, 1e-6)
    normalized_target = oracle['oracle_delta'] / max(maximum, 1e-6)
    control_element = F.smooth_l1_loss(
        normalized, normalized_target, reduction='none').mean(-1)
    preserve_element = normalized.abs().mean(-1)

    predicted_log = (
        oracle['log_base'] + pred['erc_pq3_raw_correction'].flatten(1))
    rows = torch.arange(control.shape[0], device=control.device)
    oracle_score = predicted_log[rows, oracle['oracle_index']]
    base_score = predicted_log[rows, oracle['base_index']]
    rank_element = F.relu(
        float(cfg.TRAIN.ERC_PQ3_RANK_MARGIN) - oracle_score + base_score)

    worst = float(cfg.TRAIN.ERC_PQ3_DOMAIN_WORST_WEIGHT)
    reduce = lambda values, mask: _domain_mean(
        values, mask, dataset_ids, worst, zero)
    control_loss = reduce(control_element, actionable)
    rank_loss = reduce(rank_element, actionable)
    preserve_loss = reduce(preserve_element, preserve)
    total = (
        float(cfg.TRAIN.ERC_PQ3_CONTROL_WEIGHT) * control_loss
        + float(cfg.TRAIN.ERC_PQ3_RANK_WEIGHT) * rank_loss
        + float(cfg.TRAIN.ERC_PQ3_PRESERVE_WEIGHT) * preserve_loss
    )

    with torch.no_grad():
        minimum = torch.finfo(predicted_log.dtype).min
        predicted_index = predicted_log.masked_fill(~support, minimum).argmax(-1)
        changed = valid & predicted_index.ne(oracle['base_index'])
        predicted_quality = target.gather(
            1, predicted_index[:, None]).squeeze(1)
        base_quality = target.gather(
            1, oracle['base_index'][:, None]).squeeze(1)
        hard_gain = predicted_quality - base_quality
        beneficial = changed & hard_gain.gt(1e-8)
        harmful = changed & hard_gain.lt(-1e-8)
        valid_count = valid.float().sum().clamp_min(1.0)

        # Per-valid numerators aggregate correctly across batches.  Dividing
        # changed_gain_numerator by selection_change_rate after an epoch gives
        # the true gain conditioned on a changed decision, including batches
        # with no changed samples.
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_pq3_control': control_loss.detach(),
            'Loss/erc_pq3_rank': rank_loss.detach(),
            'Loss/erc_pq3_preserve': preserve_loss.detach(),
            'ERC/pq3_valid_fraction': valid.float().mean(),
            'ERC/pq3_oracle_actionable_rate': actionable.float().sum() / valid_count,
            'ERC/pq3_oracle_gain_per_valid': oracle['oracle_gain'].sum() / valid_count,
            'ERC/pq3_oracle_delta_c_per_valid': oracle['oracle_delta'][:, 0].sum() / valid_count,
            'ERC/pq3_oracle_delta_t_per_valid': oracle['oracle_delta'][:, 1].sum() / valid_count,
            'ERC/pq3_oracle_c_nonzero_rate': (
                actionable & oracle['oracle_delta'][:, 0].ne(0)).float().sum() / valid_count,
            'ERC/pq3_oracle_t_nonzero_rate': (
                actionable & oracle['oracle_delta'][:, 1].ne(0)).float().sum() / valid_count,
            'ERC/pq3_delta_c_mean': control[valid, 0].mean(),
            'ERC/pq3_delta_t_mean': control[valid, 1].mean(),
            'ERC/pq3_delta_c_abs': control[valid, 0].abs().mean(),
            'ERC/pq3_delta_t_abs': control[valid, 1].abs().mean(),
            'ERC/pq3_selection_change_rate': changed.float().sum() / valid_count,
            'ERC/pq3_response_gain_per_valid': (hard_gain * valid).sum() / valid_count,
            'ERC/pq3_changed_gain_numerator': (hard_gain * changed).sum() / valid_count,
            'ERC/pq3_beneficial_change_rate': beneficial.float().sum() / valid_count,
            'ERC/pq3_harmful_change_rate': harmful.float().sum() / valid_count,
        }
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for dataset_id in ids[valid].unique(sorted=True):
                mask = valid & ids.eq(dataset_id)
                count = mask.float().sum().clamp_min(1.0)
                suffix = str(int(dataset_id.item()))
                status['ERC/pq3_domain_' + suffix + '_response_gain'] = (
                    (hard_gain * mask).sum() / count)
                status['ERC/pq3_domain_' + suffix + '_harmful_rate'] = (
                    harmful[mask].float().mean())
    return total, status
