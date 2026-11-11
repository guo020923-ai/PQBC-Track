"""Decision-aligned response-only supervision for PQ-ERC v2.

The oracle searches only scalar C/T fusion coefficients that the deployed
module can realize.  Its utility is measured on the official centre heatmap at
the hard Hann-aware selected location, with a soft full-map term used to expose
stable gradients.  Bounding boxes and IoU are deliberately absent.
"""

import torch
from torch.nn import functional as F


def _masked_mean(values, mask, zero):
    if mask.any():
        return values[mask].float().mean()
    return zero


def _domain_robust_mean(values, valid, dataset_ids, worst_weight, zero):
    if not valid.any():
        return zero
    if dataset_ids is None:
        return values[valid].float().mean()
    dataset_ids = dataset_ids.reshape(-1).to(values.device)
    if dataset_ids.numel() != values.numel():
        raise ValueError('PQ-ERC v2 dataset_id batch shape mismatch')
    groups = []
    for dataset_id in dataset_ids[valid].unique(sorted=True):
        selected = valid & dataset_ids.eq(dataset_id)
        if selected.any():
            groups.append(values[selected].float().mean())
    if not groups:
        return zero
    groups = torch.stack(groups)
    weight = max(float(worst_weight), 0.0)
    return (groups.mean() + weight * groups.max()) / (1.0 + weight)


def _soft_quality(log_score, target, support, temperature):
    minimum = torch.finfo(log_score.dtype).min
    probability = F.softmax(
        (log_score / temperature).masked_fill(~support, minimum), dim=-1)
    return (probability * target).sum(-1)


@torch.no_grad()
def _build_oracle(pred, target, support, valid, cfg):
    eps = max(float(cfg.MODEL.HEAD.ERC.EPS), 1e-12)
    maximum = float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
    actions = torch.tensor(
        [float(value) for value in cfg.TRAIN.ERC_PQ2_ORACLE_ALPHA_GRID],
        device=target.device, dtype=target.dtype)
    if not bool(actions.eq(0.0).any()):
        raise ValueError('ERC_PQ2_ORACLE_ALPHA_GRID must contain zero')
    if not bool(actions.lt(0.0).any()) or not bool(actions.gt(0.0).any()):
        raise ValueError('PQ-ERC v2 oracle needs both C and T directions')
    if float(actions.abs().max()) > maximum + 1e-8:
        raise ValueError('PQ-ERC v2 oracle action exceeds MAX_ALPHA')

    log_base = pred['erc_windowed_base_score'].detach().clamp_min(eps).log()
    difference = pred['erc_pq_difference'].detach().flatten(1)
    candidate_log = (
        log_base[:, None, :] + actions[None, :, None] * difference[:, None, :])
    expanded_support = support[:, None, :].expand_as(candidate_log)
    minimum = torch.finfo(candidate_log.dtype).min
    hard_index = candidate_log.masked_fill(
        ~expanded_support, minimum).argmax(-1)
    expanded_target = target[:, None, :].expand_as(candidate_log)
    hard_quality = expanded_target.gather(
        2, hard_index[:, :, None]).squeeze(-1)
    temperature = max(float(cfg.TRAIN.ERC_PQ2_SOFT_TEMPERATURE), 1e-6)
    soft_quality = _soft_quality(
        candidate_log.flatten(0, 1),
        expanded_target.flatten(0, 1),
        expanded_support.flatten(0, 1), temperature,
    ).reshape_as(hard_quality)
    soft_mix = float(cfg.TRAIN.ERC_PQ2_SOFT_ORACLE_MIX)
    quality = (1.0 - soft_mix) * hard_quality + soft_mix * soft_quality

    keep_index = actions.eq(0.0).nonzero(as_tuple=False)[0, 0]
    base_quality = quality[:, keep_index]
    action_cost = float(cfg.TRAIN.ERC_PQ2_ACTION_COST) * (
        actions.abs() / max(maximum, 1e-6))
    utility = quality - base_quality[:, None] - action_cost[None, :]
    utility[:, keep_index] = 0.0
    best_utility, best_action = utility.max(-1)
    rows = torch.arange(target.shape[0], device=target.device)
    best_quality = quality[rows, best_action]
    oracle_gain = best_quality - base_quality
    actionable = (
        valid
        & best_utility.gt(0.0)
        & oracle_gain.gt(float(cfg.TRAIN.ERC_PQ2_MIN_RESPONSE_GAIN)))
    oracle_alpha = torch.where(
        actionable, actions[best_action], torch.zeros_like(oracle_gain))

    positive_quality = quality[:, actions.gt(0.0)].amax(-1)
    negative_quality = quality[:, actions.lt(0.0)].amax(-1)
    branch_quality_target = torch.stack(
        [positive_quality, negative_quality], dim=-1).clamp(0.0, 1.0)
    gate_target = (oracle_alpha.abs() / max(maximum, 1e-6)).clamp(0.0, 1.0)
    oracle_log = log_base + oracle_alpha[:, None] * difference
    oracle_soft_quality = _soft_quality(
        oracle_log, target, support, temperature)
    base_soft_quality = _soft_quality(
        log_base, target, support, temperature)
    return {
        'actions': actions,
        'log_base': log_base,
        'difference': difference,
        'base_index': hard_index[:, keep_index],
        'base_quality': base_quality,
        'base_soft_quality': base_soft_quality,
        'oracle_alpha': oracle_alpha,
        'oracle_gain': oracle_gain,
        'oracle_soft_quality': oracle_soft_quality,
        'branch_quality_target': branch_quality_target,
        'gate_target': gate_target,
        'actionable': actionable,
    }


def compute_peak_quality_decision_loss_and_metrics(
        pred, gt_heatmap, cfg, dataset_ids=None):
    reliability = pred['erc_pq_reliability']
    zero = reliability.sum() * 0.0
    active = pred['erc_active_mask'].reshape(-1).bool()
    support = pred['erc_pq_support_mask'].detach().flatten(1).bool()
    target = gt_heatmap.detach().flatten(1).to(
        device=reliability.device, dtype=reliability.dtype).clamp_min(0.0)
    target = target / target.amax(-1, keepdim=True).clamp_min(1e-6)
    valid = active & support.any(-1) & target.amax(-1).gt(0.0)
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/pq2_valid_fraction': valid.float().mean(),
        }

    oracle = _build_oracle(pred, target, support, valid, cfg)
    quality_target = oracle['branch_quality_target']
    quality_element = F.smooth_l1_loss(
        reliability, quality_target, reduction='none').mean(-1)
    calibration_element = F.binary_cross_entropy(
        reliability.clamp(1e-6, 1.0 - 1e-6),
        quality_target, reduction='none').mean(-1)

    gate = pred['erc_pq2_gate']
    gate_element = F.binary_cross_entropy(
        gate.clamp(1e-6, 1.0 - 1e-6),
        oracle['gate_target'], reduction='none')
    maximum = float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
    predicted_alpha = pred['erc_pq2_raw_alpha']
    alpha_element = F.smooth_l1_loss(
        predicted_alpha / max(maximum, 1e-6),
        oracle['oracle_alpha'] / max(maximum, 1e-6),
        reduction='none')

    actionable = oracle['actionable']
    direction_target = oracle['oracle_alpha'].gt(0.0).to(reliability.dtype)
    direction_element = F.binary_cross_entropy_with_logits(
        pred['erc_pq_reliability_gap'], direction_target, reduction='none')

    predicted_log = (
        oracle['log_base']
        + pred['erc_pq_raw_correction'].flatten(1))
    temperature = max(float(cfg.TRAIN.ERC_PQ2_SOFT_TEMPERATURE), 1e-6)
    predicted_soft_quality = _soft_quality(
        predicted_log, target, support, temperature)
    decision_element = F.relu(
        oracle['oracle_soft_quality'] - predicted_soft_quality)
    noharm_element = F.relu(
        oracle['base_soft_quality'] - predicted_soft_quality)
    preserve = valid & ~actionable
    preserve_element = predicted_alpha.abs() / max(maximum, 1e-6)

    worst_weight = float(cfg.TRAIN.ERC_PQ2_DOMAIN_WORST_WEIGHT)
    reduce = lambda values, mask: _domain_robust_mean(
        values, mask, dataset_ids, worst_weight, zero)
    quality_loss = reduce(quality_element, valid)
    calibration_loss = reduce(calibration_element, valid)
    gate_loss = reduce(gate_element, valid)
    alpha_loss = reduce(alpha_element, valid)
    direction_loss = reduce(direction_element, actionable)
    decision_loss = reduce(decision_element, valid)
    noharm_loss = reduce(noharm_element, valid)
    preserve_loss = reduce(preserve_element, preserve)
    total = (
        float(cfg.TRAIN.ERC_PQ2_QUALITY_WEIGHT) * quality_loss
        + float(cfg.TRAIN.ERC_PQ2_CALIBRATION_WEIGHT) * calibration_loss
        + float(cfg.TRAIN.ERC_PQ2_GATE_WEIGHT) * gate_loss
        + float(cfg.TRAIN.ERC_PQ2_ALPHA_WEIGHT) * alpha_loss
        + float(cfg.TRAIN.ERC_PQ2_DIRECTION_WEIGHT) * direction_loss
        + float(cfg.TRAIN.ERC_PQ2_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_PQ2_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_PQ2_PRESERVE_WEIGHT) * preserve_loss
    )

    with torch.no_grad():
        minimum = torch.finfo(predicted_log.dtype).min
        predicted_index = predicted_log.masked_fill(~support, minimum).argmax(-1)
        changed = valid & predicted_index.ne(oracle['base_index'])
        predicted_hard_quality = target.gather(
            1, predicted_index[:, None]).squeeze(1)
        base_hard_quality = target.gather(
            1, oracle['base_index'][:, None]).squeeze(1)
        hard_gain = predicted_hard_quality - base_hard_quality
        beneficial = changed & hard_gain.gt(1e-8)
        harmful = changed & hard_gain.lt(-1e-8)

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_pq2_quality': quality_loss.detach(),
            'Loss/erc_pq2_calibration': calibration_loss.detach(),
            'Loss/erc_pq2_gate': gate_loss.detach(),
            'Loss/erc_pq2_alpha': alpha_loss.detach(),
            'Loss/erc_pq2_direction': direction_loss.detach(),
            'Loss/erc_pq2_decision': decision_loss.detach(),
            'Loss/erc_pq2_noharm': noharm_loss.detach(),
            'Loss/erc_pq2_preserve': preserve_loss.detach(),
            'ERC/pq2_valid_fraction': valid.float().mean(),
            'ERC/pq2_oracle_actionable_rate': mean(actionable.float()),
            'ERC/pq2_oracle_gain': mean(oracle['oracle_gain']),
            'ERC/pq2_gate_target_mean': mean(oracle['gate_target']),
            'ERC/pq2_gate_mean': mean(gate),
            'ERC/pq2_quality_mae': mean(
                (reliability - quality_target).abs().mean(-1)),
            'ERC/pq2_direction_accuracy': mean(
                predicted_alpha.sign().eq(
                    oracle['oracle_alpha'].sign()).float(), actionable),
            'ERC/pq2_alpha_mean': mean(predicted_alpha),
            'ERC/pq2_alpha_abs': mean(predicted_alpha.abs()),
            'ERC/pq2_alpha_std': predicted_alpha[valid].std(unbiased=False),
            'ERC/pq2_selection_change_rate': mean(changed.float()),
            'ERC/pq2_response_gain': mean(hard_gain),
            'ERC/pq2_changed_response_gain': mean(hard_gain, changed),
            'ERC/pq2_beneficial_change_rate': mean(beneficial.float()),
            'ERC/pq2_harmful_change_rate': mean(harmful.float()),
        }
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for dataset_id in ids[valid].unique(sorted=True):
                mask = valid & ids.eq(dataset_id)
                suffix = str(int(dataset_id.item()))
                status['ERC/pq2_domain_' + suffix + '_response_gain'] = mean(
                    hard_gain, mask)
                status['ERC/pq2_domain_' + suffix + '_harmful_rate'] = mean(
                    harmful.float(), mask)
    return total, status
