"""Bidirectional response-oracle supervision for ERC-AV5.

The oracle searches the exact signed global/regional correction family used
at inference.  It uses only the GT response heatmap, never candidate boxes or
IoU, preserving the boundary between ERC (evidence calibration) and RQC
(candidate quality estimation).
"""

import torch
from torch.nn import functional as F


def _masked_mean(values, mask, zero):
    if mask.any():
        return values[mask].float().mean()
    return zero


def _response_margin(log_score, target, support, temperature, core_threshold):
    """Target evidence minus soft hardest-distractor evidence."""
    target = target / target.amax(-1, keepdim=True).clamp_min(1e-6)
    core = target.ge(core_threshold) & support
    # Degenerate small targets still receive supervision at their maximum.
    fallback = target.eq(target.amax(-1, keepdim=True)) & support
    core = torch.where(core.any(-1, keepdim=True), core, fallback)
    weights = target * core.to(target.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    positive = (log_score * weights).sum(-1)

    distractor = support & ~core
    minimum = torch.finfo(log_score.dtype).min
    scaled = (log_score / temperature).masked_fill(~distractor, minimum)
    count = distractor.sum(-1).clamp_min(1).to(log_score.dtype)
    negative = temperature * (
        torch.logsumexp(scaled, dim=-1) - count.log())
    negative = torch.where(
        distractor.any(-1), negative, torch.zeros_like(negative))
    return positive - negative


def _domain_robust_mean(values, valid, dataset_ids, worst_weight, zero):
    """Average domains equally and mix in the current worst domain."""
    if not valid.any():
        return zero
    if dataset_ids is None:
        return values[valid].float().mean()
    dataset_ids = dataset_ids.reshape(-1).to(device=values.device)
    if dataset_ids.numel() != values.numel():
        raise ValueError('ERC-AV5 dataset_id batch shape mismatch')
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


@torch.no_grad()
def _build_oracle(pred, gt_heatmap, cfg, num_search_frames):
    support = pred['erc_cd5_support_mask'].detach().flatten(1).bool()
    dtype = pred['erc_windowed_base_score'].dtype
    target = gt_heatmap.detach().flatten(1).to(dtype=dtype).clamp_min(0.0)
    active = pred['erc_active_mask'].detach().reshape(-1).bool()
    valid = active & support.any(-1) & target.amax(-1).gt(0.0)
    eps = max(float(cfg.MODEL.HEAD.ERC.EPS), 1e-12)
    temperature = max(float(cfg.TRAIN.ERC_CD5_ORACLE_TEMPERATURE), 1e-6)
    core_threshold = float(cfg.TRAIN.ERC_CD5_TARGET_CORE_THRESHOLD)

    log_base = pred['erc_windowed_base_score'].detach().flatten(1).clamp_min(
        eps).log()
    log_cls = pred['erc_windowed_cls_score'].detach().flatten(1).clamp_min(
        eps).log()
    log_cont = pred['erc_windowed_cont_score'].detach().flatten(1).clamp_min(
        eps).log()
    base_quality = _response_margin(
        log_base, target, support, temperature, core_threshold)
    cls_quality = _response_margin(
        log_cls, target, support, temperature, core_threshold)
    cont_quality = _response_margin(
        log_cont, target, support, temperature, core_threshold)
    reliability_temperature = max(
        float(cfg.TRAIN.ERC_CD5_RELIABILITY_TEMPERATURE), 1e-6)
    reliability_target = F.softmax(
        torch.stack([cls_quality, cont_quality], dim=-1)
        / reliability_temperature, dim=-1)

    values = [float(value) for value in cfg.TRAIN.ERC_CD5_ORACLE_ALPHA_GRID]
    if 0.0 not in values or not any(v < 0.0 for v in values) or not any(
            v > 0.0 for v in values):
        raise ValueError('ERC_CD5_ORACLE_ALPHA_GRID needs zero and both signs')
    maximum = float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
    if max(abs(v) for v in values) > maximum + 1e-8:
        raise ValueError('ERC-AV5 oracle action exceeds MAX_ALPHA')
    actions = [(global_alpha, regional_alpha)
               for global_alpha in values for regional_alpha in values]
    difference = pred['erc_cd5_difference'].detach().flatten(1)
    regional_mask = pred['erc_cd5_regional_mask'].detach().flatten(1)
    correction_maps = []
    action_coefficients = []
    for global_alpha, regional_alpha in actions:
        coefficient = (
            (1.0 - regional_mask) * global_alpha
            + regional_mask * regional_alpha)
        correction_maps.append(coefficient * difference)
        action_coefficients.append((global_alpha, regional_alpha))
    corrections = torch.stack(correction_maps, dim=1)
    coefficients = torch.tensor(
        action_coefficients, device=log_base.device, dtype=log_base.dtype)
    candidate_log_score = log_base[:, None] + corrections
    action_count = len(actions)
    candidate_quality = _response_margin(
        candidate_log_score.reshape(-1, candidate_log_score.shape[-1]),
        target[:, None].expand(-1, action_count, -1).reshape(
            -1, target.shape[-1]),
        support[:, None].expand(-1, action_count, -1).reshape(
            -1, support.shape[-1]),
        temperature, core_threshold,
    ).reshape(log_base.shape[0], action_count)
    action_cost = float(cfg.TRAIN.ERC_CD5_ACTION_COST) * (
        coefficients.abs().mean(-1) / max(maximum, 1e-6))
    utility = candidate_quality - base_quality[:, None] - action_cost[None]
    keep_index = actions.index((0.0, 0.0))
    utility[:, keep_index] = 0.0
    best_utility, best_action = utility.max(-1)
    rows = torch.arange(log_base.shape[0], device=log_base.device)
    oracle_quality = candidate_quality[rows, best_action]
    oracle_gain = oracle_quality - base_quality
    oracle_coefficients = coefficients[best_action]
    minimum_gain = float(cfg.TRAIN.ERC_CD5_MIN_RESPONSE_GAIN)
    gain_scale = max(float(cfg.TRAIN.ERC_CD5_GATE_GAIN_SCALE), 1e-6)
    gate_target = ((oracle_gain - minimum_gain) / gain_scale).clamp(0.0, 1.0)
    gate_target = torch.where(valid, gate_target, torch.zeros_like(gate_target))

    raw_controls = torch.cat([
        gate_target[:, None], oracle_coefficients / max(maximum, 1e-6)
    ], dim=-1)
    temporal_controls = raw_controls.clone()
    if int(num_search_frames) == 2 and raw_controls.shape[0] % 2 == 0:
        half = raw_controls.shape[0] // 2
        pair_valid = valid[:half] & valid[half:]
        smoothed_second = (
            float(cfg.MODEL.HEAD.ERC.AV2_TEMPORAL_EMA) * raw_controls[:half]
            + (1.0 - float(cfg.MODEL.HEAD.ERC.AV2_TEMPORAL_EMA))
            * raw_controls[half:]
        )
        temporal_controls[half:] = torch.where(
            pair_valid[:, None], smoothed_second, raw_controls[half:])

    return {
        'valid': valid,
        'target': target,
        'support': support,
        'log_base': log_base,
        'base_quality': base_quality,
        'cls_quality': cls_quality,
        'cont_quality': cont_quality,
        'reliability_target': reliability_target,
        'raw_controls': raw_controls,
        'temporal_controls': temporal_controls,
        'oracle_quality': oracle_quality,
        'oracle_gain': oracle_gain,
        'oracle_coefficients': oracle_coefficients,
        'best_action': best_action,
        'keep_index': keep_index,
    }


def compute_cross_domain_continuous_oracle_loss_and_metrics(
        pred, gt_heatmap, cfg, num_search_frames=1, dataset_ids=None):
    controls = pred['erc_cd5_temporal_controls']
    zero = controls.sum() * 0.0
    oracle = _build_oracle(pred, gt_heatmap, cfg, num_search_frames)
    valid = oracle['valid']
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/cd5_valid_fraction': valid.float().mean(),
        }

    target_controls = oracle['temporal_controls']
    reliability_element = F.kl_div(
        pred['erc_cd5_reliability'].clamp_min(1e-8).log(),
        oracle['reliability_target'], reduction='none').sum(-1)
    gate_element = F.binary_cross_entropy(
        controls[:, 0].clamp(1e-6, 1.0 - 1e-6),
        target_controls[:, 0], reduction='none')
    alpha_element = F.smooth_l1_loss(
        controls[:, 1:], target_controls[:, 1:], reduction='none').mean(-1)
    # Give useful non-KEEP actions full alpha supervision while retaining a
    # small identity anchor for non-actionable frames.
    alpha_weight = 0.10 + 0.90 * target_controls[:, 0].detach()
    alpha_element = alpha_element * alpha_weight

    eps = max(float(cfg.MODEL.HEAD.ERC.EPS), 1e-12)
    temperature = max(float(cfg.TRAIN.ERC_CD5_ORACLE_TEMPERATURE), 1e-6)
    core_threshold = float(cfg.TRAIN.ERC_CD5_TARGET_CORE_THRESHOLD)
    predicted_log_score = (
        oracle['log_base'] + pred['erc_cd5_raw_correction'].flatten(1))
    predicted_quality = _response_margin(
        predicted_log_score, oracle['target'], oracle['support'],
        temperature, core_threshold)
    decision_element = F.relu(oracle['oracle_quality'] - predicted_quality)
    noharm_element = F.relu(oracle['base_quality'] - predicted_quality)
    noharm_element = noharm_element * (
        1.0 + float(cfg.TRAIN.ERC_CD5_PRESERVE_BOOST)
        * (1.0 - target_controls[:, 0].detach()))

    if int(num_search_frames) == 2 and controls.shape[0] % 2 == 0:
        half = controls.shape[0] // 2
        pair_valid = valid[:half] & valid[half:]
        temporal_element = F.smooth_l1_loss(
            controls[half:] - controls[:half],
            target_controls[half:] - target_controls[:half],
            reduction='none').mean(-1)
        if dataset_ids is not None:
            temporal_ids = dataset_ids.reshape(-1)[half:]
        else:
            temporal_ids = None
    else:
        pair_valid = valid.new_zeros((0,))
        temporal_element = controls.new_zeros((0,))
        temporal_ids = None

    worst_weight = float(cfg.TRAIN.ERC_CD5_DOMAIN_WORST_WEIGHT)
    reduce = lambda values, mask: _domain_robust_mean(
        values, mask, dataset_ids, worst_weight, zero)
    reliability_loss = reduce(reliability_element, valid)
    gate_loss = reduce(gate_element, valid)
    alpha_loss = reduce(alpha_element, valid)
    decision_loss = reduce(decision_element, valid)
    noharm_loss = reduce(noharm_element, valid)
    temporal_loss = _domain_robust_mean(
        temporal_element, pair_valid, temporal_ids, worst_weight, zero)

    total = (
        float(cfg.TRAIN.ERC_CD5_RELIABILITY_WEIGHT) * reliability_loss
        + float(cfg.TRAIN.ERC_CD5_GATE_WEIGHT) * gate_loss
        + float(cfg.TRAIN.ERC_CD5_ALPHA_WEIGHT) * alpha_loss
        + float(cfg.TRAIN.ERC_CD5_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_CD5_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_CD5_TEMPORAL_WEIGHT) * temporal_loss
    )

    with torch.no_grad():
        base_index = pred['erc_windowed_base_score'].detach().argmax(-1)
        full_index = pred['erc_windowed_score'].detach().argmax(-1)
        changed = valid & base_index.ne(full_index)
        response_gain = predicted_quality - oracle['base_quality']
        beneficial = changed & response_gain.gt(1e-8)
        harmful = changed & response_gain.lt(-1e-8)
        actionable = valid & oracle['raw_controls'][:, 0].gt(0.0)
        predicted_sign = torch.sign(
            pred['erc_cd5_effective_alpha_global'])
        oracle_sign = torch.sign(oracle['oracle_coefficients'][:, 0])

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_cd5_reliability': reliability_loss.detach(),
            'Loss/erc_cd5_gate': gate_loss.detach(),
            'Loss/erc_cd5_alpha': alpha_loss.detach(),
            'Loss/erc_cd5_decision': decision_loss.detach(),
            'Loss/erc_cd5_noharm': noharm_loss.detach(),
            'Loss/erc_cd5_temporal': temporal_loss.detach(),
            'ERC/cd5_valid_fraction': valid.float().mean(),
            'ERC/cd5_oracle_actionable_rate': mean(actionable.float()),
            'ERC/cd5_gate_target_mean': mean(target_controls[:, 0]),
            'ERC/cd5_gate_mean': mean(controls[:, 0]),
            'ERC/cd5_gate_std': controls[valid, 0].std(unbiased=False),
            'ERC/cd5_reliability_cls_mean': mean(
                pred['erc_cd5_reliability'][:, 0]),
            'ERC/cd5_reliability_cont_mean': mean(
                pred['erc_cd5_reliability'][:, 1]),
            'ERC/cd5_alpha_global_mean': mean(
                pred['erc_cd5_effective_alpha_global']),
            'ERC/cd5_alpha_global_abs': mean(
                pred['erc_cd5_effective_alpha_global'].abs()),
            'ERC/cd5_alpha_regional_mean': mean(
                pred['erc_cd5_effective_alpha_regional']),
            'ERC/cd5_alpha_regional_abs': mean(
                pred['erc_cd5_effective_alpha_regional'].abs()),
            'ERC/cd5_oracle_global_positive_rate': mean(
                oracle['oracle_coefficients'][:, 0].gt(0).float(), actionable),
            'ERC/cd5_oracle_global_negative_rate': mean(
                oracle['oracle_coefficients'][:, 0].lt(0).float(), actionable),
            'ERC/cd5_oracle_regional_positive_rate': mean(
                oracle['oracle_coefficients'][:, 1].gt(0).float(), actionable),
            'ERC/cd5_oracle_regional_negative_rate': mean(
                oracle['oracle_coefficients'][:, 1].lt(0).float(), actionable),
            'ERC/cd5_global_direction_accuracy': mean(
                predicted_sign.eq(oracle_sign).float(), actionable),
            'ERC/cd5_selection_change_rate': mean(changed.float()),
            'ERC/cd5_response_gain': mean(response_gain),
            'ERC/cd5_changed_response_gain': mean(response_gain, changed),
            'ERC/cd5_beneficial_change_rate': mean(beneficial.float()),
            'ERC/cd5_harmful_change_rate': mean(harmful.float()),
            'ERC/cd5_oracle_response_gain': mean(oracle['oracle_gain']),
            'ERC/cd5_memory_used_rate': mean(
                pred['erc_cd5_memory_used'].float()),
        }
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for dataset_id in ids[valid].unique(sorted=True):
                mask = valid & ids.eq(dataset_id)
                suffix = str(int(dataset_id.item()))
                status['ERC/cd5_domain_' + suffix + '_response_gain'] = mean(
                    response_gain, mask)
                status['ERC/cd5_domain_' + suffix + '_harmful_rate'] = mean(
                    harmful.float(), mask)
    return total, status
