"""Response-level supervision for cross-domain global-regional ERC.

The oracle uses only the official GT center heatmap and the reachable ERC
response family.  Bbox geometry and candidate IoU are intentionally excluded
so candidate-quality modelling remains the responsibility of RQC.
"""

import torch
from torch.nn import functional as F


def _masked_mean(values, mask, zero):
    if mask.any():
        return values[mask].float().mean()
    return zero


def _response_quality(log_score, target, support, temperature):
    minimum = torch.finfo(log_score.dtype).min
    probability = F.softmax(
        log_score.masked_fill(~support, minimum) / temperature, dim=-1)
    target = target / target.amax(-1, keepdim=True).clamp_min(1e-6)
    return (probability * target).sum(-1)


@torch.no_grad()
def _build_oracle(pred, gt_heatmap, cfg, num_search_frames):
    support = pred['erc_cd1_support_mask'].detach().flatten(1).bool()
    response_dtype = pred['erc_windowed_base_score'].dtype
    target = (
        gt_heatmap.detach().flatten(1).to(dtype=response_dtype).clamp_min(0.0)
    )
    active = pred['erc_active_mask'].detach().reshape(-1).bool()
    valid = active & support.any(-1) & target.amax(-1).gt(0.0)
    eps = float(cfg.MODEL.HEAD.ERC.EPS)
    temperature = max(float(cfg.TRAIN.ERC_CD1_ORACLE_TEMPERATURE), 1e-6)

    cls_log = pred['erc_windowed_cls_score'].detach().flatten(1).clamp_min(
        eps).log()
    cont_log = pred['erc_windowed_cont_score'].detach().flatten(1).clamp_min(
        eps).log()
    cls_quality = _response_quality(
        cls_log, target, support, temperature)
    cont_quality = _response_quality(
        cont_log, target, support, temperature)
    relative_temperature = max(
        float(cfg.TRAIN.ERC_CD1_RELIABILITY_TEMPERATURE), 1e-6)
    reliability_target = F.softmax(torch.stack([
        cls_quality.clamp_min(eps).log(),
        cont_quality.clamp_min(eps).log(),
    ], dim=-1) / relative_temperature, dim=-1)
    direction_target = torch.tanh(
        float(cfg.MODEL.HEAD.ERC.DIRECTION_SCALE)
        * (reliability_target[:, 0] - reliability_target[:, 1]))
    direction_sign = direction_target.sign()

    log_base = pred['erc_windowed_base_score'].detach().flatten(1).clamp_min(
        eps).log()
    difference = pred['erc_cd1_difference'].detach().flatten(1)
    regional_mask = pred['erc_cd1_regional_mask'].detach().flatten(1)
    magnitudes = [float(v) for v in cfg.TRAIN.ERC_CD1_ORACLE_MAGNITUDES]
    regional_mixes = [
        float(v) for v in cfg.TRAIN.ERC_CD1_ORACLE_REGIONAL_MIXES]
    actions = [(0.0, 0.0)]
    actions.extend(
        (magnitude, mix)
        for magnitude in magnitudes if magnitude > 0.0
        for mix in regional_mixes
    )
    correction_maps = []
    signed_alphas = []
    action_mixes = []
    for magnitude, mix in actions:
        signed = direction_sign * magnitude
        scale = signed[:, None] * (
            (1.0 - mix) + mix * regional_mask)
        correction_maps.append(scale * difference)
        signed_alphas.append(signed)
        action_mixes.append(torch.full_like(signed, mix))
    corrections = torch.stack(correction_maps, dim=1)
    signed_alphas = torch.stack(signed_alphas, dim=1)
    action_mixes = torch.stack(action_mixes, dim=1)
    candidate_log_score = log_base[:, None] + corrections
    candidate_quality = _response_quality(
        candidate_log_score.reshape(-1, candidate_log_score.shape[-1]),
        target[:, None].expand_as(candidate_log_score).reshape(
            -1, target.shape[-1]),
        support[:, None].expand_as(candidate_log_score).reshape(
            -1, support.shape[-1]),
        temperature,
    ).reshape(log_base.shape[0], len(actions))
    base_quality = candidate_quality[:, 0]
    magnitude_cost = signed_alphas.abs() / max(
        float(cfg.MODEL.HEAD.ERC.MAX_ALPHA), 1e-6)
    utility = candidate_quality - base_quality[:, None] - (
        float(cfg.TRAIN.ERC_CD1_ACTION_COST) * magnitude_cost)
    utility[:, 0] = 0.0
    best_utility, best_action = utility.max(-1)
    rows = torch.arange(log_base.shape[0], device=log_base.device)
    oracle_quality = candidate_quality[rows, best_action]
    oracle_alpha = signed_alphas[rows, best_action]
    oracle_regional_mix = action_mixes[rows, best_action]
    oracle_gain = oracle_quality - base_quality
    minimum_gain = float(cfg.TRAIN.ERC_CD1_MIN_RESPONSE_GAIN)
    gain_scale = max(float(cfg.TRAIN.ERC_CD1_GATE_GAIN_SCALE), 1e-6)
    gate_target = ((oracle_gain - minimum_gain) / gain_scale).clamp(0.0, 1.0)
    gate_target = torch.where(valid, gate_target, torch.zeros_like(gate_target))
    strength_target = (
        oracle_alpha.abs() / max(float(cfg.MODEL.HEAD.ERC.MAX_ALPHA), 1e-6)
    ).clamp(0.0, 1.0)

    raw_controls = torch.cat([
        reliability_target,
        gate_target[:, None],
        strength_target[:, None],
        oracle_regional_mix[:, None],
    ], dim=-1)
    temporal_controls = raw_controls.clone()
    frame_count = int(num_search_frames)
    if frame_count == 2 and raw_controls.shape[0] % 2 == 0:
        half = raw_controls.shape[0] // 2
        pair_valid = valid[:half] & valid[half:]
        smoothed_second = (
            float(cfg.MODEL.HEAD.ERC.AV2_TEMPORAL_EMA)
            * raw_controls[:half]
            + (1.0 - float(cfg.MODEL.HEAD.ERC.AV2_TEMPORAL_EMA))
            * raw_controls[half:]
        )
        temporal_controls[half:] = torch.where(
            pair_valid[:, None], smoothed_second, raw_controls[half:])

    temporal_direction = torch.tanh(
        float(cfg.MODEL.HEAD.ERC.DIRECTION_SCALE)
        * (temporal_controls[:, 0] - temporal_controls[:, 1]))
    temporal_alpha = (
        float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
        * temporal_controls[:, 2]
        * temporal_controls[:, 3]
        * temporal_direction
    )
    return {
        'valid': valid,
        'target': target,
        'support': support,
        'log_base': log_base,
        'base_quality': base_quality,
        'cls_quality': cls_quality,
        'cont_quality': cont_quality,
        'reliability_target': reliability_target,
        'direction_target': direction_target,
        'gate_target': gate_target,
        'strength_target': strength_target,
        'regional_mix_target': oracle_regional_mix,
        'temporal_controls': temporal_controls,
        'temporal_alpha': temporal_alpha,
        'oracle_quality': oracle_quality,
        'oracle_gain': oracle_gain,
        'oracle_alpha': oracle_alpha,
        'oracle_regional_mix': oracle_regional_mix,
        'best_action': best_action,
    }


def compute_cross_domain_reliability_loss_and_metrics(
        pred, gt_heatmap, cfg, num_search_frames=1):
    predicted_controls = pred['erc_cd1_temporal_controls']
    zero = predicted_controls.sum() * 0.0
    oracle = _build_oracle(pred, gt_heatmap, cfg, num_search_frames)
    valid = oracle['valid']
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/cd1_valid_fraction': valid.float().mean(),
        }

    target_controls = oracle['temporal_controls']
    reliability_loss = _masked_mean(
        F.binary_cross_entropy(
            predicted_controls[:, :2], target_controls[:, :2],
            reduction='none').mean(-1), valid, zero)
    gate_loss = _masked_mean(
        F.binary_cross_entropy(
            predicted_controls[:, 2], target_controls[:, 2],
            reduction='none'), valid, zero)
    strength_loss = _masked_mean(
        F.smooth_l1_loss(
            predicted_controls[:, 3], target_controls[:, 3],
            reduction='none'), valid, zero)
    regional_loss = _masked_mean(
        F.smooth_l1_loss(
            predicted_controls[:, 4], target_controls[:, 4],
            reduction='none'), valid, zero)
    control_loss = 0.5 * (strength_loss + regional_loss)

    predicted_alpha = pred['erc_cd1_alpha_total']
    alpha_loss = _masked_mean(
        F.smooth_l1_loss(
            predicted_alpha, oracle['temporal_alpha'], reduction='none'),
        valid, zero)

    eps = float(cfg.MODEL.HEAD.ERC.EPS)
    temperature = max(float(cfg.TRAIN.ERC_CD1_ORACLE_TEMPERATURE), 1e-6)
    predicted_log_score = (
        oracle['log_base']
        + pred['erc_cd1_raw_correction'].flatten(1)
    )
    predicted_quality = _response_quality(
        predicted_log_score, oracle['target'], oracle['support'], temperature)
    decision_loss = _masked_mean(
        F.relu(oracle['oracle_quality'] - predicted_quality),
        valid, zero)
    noharm_loss = _masked_mean(
        F.relu(oracle['base_quality'] - predicted_quality),
        valid, zero)

    if int(num_search_frames) == 2 and predicted_controls.shape[0] % 2 == 0:
        half = predicted_controls.shape[0] // 2
        pair_valid = valid[:half] & valid[half:]
        predicted_change = predicted_controls[half:] - predicted_controls[:half]
        target_change = target_controls[half:] - target_controls[:half]
        temporal_loss = _masked_mean(
            F.smooth_l1_loss(
                predicted_change, target_change, reduction='none').mean(-1),
            pair_valid, zero)
    else:
        temporal_loss = zero

    total = (
        float(cfg.TRAIN.ERC_CD1_RELIABILITY_WEIGHT) * reliability_loss
        + float(cfg.TRAIN.ERC_CD1_GATE_WEIGHT) * gate_loss
        + float(cfg.TRAIN.ERC_CD1_CONTROL_WEIGHT) * control_loss
        + float(cfg.TRAIN.ERC_CD1_ALPHA_WEIGHT) * alpha_loss
        + float(cfg.TRAIN.ERC_CD1_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_CD1_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_CD1_TEMPORAL_WEIGHT) * temporal_loss
    )

    with torch.no_grad():
        base_index = pred['erc_windowed_base_score'].detach().argmax(-1)
        full_index = pred['erc_windowed_score'].detach().argmax(-1)
        changed = valid & base_index.ne(full_index)
        response_gain = predicted_quality - oracle['base_quality']
        beneficial = changed & response_gain.gt(1e-8)
        harmful = changed & response_gain.lt(-1e-8)
        actionable = valid & oracle['gate_target'].gt(0.0)
        predicted_direction = pred['erc_cd1_direction'].sign()
        target_direction = oracle['direction_target'].sign()

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_cd1_reliability': reliability_loss.detach(),
            'Loss/erc_cd1_gate': gate_loss.detach(),
            'Loss/erc_cd1_control': control_loss.detach(),
            'Loss/erc_cd1_alpha': alpha_loss.detach(),
            'Loss/erc_cd1_decision': decision_loss.detach(),
            'Loss/erc_cd1_noharm': noharm_loss.detach(),
            'Loss/erc_cd1_temporal': temporal_loss.detach(),
            'ERC/cd1_valid_fraction': valid.float().mean(),
            'ERC/cd1_oracle_actionable_rate': mean(actionable.float()),
            'ERC/cd1_gate_target_mean': mean(oracle['gate_target']),
            'ERC/cd1_gate_mean': mean(predicted_controls[:, 2]),
            'ERC/cd1_reliability_cls_mean': mean(predicted_controls[:, 0]),
            'ERC/cd1_reliability_cont_mean': mean(predicted_controls[:, 1]),
            'ERC/cd1_direction_accuracy': mean(
                predicted_direction.eq(target_direction).float(), actionable),
            'ERC/cd1_alpha_abs': mean(predicted_alpha.abs()),
            'ERC/cd1_alpha_mean': mean(predicted_alpha),
            'ERC/cd1_regional_mix_mean': mean(predicted_controls[:, 4]),
            'ERC/cd1_selection_change_rate': mean(changed.float()),
            'ERC/cd1_response_gain': mean(response_gain),
            'ERC/cd1_changed_response_gain': mean(response_gain, changed),
            'ERC/cd1_beneficial_change_rate': mean(beneficial.float()),
            'ERC/cd1_harmful_change_rate': mean(harmful.float()),
            'ERC/cd1_oracle_response_gain': mean(oracle['oracle_gain']),
            'ERC/cd1_memory_used_rate': mean(
                pred['erc_cd1_memory_used'].float()),
        }
    return total, status
