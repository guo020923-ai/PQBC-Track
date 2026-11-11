"""Risk-sensitive response supervision for trajectory-safe dense ERC DR2.

The objective deliberately avoids bbox geometry and IoU.  It replaces DR1's
exact-centre, +/-MAX_ALPHA target with three response-level principles:

1. a high-response *safe region* rather than one mandatory grid cell;
2. the smallest correction that makes the safest reachable response win;
3. continuous abstention when the response advantage is weak or unstable
   across the two search views sampled from the same sequence.
"""

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


def _temporal_robust_gate_target(
        local_target, valid, num_search_frames, risk_weight, consensus_mix):
    frame_count = int(num_search_frames)
    if frame_count < 2 or local_target.shape[0] % frame_count != 0:
        return local_target, None, None
    batch = local_target.shape[0] // frame_count
    target_view = local_target.reshape(frame_count, batch)
    valid_view = valid.reshape(frame_count, batch)
    pair_valid = valid_view.all(0)
    robust = (
        target_view.mean(0)
        - max(float(risk_weight), 0.0)
        * target_view.std(0, unbiased=False)
    ).clamp(0.0, 1.0)
    mix = min(max(float(consensus_mix), 0.0), 1.0)
    blended = (
        (1.0 - mix) * target_view
        + mix * robust.unsqueeze(0)
    )
    target_view = torch.where(
        pair_valid.unsqueeze(0), blended, target_view)
    return target_view.reshape(-1), pair_valid, robust


def _causal_deploy_gate(
        current_gate, active, num_search_frames, temporal_ema,
        enable_temporal_memory):
    """Simulate the causal inference gate on ordered training views.

    Training batches contain two search frames from the same sequence.  DR2
    must optimize the gate that is actually deployed: an EMA permission with
    an unconditional current-frame veto.  This short causal rollout prevents
    validation from reporting an optimistic current-gate gain that disappears
    once temporal safety is enabled at test time.
    """
    frame_count = int(num_search_frames)
    if (not bool(enable_temporal_memory) or frame_count < 2
            or current_gate.shape[0] % frame_count != 0):
        return current_gate
    batch = current_gate.shape[0] // frame_count
    gates = current_gate.reshape(frame_count, batch)
    active_view = active.reshape(frame_count, batch)
    ema = min(max(float(temporal_ema), 0.0), 1.0 - 1e-6)
    temporal = None
    effective = []
    for frame_index in range(frame_count):
        gate = gates[frame_index]
        if temporal is None:
            temporal = (1.0 - ema) * gate
        else:
            temporal = ema * temporal + (1.0 - ema) * gate
        deployed = torch.minimum(gate, temporal)
        deployed = torch.where(
            active_view[frame_index], deployed, torch.zeros_like(deployed))
        effective.append(deployed)
    return torch.stack(effective, dim=0).reshape(-1)


def compute_target_reference_trajectory_safe_dense_loss_and_metrics(
        pred, gt_heatmap, cfg, num_search_frames=1, dataset_ids=None):
    raw_correction = pred['erc_dr2_raw_correction']
    gate = pred['erc_dr2_current_gate'].reshape(-1)
    active = pred['erc_active_mask'].reshape(-1).bool()
    deploy_gate = _causal_deploy_gate(
        gate, active, num_search_frames,
        cfg.MODEL.HEAD.ERC.DR2_TEMPORAL_EMA,
        cfg.MODEL.HEAD.ERC.DR2_ENABLE_MEMORY,
    )
    correction = raw_correction * deploy_gate[:, None, None]
    zero = correction.sum() * 0.0
    support = pred['erc_dr1_support_mask'].bool()
    target = gt_heatmap.detach().to(correction).clamp_min(0.0)
    target = target / target.flatten(1).amax(-1).clamp_min(1e-6)[:, None, None]
    target_flat = target.flatten(1)
    batch = target.shape[0]
    rows = torch.arange(batch, device=target.device)

    tiny = torch.finfo(correction.dtype).tiny
    base_log = pred['erc_windowed_base_score'].detach().clamp_min(tiny).log()
    minimum = torch.finfo(base_log.dtype).min
    base_log = base_log.masked_fill(~support.flatten(1), minimum)
    baseline_index = base_log.argmax(-1)
    baseline_quality = target_flat[rows, baseline_index]

    safe_threshold = float(cfg.TRAIN.ERC_DR2_SAFE_TARGET_THRESHOLD)
    safe_mask = target_flat.ge(safe_threshold) & support.flatten(1)
    safe_score = base_log.masked_fill(~safe_mask, minimum)
    safe_index = safe_score.argmax(-1)
    safe_quality = target_flat[rows, safe_index]
    required_span = base_log[rows, baseline_index] - base_log[rows, safe_index]
    valid = active & target_flat.amax(-1).gt(0.0) & safe_mask.any(-1)
    baseline_safe = valid & baseline_quality.ge(safe_threshold)
    response_advantage = (safe_quality - baseline_quality).clamp_min(0.0)
    max_pair_span = 2.0 * float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
    soft_reachability = (
        1.0 - required_span.clamp_min(0.0) / max(max_pair_span, 1e-6)
    ).clamp(0.0, 1.0)
    need = (
        (safe_threshold - baseline_quality).clamp_min(0.0)
        / max(safe_threshold, 1e-6)
    ).clamp(0.0, 1.0)
    actionable = (
        valid & ~baseline_safe
        & response_advantage.gt(float(cfg.TRAIN.ERC_DR2_MIN_ADVANTAGE))
        & required_span.lt(max_pair_span)
    )
    local_gate_target = torch.where(
        actionable, need * soft_reachability,
        torch.zeros_like(need),
    )
    gate_target, pair_valid, robust_target = _temporal_robust_gate_target(
        local_gate_target, valid, num_search_frames,
        cfg.TRAIN.ERC_DR2_TEMPORAL_RISK_WEIGHT,
        cfg.TRAIN.ERC_DR2_TEMPORAL_CONSENSUS_MIX,
    )
    gate_target = gate_target.detach()

    predicted_log = base_log + correction.flatten(1)
    predicted_probability = torch.softmax(predicted_log, dim=-1)
    base_probability = torch.softmax(base_log, dim=-1)
    predicted_index = predicted_log.argmax(-1)

    # Minimum pairwise displacement: rank the best baseline-supported point in
    # the safe target region over the current baseline peak, without pushing
    # either endpoint to +/-MAX_ALPHA.
    predicted_safe = predicted_log[rows, safe_index]
    predicted_baseline = predicted_log[rows, baseline_index]
    rank_element = F.relu(
        float(cfg.TRAIN.ERC_DR2_RANK_MARGIN)
        - predicted_safe + predicted_baseline)

    advantage_map = target_flat - baseline_quality[:, None]
    expected_advantage = (predicted_probability * advantage_map).sum(-1)
    base_expected_advantage = (base_probability * advantage_map).sum(-1)
    desired_advantage = gate_target * response_advantage
    utility_element = F.relu(
        desired_advantage
        - (expected_advantage - base_expected_advantage))

    negative_advantage = (-advantage_map).clamp_min(0.0)
    predicted_harm = (predicted_probability * negative_advantage).sum(-1)
    baseline_harm = (base_probability * negative_advantage).sum(-1)
    noharm_element = (predicted_harm - baseline_harm).clamp_min(0.0)

    log_predicted_probability = F.log_softmax(predicted_log, dim=-1)
    preserve_kl = F.kl_div(
        log_predicted_probability, base_probability,
        reduction='none').sum(-1)
    preserve_element = (
        preserve_kl + correction.abs().flatten(1).mean(-1))
    minimal_element = correction.abs().flatten(1).mean(-1)
    gate_element = F.smooth_l1_loss(
        gate, gate_target, reduction='none')

    worst = float(cfg.TRAIN.ERC_DR2_DOMAIN_WORST_WEIGHT)
    reduce = lambda values, mask: _domain_mean(
        values, mask, dataset_ids, worst, zero)
    rank_loss = reduce(rank_element * gate_target, actionable)
    utility_loss = reduce(utility_element, valid)
    noharm_loss = reduce(noharm_element, valid)
    preserve_loss = reduce(preserve_element, valid & ~actionable)
    minimal_loss = reduce(minimal_element, actionable)
    gate_loss = reduce(gate_element, valid)

    total = (
        float(cfg.TRAIN.ERC_DR2_GATE_WEIGHT) * gate_loss
        + float(cfg.TRAIN.ERC_DR2_UTILITY_WEIGHT) * utility_loss
        + float(cfg.TRAIN.ERC_DR2_RANK_WEIGHT) * rank_loss
        + float(cfg.TRAIN.ERC_DR2_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_DR2_PRESERVE_WEIGHT) * preserve_loss
        + float(cfg.TRAIN.ERC_DR2_MINIMAL_WEIGHT) * minimal_loss
    )

    with torch.no_grad():
        changed = valid & predicted_index.ne(baseline_index)
        predicted_quality = target_flat[rows, predicted_index]
        hard_gain = predicted_quality - baseline_quality
        beneficial = changed & hard_gain.gt(1e-8)
        harmful = changed & hard_gain.lt(-1e-8)
        valid_count = valid.float().sum().clamp_min(1.0)
        changed_count = changed.float().sum().clamp_min(1.0)
        actionable_count = actionable.float().sum().clamp_min(1.0)
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_dr2_gate': gate_loss.detach(),
            'Loss/erc_dr2_utility': utility_loss.detach(),
            'Loss/erc_dr2_rank': rank_loss.detach(),
            'Loss/erc_dr2_noharm': noharm_loss.detach(),
            'Loss/erc_dr2_preserve': preserve_loss.detach(),
            'Loss/erc_dr2_minimal': minimal_loss.detach(),
            'ERC/dr2_valid_fraction': valid.float().mean(),
            'ERC/dr2_baseline_safe_rate': baseline_safe.float().sum() / valid_count,
            'ERC/dr2_actionable_rate': actionable.float().sum() / valid_count,
            'ERC/dr2_gate_target_mean': (
                gate_target[valid].mean() if valid.any() else zero.detach()),
            'ERC/dr2_gate_mean': (
                gate[valid].mean() if valid.any() else zero.detach()),
            'ERC/dr2_deploy_gate_mean': (
                deploy_gate[valid].mean()
                if valid.any() else zero.detach()),
            'ERC/dr2_temporal_veto_mean': (
                (gate[valid] - deploy_gate[valid]).clamp_min(0.0).mean()
                if valid.any() else zero.detach()),
            'ERC/dr2_gate_actionable_mean': (
                gate[actionable].mean() if actionable.any() else zero.detach()),
            'ERC/dr2_gate_preserve_mean': (
                gate[valid & ~actionable].mean()
                if (valid & ~actionable).any() else zero.detach()),
            'ERC/dr2_required_span_mean': (
                required_span[actionable].mean()
                if actionable.any() else zero.detach()),
            'ERC/dr2_correction_abs': (
                correction[valid].abs().mean()
                if valid.any() else zero.detach()),
            'ERC/dr2_raw_correction_abs': (
                raw_correction[valid].abs().mean()
                if valid.any() else zero.detach()),
            'ERC/dr2_correction_saturation': (
                correction[valid].abs().gt(
                    0.95 * float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
                ).float().mean() if valid.any() else zero.detach()),
            'ERC/dr2_selection_change_rate': changed.float().sum() / valid_count,
            'ERC/dr2_response_gain_per_valid': (
                (hard_gain * valid).sum() / valid_count),
            'ERC/dr2_changed_gain': (
                (hard_gain * changed).sum() / changed_count),
            'ERC/dr2_beneficial_change_rate': (
                beneficial.float().sum() / valid_count),
            'ERC/dr2_harmful_change_rate': (
                harmful.float().sum() / valid_count),
            'ERC/dr2_actionable_hit_rate': (
                (predicted_index.eq(safe_index) & actionable).float().sum()
                / actionable_count),
            'ERC/dr2_proposal_distance': (
                pred['erc_dr2_proposal_distance'][valid].mean()
                if valid.any() else zero.detach()),
        }
        if pair_valid is not None:
            status['ERC/dr2_temporal_pair_valid_rate'] = pair_valid.float().mean()
            status['ERC/dr2_temporal_robust_target_mean'] = (
                robust_target[pair_valid].mean()
                if pair_valid.any() else zero.detach())
        else:
            status['ERC/dr2_temporal_pair_valid_rate'] = zero.detach()
            status['ERC/dr2_temporal_robust_target_mean'] = zero.detach()
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for dataset_id in ids[valid].unique(sorted=True):
                mask = valid & ids.eq(dataset_id)
                count = mask.float().sum().clamp_min(1.0)
                suffix = str(int(dataset_id.item()))
                status['ERC/dr2_domain_' + suffix + '_response_gain'] = (
                    (hard_gain * mask).sum() / count)
                status['ERC/dr2_domain_' + suffix + '_harmful_rate'] = (
                    harmful[mask].float().mean())
                status['ERC/dr2_domain_' + suffix + '_gate_mean'] = (
                    gate[mask].mean())
    return total, status
