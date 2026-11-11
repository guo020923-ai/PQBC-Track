"""Action-aligned response supervision for FER-v4.

The C/T reliability target is defined at the exact action point used by
deployment.  A single hard negative is selected by the native Hann*C*T score,
then the target-versus-negative gap is measured directly on the same clipped
``D = center(log(C)-log(T))`` correction basis used at inference.  With the
current spatial strength detached, its sign is the exact beneficial C/T
direction for the deployed action.  No candidate boxes or IoU targets are
used here.
"""

import torch
from torch.nn import functional as F


def _masked_average(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_branch_aligned_dense_response_loss_and_metrics(
    pred, gt_heatmap, cfg,
):
    active = pred['erc_active_mask'].reshape(-1).bool()
    gt = gt_heatmap.detach().flatten(1).clamp_min(0.0)
    valid = active & torch.isfinite(gt).all(-1) & gt.max(-1).values.gt(0.0)
    reference = pred['erc_windowed_score']
    zero = reference.sum() * 0.0
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'Loss/erc_v4_target': zero.detach(),
            'Loss/erc_v4_rank': zero.detach(),
            'Loss/erc_v4_noharm': zero.detach(),
            'Loss/erc_v4_hard_noharm': zero.detach(),
            'Loss/erc_v4_reliability': zero.detach(),
            'Loss/erc_v4_reliability_rank': zero.detach(),
            'Loss/erc_v4_alpha_reg': zero.detach(),
            'Loss/erc_v4_spatial_tv': zero.detach(),
            'ERC/v4_active_fraction': active.float().mean(),
        }

    eps = max(float(cfg.MODEL.HEAD.ERC.EPS), 1e-12)
    target_temperature = max(
        float(cfg.TRAIN.ERC_DENSE_TARGET_TEMPERATURE), 1e-6)
    target_probability = F.softmax(gt / target_temperature, dim=-1)

    # ERC auxiliary gradients must remain inside ERC.  The original Head is
    # optimized only by UVLTrack's native objectives, anchoring and response
    # distillation.
    log_base = pred['erc_windowed_base_score'].detach().clamp_min(eps).log()
    log_calibrated = log_base + pred['erc_v4_correction_map'].flatten(1)

    target_loss_frame = -(
        target_probability * F.log_softmax(log_calibrated, dim=-1)
    ).sum(-1)
    target_loss = _masked_average(target_loss_frame, valid)

    target_log_base = (target_probability * log_base).sum(-1)
    target_log_calibrated = (
        target_probability * log_calibrated).sum(-1)
    peak = gt.max(-1, keepdim=True).values
    negative_mask = gt.lt(0.10 * peak) & torch.isfinite(log_base)
    hard_negative_index = log_base.masked_fill(
        ~negative_mask, torch.finfo(log_base.dtype).min
    ).argmax(-1)
    rows = torch.arange(log_base.shape[0], device=log_base.device)
    negative_log_base = log_base[rows, hard_negative_index]
    negative_log_calibrated = log_calibrated[rows, hard_negative_index]
    baseline_margin = target_log_base - negative_log_base
    calibrated_margin = target_log_calibrated - negative_log_calibrated
    rank_loss_frame = F.softplus(
        float(cfg.TRAIN.ERC_DENSE_RANK_MARGIN) - calibrated_margin)
    rank_loss = _masked_average(rank_loss_frame, valid)
    margin_noharm_frame = F.relu(
        baseline_margin.detach() - calibrated_margin)
    margin_noharm_loss = _masked_average(margin_noharm_frame, valid)

    # Use the exact centered/clipped D that deployment multiplies by alpha,
    # rather than reconstructing it from unclipped C/T logs.  Spatial strength
    # is detached only for the target: with g fixed, effective_gap is the exact
    # derivative of the target-vs-baseline-negative margin with respect to the
    # signed C/T reliability direction.  Positive means "trust C more" and
    # negative means "trust T more".
    correction_basis = pred['erc_v4_correction_basis'].detach().flatten(1)
    target_basis = (target_probability * correction_basis).sum(-1)
    negative_basis = correction_basis[rows, hard_negative_index]
    global_branch_gap = target_basis - negative_basis
    spatial_strength_detached = (
        pred['erc_v4_spatial_strength'].detach().flatten(1))
    effective_basis = spatial_strength_detached * correction_basis
    effective_branch_gap = (
        (target_probability * effective_basis).sum(-1)
        - effective_basis[rows, hard_negative_index])
    branch_quality = torch.stack([
        0.5 * effective_branch_gap,
        -0.5 * effective_branch_gap,
    ], dim=-1).detach()
    reliability_target = F.softmax(
        branch_quality / max(
            float(cfg.TRAIN.ERC_FEATURE_TARGET_TEMPERATURE), 1e-6),
        dim=-1)
    reliability_log_probability = F.log_softmax(
        pred['erc_v4_reliability_logits'] / max(
            float(cfg.MODEL.HEAD.ERC.FEATURE_RELIABILITY_TEMPERATURE), 1e-6),
        dim=-1)
    reliability_frame = F.kl_div(
        reliability_log_probability, reliability_target,
        reduction='none').sum(-1)
    reliability_loss = _masked_average(reliability_frame, valid)

    branch_gap = effective_branch_gap.detach()
    predicted_gap = (
        pred['erc_v4_reliability_logits'][:, 0]
        - pred['erc_v4_reliability_logits'][:, 1])
    reliability_ranked = valid & branch_gap.abs().gt(
        float(cfg.TRAIN.ERC_FEATURE_MIN_QUALITY_GAP))
    reliability_rank_frame = F.softplus(
        float(cfg.TRAIN.ERC_FEATURE_RANK_MARGIN)
        - branch_gap.sign() * predicted_gap)
    reliability_rank_loss = _masked_average(
        reliability_rank_frame, reliability_ranked)

    # A hard decision mask is detached, but the pairwise preservation penalty
    # is differentiable.  It directly repairs cases where ERC promotes a
    # lower-relevance location over the native winner, which the old averaged
    # response-margin no-harm term could miss.
    with torch.no_grad():
        base_index = log_base.argmax(-1)
        calibrated_index = log_calibrated.argmax(-1)
        changed = valid & calibrated_index.ne(base_index)
        base_relevance = gt[rows, base_index]
        calibrated_relevance = gt[rows, calibrated_index]
        relevance_gain = calibrated_relevance - base_relevance
        harmful = changed & relevance_gain.lt(
            -float(cfg.TRAIN.ERC_FEATURE_MIN_QUALITY_GAP))
    hard_noharm_frame = F.relu(
        log_calibrated[rows, calibrated_index]
        - log_calibrated[rows, base_index]
        + float(cfg.TRAIN.ERC_DENSE_RANK_MARGIN))
    hard_noharm_loss = _masked_average(hard_noharm_frame, harmful)
    noharm_loss = margin_noharm_loss + hard_noharm_loss

    predicted_alpha = pred['erc_v4_predicted_alpha']
    alpha_reg_loss = _masked_average(
        predicted_alpha.abs().flatten(1).mean(-1), active)
    spatial_strength = pred['erc_v4_spatial_strength']
    spatial_tv_frame = (
        (spatial_strength[:, 1:] - spatial_strength[:, :-1]).abs().mean(
            dim=(-2, -1))
        + (spatial_strength[:, :, 1:] - spatial_strength[:, :, :-1]).abs().mean(
            dim=(-2, -1)))
    spatial_tv_loss = _masked_average(spatial_tv_frame, active)

    total = (
        float(cfg.TRAIN.ERC_DENSE_TARGET_WEIGHT) * target_loss
        + float(cfg.TRAIN.ERC_DENSE_RANK_WEIGHT) * rank_loss
        + float(cfg.TRAIN.ERC_DENSE_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_DENSE_RELIABILITY_WEIGHT) * reliability_loss
        + float(cfg.TRAIN.ERC_DENSE_RELIABILITY_RANK_WEIGHT)
        * reliability_rank_loss
        + float(cfg.TRAIN.ERC_DENSE_ALPHA_REG_WEIGHT) * alpha_reg_loss
        + float(cfg.TRAIN.ERC_DENSE_SPATIAL_TV_WEIGHT) * spatial_tv_loss
    )

    with torch.no_grad():
        response_gain = calibrated_margin - baseline_margin
        reliability = pred['erc_v4_reliability']
        predicted_branch = reliability.argmax(-1)
        target_branch = branch_quality.argmax(-1)
        direction = pred['erc_v4_reliability_direction'].reshape(-1)
        max_alpha = max(float(cfg.MODEL.HEAD.ERC.MAX_ALPHA), eps)
        alpha_flat = predicted_alpha.flatten(1)
        sign_violation = (
            alpha_flat * direction[:, None]).lt(-1e-8)

        def mean(values, mask):
            return values[mask].mean() if mask.any() else zero.detach()

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_v4_target': target_loss.detach(),
            'Loss/erc_v4_rank': rank_loss.detach(),
            'Loss/erc_v4_noharm': noharm_loss.detach(),
            'Loss/erc_v4_margin_noharm': margin_noharm_loss.detach(),
            'Loss/erc_v4_hard_noharm': hard_noharm_loss.detach(),
            'Loss/erc_v4_reliability': reliability_loss.detach(),
            'Loss/erc_v4_reliability_rank': reliability_rank_loss.detach(),
            'Loss/erc_v4_alpha_reg': alpha_reg_loss.detach(),
            'Loss/erc_v4_spatial_tv': spatial_tv_loss.detach(),
            'ERC/v4_active_fraction': active.float().mean(),
            'ERC/v4_alpha_mean': mean(alpha_flat.mean(-1), active),
            'ERC/v4_alpha_abs_mean': mean(
                alpha_flat.abs().mean(-1), active),
            'ERC/v4_alpha_std': (
                predicted_alpha[active].std(unbiased=False)
                if active.any() else zero.detach()),
            'ERC/v4_alpha_saturation_rate': mean(
                alpha_flat.abs().gt(0.95 * max_alpha).float().mean(-1), active),
            'ERC/v4_alpha_positive_rate': mean(
                alpha_flat.gt(1e-8).float().mean(-1), active),
            'ERC/v4_alpha_negative_rate': mean(
                alpha_flat.lt(-1e-8).float().mean(-1), active),
            'ERC/v4_alpha_sign_violation_rate': mean(
                sign_violation.float().mean(-1), active),
            'ERC/v4_spatial_strength_mean': mean(
                spatial_strength.flatten(1).mean(-1), active),
            'ERC/v4_spatial_strength_std': (
                spatial_strength[active].std(unbiased=False)
                if active.any() else zero.detach()),
            'ERC/v4_selection_change_rate': mean(changed.float(), valid),
            'ERC/v4_response_margin_gain': mean(response_gain, valid),
            'ERC/v4_changed_response_margin_gain': mean(
                response_gain, changed),
            'ERC/v4_target_relevance_gain': mean(relevance_gain, valid),
            'ERC/v4_changed_target_relevance_gain': mean(
                relevance_gain, changed),
            'ERC/v4_beneficial_change_rate': mean(
                (changed & relevance_gain.gt(0.0)).float(), valid),
            'ERC/v4_harmful_change_rate': mean(
                (changed & relevance_gain.lt(0.0)).float(), valid),
            'ERC/v4_branch_cls_utility_mean': mean(
                branch_quality[:, 0], valid),
            'ERC/v4_branch_cont_utility_mean': mean(
                branch_quality[:, 1], valid),
            'ERC/v4_global_branch_gap_mean': mean(
                global_branch_gap, valid),
            'ERC/v4_effective_branch_gap_mean': mean(branch_gap, valid),
            'ERC/v4_effective_branch_gap_abs_mean': mean(
                branch_gap.abs(), valid),
            'ERC/v4_global_effective_direction_agreement': mean(
                global_branch_gap.sign().eq(branch_gap.sign()).float(),
                valid & global_branch_gap.abs().gt(
                    float(cfg.TRAIN.ERC_FEATURE_MIN_QUALITY_GAP))
                & branch_gap.abs().gt(
                    float(cfg.TRAIN.ERC_FEATURE_MIN_QUALITY_GAP))),
            'ERC/v4_cls_reliability_mean': mean(reliability[:, 0], valid),
            'ERC/v4_cont_reliability_mean': mean(reliability[:, 1], valid),
            'ERC/v4_cls_target_mean': mean(reliability_target[:, 0], valid),
            'ERC/v4_cont_target_mean': mean(reliability_target[:, 1], valid),
            'ERC/v4_reliability_accuracy': mean(
                predicted_branch.eq(target_branch).float(), reliability_ranked),
            'ERC/v4_direction_accuracy': mean(
                direction.sign().eq(branch_gap.sign()).float(),
                reliability_ranked),
        }
    return total, status
