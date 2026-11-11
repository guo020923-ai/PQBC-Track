"""Response-only supervision for dense reliability calibration (FER-v3)."""

import torch
from torch.nn import functional as F


def _masked_average(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _response_quality(response, relevance, temperature):
    probability = F.softmax(response / max(float(temperature), 1e-6), -1)
    return (probability * relevance).sum(-1)


def compute_dense_response_reliability_loss_and_metrics(pred, gt_heatmap, cfg):
    active = pred['erc_active_mask'].reshape(-1).bool()
    gt = gt_heatmap.detach().flatten(1).clamp_min(0.0)
    valid = active & torch.isfinite(gt).all(-1) & gt.max(-1).values.gt(0.0)
    reference = pred['erc_windowed_score']
    zero = reference.sum() * 0.0
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'Loss/erc_v3_target': zero.detach(),
            'Loss/erc_v3_rank': zero.detach(),
            'Loss/erc_v3_noharm': zero.detach(),
            'Loss/erc_v3_reliability': zero.detach(),
            'Loss/erc_v3_reliability_rank': zero.detach(),
            'Loss/erc_v3_alpha_reg': zero.detach(),
            'ERC/v3_active_fraction': active.float().mean(),
        }

    eps = max(float(cfg.MODEL.HEAD.ERC.EPS), 1e-12)
    target_temperature = max(
        float(cfg.TRAIN.ERC_DENSE_TARGET_TEMPERATURE), 1e-6)
    target_probability = F.softmax(gt / target_temperature, dim=-1)
    # The ERC auxiliary objective trains only ERC.  The original Head is still
    # jointly optimized by UVLTrack's native losses, anchor and distillation;
    # it must not receive a second hidden gradient through this response loss.
    log_base = pred['erc_windowed_base_score'].detach().clamp_min(eps).log()
    log_calibrated = log_base + pred['erc_v3_correction_map'].flatten(1)

    target_loss_frame = -(
        target_probability * F.log_softmax(log_calibrated, dim=-1)
    ).sum(-1)
    target_loss = _masked_average(target_loss_frame, valid)

    target_log_base = (target_probability * log_base).sum(-1)
    target_log_calibrated = (target_probability * log_calibrated).sum(-1)
    peak = gt.max(-1, keepdim=True).values
    negative_mask = gt.lt(0.10 * peak)
    negative_mask = negative_mask & torch.isfinite(log_base)
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
    noharm_loss = _masked_average(
        F.relu(baseline_margin.detach() - calibrated_margin), valid)

    visual_quality = _response_quality(
        pred['erc_v3_visual_response'], gt,
        cfg.TRAIN.ERC_FEATURE_RESPONSE_TEMPERATURE)
    language_quality = _response_quality(
        pred['erc_v3_language_response'], gt,
        cfg.TRAIN.ERC_FEATURE_RESPONSE_TEMPERATURE)
    quality = torch.stack([visual_quality, language_quality], dim=-1).detach()
    reliability_target = F.softmax(
        quality / max(
            float(cfg.TRAIN.ERC_FEATURE_TARGET_TEMPERATURE), 1e-6),
        dim=-1)
    reliability_log_probability = F.log_softmax(
        pred['erc_v3_reliability_logits'] / max(
            float(cfg.MODEL.HEAD.ERC.FEATURE_RELIABILITY_TEMPERATURE), 1e-6),
        dim=-1)
    reliability_frame = F.kl_div(
        reliability_log_probability, reliability_target,
        reduction='none').sum(-1)
    reliability_loss = _masked_average(reliability_frame, valid)

    quality_gap_signed = quality[:, 0] - quality[:, 1]
    predicted_gap = (
        pred['erc_v3_reliability_logits'][:, 0]
        - pred['erc_v3_reliability_logits'][:, 1])
    reliability_ranked = valid & quality_gap_signed.abs().gt(
        float(cfg.TRAIN.ERC_FEATURE_MIN_QUALITY_GAP))
    reliability_rank_frame = F.softplus(
        float(cfg.TRAIN.ERC_FEATURE_RANK_MARGIN)
        - quality_gap_signed.sign() * predicted_gap)
    reliability_rank_loss = _masked_average(
        reliability_rank_frame, reliability_ranked)

    predicted_alpha = pred['erc_v3_predicted_alpha']
    alpha_reg_loss = _masked_average(
        predicted_alpha.abs().flatten(1).mean(-1), active)
    total = (
        float(cfg.TRAIN.ERC_DENSE_TARGET_WEIGHT) * target_loss
        + float(cfg.TRAIN.ERC_DENSE_RANK_WEIGHT) * rank_loss
        + float(cfg.TRAIN.ERC_DENSE_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_DENSE_RELIABILITY_WEIGHT) * reliability_loss
        + float(cfg.TRAIN.ERC_DENSE_RELIABILITY_RANK_WEIGHT)
        * reliability_rank_loss
        + float(cfg.TRAIN.ERC_DENSE_ALPHA_REG_WEIGHT) * alpha_reg_loss
    )

    with torch.no_grad():
        base_index = pred['erc_windowed_base_score'].argmax(-1)
        calibrated_index = pred['erc_windowed_score'].argmax(-1)
        changed = valid & calibrated_index.ne(base_index)
        base_relevance = gt[rows, base_index]
        calibrated_relevance = gt[rows, calibrated_index]
        relevance_gain = calibrated_relevance - base_relevance
        response_gain = calibrated_margin - baseline_margin
        reliability = pred['erc_v3_reliability']
        predicted_branch = reliability.argmax(-1)
        oracle_branch = quality.argmax(-1)
        max_alpha = max(float(cfg.MODEL.HEAD.ERC.MAX_ALPHA), eps)

        def mean(values, mask):
            return values[mask].mean() if mask.any() else zero.detach()

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_v3_target': target_loss.detach(),
            'Loss/erc_v3_rank': rank_loss.detach(),
            'Loss/erc_v3_noharm': noharm_loss.detach(),
            'Loss/erc_v3_reliability': reliability_loss.detach(),
            'Loss/erc_v3_reliability_rank': reliability_rank_loss.detach(),
            'Loss/erc_v3_alpha_reg': alpha_reg_loss.detach(),
            'ERC/v3_active_fraction': active.float().mean(),
            'ERC/v3_alpha_mean': mean(
                predicted_alpha.flatten(1).mean(-1), active),
            'ERC/v3_alpha_abs_mean': mean(
                predicted_alpha.abs().flatten(1).mean(-1), active),
            'ERC/v3_alpha_std': (
                predicted_alpha[active].std(unbiased=False)
                if active.any() else zero.detach()),
            'ERC/v3_alpha_saturation_rate': mean(
                predicted_alpha.abs().flatten(1).gt(
                    0.95 * max_alpha).float().mean(-1), active),
            'ERC/v3_selection_change_rate': mean(changed.float(), valid),
            'ERC/v3_response_margin_gain': mean(response_gain, valid),
            'ERC/v3_changed_response_margin_gain': mean(
                response_gain, changed),
            'ERC/v3_target_relevance_gain': mean(relevance_gain, valid),
            'ERC/v3_changed_target_relevance_gain': mean(
                relevance_gain, changed),
            'ERC/v3_beneficial_change_rate': mean(
                (changed & relevance_gain.gt(0.0)).float(), valid),
            'ERC/v3_harmful_change_rate': mean(
                (changed & relevance_gain.lt(0.0)).float(), valid),
            'ERC/v3_visual_reliability_mean': mean(
                reliability[:, 0], valid),
            'ERC/v3_language_reliability_mean': mean(
                reliability[:, 1], valid),
            'ERC/v3_reliability_accuracy': mean(
                predicted_branch.eq(oracle_branch).float(),
                reliability_ranked),
            'ERC/v3_reference_quality_gap': mean(
                quality_gap_signed.abs(), valid),
        }
    return total, status
