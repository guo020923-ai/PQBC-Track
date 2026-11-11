"""Response-quality supervision for reference-reliability Feature ERC."""

import torch
from torch.nn import functional as F


def _masked_average(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _response_quality(response, relevance, temperature):
    probability = F.softmax(response / max(float(temperature), 1e-6), -1)
    return (probability * relevance).sum(-1)


def compute_feature_reliability_loss_and_metrics(pred, gt_heatmap, cfg):
    packet = pred['erc_feature_output']
    active = packet['active'].bool()
    relevance = gt_heatmap.detach().flatten(1)
    valid = active & torch.isfinite(relevance).all(-1) & (
        relevance.max(-1).values - relevance.min(-1).values > 1e-6
    )
    response_temperature = cfg.TRAIN.ERC_FEATURE_RESPONSE_TEMPERATURE
    visual_quality = _response_quality(
        packet['visual_response'], relevance, response_temperature)
    language_quality = _response_quality(
        packet['language_response'], relevance, response_temperature)
    quality = torch.stack([visual_quality, language_quality], -1).detach()
    target_temperature = max(
        float(cfg.TRAIN.ERC_FEATURE_TARGET_TEMPERATURE), 1e-6)
    target = F.softmax(quality / target_temperature, -1)
    log_probability = F.log_softmax(
        packet['reliability_logits'] / max(
            float(cfg.MODEL.HEAD.ERC.FEATURE_RELIABILITY_TEMPERATURE), 1e-6
        ), -1)
    reliability_frame = F.kl_div(
        log_probability, target, reduction='none').sum(-1)
    reliability_loss = _masked_average(reliability_frame, valid)

    gap = quality[:, 0] - quality[:, 1]
    predicted_gap = (
        packet['reliability_logits'][:, 0]
        - packet['reliability_logits'][:, 1]
    )
    minimum_gap = float(cfg.TRAIN.ERC_FEATURE_MIN_QUALITY_GAP)
    ranked = valid & gap.abs().gt(minimum_gap)
    signed = gap.sign()
    rank_frame = F.softplus(
        float(cfg.TRAIN.ERC_FEATURE_RANK_MARGIN)
        - signed * predicted_gap
    )
    rank_loss = _masked_average(rank_frame, ranked)

    # Small identity regularizer: enough to prevent uncontrolled feature
    # growth, but not a Gate/alpha penalty that can force ERC to zero.
    delta_ratio = packet['raw_feature_delta'].float().norm(dim=-1) / (
        packet['search_token_norm'].detach().float().clamp_min(1e-6)
    )
    delta_loss = _masked_average(
        delta_ratio.mean(-1), active)
    loss = (
        cfg.TRAIN.ERC_FEATURE_RELIABILITY_WEIGHT * reliability_loss
        + cfg.TRAIN.ERC_FEATURE_RANK_WEIGHT * rank_loss
        + cfg.TRAIN.ERC_FEATURE_DELTA_WEIGHT * delta_loss
    )

    with torch.no_grad():
        zero = loss.detach().new_zeros(())

        def mean(value, mask):
            return value[mask].mean() if mask.any() else zero

        predicted = packet['reliability'].argmax(-1)
        oracle = quality.argmax(-1)
        quality_gap = quality.max(-1).values - quality.min(-1).values
        status = {
            'Loss/erc_feature_reliability': reliability_loss.detach(),
            'Loss/erc_feature_rank': rank_loss.detach(),
            'Loss/erc_feature_delta': delta_loss.detach(),
            'Loss/erc_total': loss.detach(),
            'ERC/active_fraction': active.float().mean(),
            'ERC/visual_reliability_mean': mean(
                packet['reliability'][:, 0], active),
            'ERC/language_reliability_mean': mean(
                packet['reliability'][:, 1], active),
            'ERC/visual_reliability_std': (
                packet['reliability'][active, 0].std(unbiased=False)
                if active.any() else zero
            ),
            'ERC/language_reliability_std': (
                packet['reliability'][active, 1].std(unbiased=False)
                if active.any() else zero
            ),
            'ERC/visual_quality_mean': mean(visual_quality, valid),
            'ERC/language_quality_mean': mean(language_quality, valid),
            'ERC/reference_quality_gap': mean(quality_gap, valid),
            'ERC/signed_reference_quality_gap_mean': mean(gap, valid),
            'ERC/signed_reference_quality_gap_std': (
                gap[valid].std(unbiased=False) if valid.any() else zero
            ),
            'ERC/reliability_accuracy': mean(
                predicted.eq(oracle).float(), ranked),
            'ERC/visual_preferred_fraction': mean(
                predicted.eq(0).float(), valid),
            'ERC/language_preferred_fraction': mean(
                predicted.eq(1).float(), valid),
            'ERC/oracle_visual_fraction': mean(
                oracle.eq(0).float(), valid),
            'ERC/oracle_language_fraction': mean(
                oracle.eq(1).float(), valid),
            'ERC/feature_delta_relative_norm': mean(
                delta_ratio.mean(-1), active),
        }
    return loss, status
