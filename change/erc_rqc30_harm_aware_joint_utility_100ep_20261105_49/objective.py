"""Metric-aligned, harm-aware supervision for joint utility prediction."""

import math

import torch
from torch.nn import functional as F


GAIN_STEP = 1.0 / 21.0
LOG_ACTIONS = math.log(25.0)


def _masked_mean(value, mask):
    selected = value.masked_select(mask)
    if selected.numel() == 0:
        return value.sum() * 0.0
    return selected.mean()


def _group_balanced_regression(predicted_gain, utility, valid_nonkeep):
    """Give harmful, neutral, and actionable actions equal group influence."""
    scaled_prediction = predicted_gain / GAIN_STEP
    scaled_target = utility / GAIN_STEP
    element = F.smooth_l1_loss(
        scaled_prediction, scaled_target, reduction="none", beta=1.0
    )
    groups = (
        valid_nonkeep & utility.le(0.0),
        valid_nonkeep & utility.gt(0.0) & utility.lt(GAIN_STEP),
        valid_nonkeep & utility.ge(GAIN_STEP),
    )
    present = [_masked_mean(element, mask) for mask in groups if mask.any()]
    if not present:
        return element.sum() * 0.0
    return torch.stack(present).mean()


def harm_aware_utility_loss(predicted_gain, executable, anchor_index, utility):
    batch, actions = predicted_gain.shape
    if actions != 25 or executable.shape != (batch, 25):
        raise ValueError("action schema mismatch")
    if utility.shape != (batch, 25):
        raise ValueError("utility schema mismatch")
    if not torch.isfinite(predicted_gain).all() or not torch.isfinite(utility).all():
        raise ValueError("non-finite score or utility")
    rows = torch.arange(batch, device=predicted_gain.device)
    if not predicted_gain[rows, anchor_index].eq(0).all():
        raise ValueError("KEEP gain must be exactly zero")

    valid_nonkeep = executable.clone()
    valid_nonkeep[rows, anchor_index] = False
    regression_loss = _group_balanced_regression(
        predicted_gain, utility, valid_nonkeep
    )

    # A soft utility distribution teaches the complete ordering rather than a
    # single hard class.  KEEP is included, so abstention and action identity
    # cannot become contradictory independent predictions.
    predicted_logits = (predicted_gain / GAIN_STEP).masked_fill(~executable, -torch.inf)
    target_logits = (utility / GAIN_STEP).masked_fill(~executable, -torch.inf)
    target_log_probability = target_logits.log_softmax(-1)
    predicted_log_probability = predicted_logits.log_softmax(-1)
    target_probability = target_log_probability.exp()
    # Avoid the undefined 0 * (-inf - -inf) expression at masked actions.
    # These entries have exactly zero target mass and must contribute zero KL.
    safe_target_log = torch.where(
        executable, target_log_probability, torch.zeros_like(target_log_probability)
    )
    safe_predicted_log = torch.where(
        executable, predicted_log_probability, torch.zeros_like(predicted_log_probability)
    )
    distribution_loss = torch.where(
        executable,
        target_probability * (safe_target_log - safe_predicted_log),
        torch.zeros_like(target_probability),
    ).sum(-1).mean() / LOG_ACTIONS

    # Explicitly penalize false positive utility on actions known to damage or
    # tie the RQC anchor.  This is the failure mode seen in both completed
    # factorized tests and is learned exclusively from the sealed TRAIN cache.
    harmful = valid_nonkeep & utility.le(0.0)
    false_positive = (predicted_gain / GAIN_STEP).clamp_min(0).square()
    false_positive_harm_loss = _masked_mean(false_positive, harmful)

    loss = regression_loss + distribution_loss + false_positive_harm_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite harm-aware objective")

    with torch.no_grad():
        safe_utility = utility.masked_fill(~valid_nonkeep, -torch.inf)
        best_gain, best_action = safe_utility.max(-1)
        actionable = valid_nonkeep.any(-1) & best_gain.ge(GAIN_STEP)
        oracle = torch.where(actionable, best_action, anchor_index)

        candidate = predicted_gain.masked_fill(~valid_nonkeep, -torch.inf).argmax(-1)
        candidate_gain = predicted_gain[rows, candidate]
        deployed = valid_nonkeep.any(-1) & candidate_gain.ge(GAIN_STEP)
        chosen = torch.where(deployed, candidate, anchor_index)
        chosen_utility = utility[rows, chosen]
        oracle_utility = utility[rows, oracle]
        nonkeep_count = deployed.double().sum().clamp_min(1.0)
        actionable_count = actionable.double().sum().clamp_min(1.0)
        harmful_deploy = deployed & chosen_utility.le(0.0)
        stats = {
            "loss": loss.detach(),
            "regression_loss": regression_loss.detach(),
            "distribution_loss": distribution_loss.detach(),
            "false_positive_harm_loss": false_positive_harm_loss.detach(),
            "actionable_rate": actionable.double().mean(),
            "predicted_keep_rate": chosen.eq(anchor_index).double().mean(),
            "actionable_recall": (deployed & actionable).double().sum() / actionable_count,
            "harmful_deploy_rate": harmful_deploy.double().sum() / nonkeep_count,
            "mean_deployed_utility": chosen_utility.masked_select(deployed).double().sum() / nonkeep_count,
            "chosen_regret": (oracle_utility - chosen_utility).clamp_min(0).double().mean(),
            "predicted_action24_share_nonkeep": (chosen.eq(24) & deployed).double().sum() / nonkeep_count,
            "oracle_action24_share_actionable": (best_action.eq(24) & actionable).double().sum() / actionable_count,
        }
    return loss, stats

