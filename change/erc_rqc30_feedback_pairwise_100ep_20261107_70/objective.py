import torch
from torch.nn import functional as F

from .config import PROTOCOL


def pairwise_utility_loss(prediction, executable, anchor, utility):
    score = prediction["utility_score"]
    batch, actions = score.shape
    rows = torch.arange(batch, device=score.device)
    valid = executable.clone(); valid[rows, anchor] = True
    delta_u = utility[:, :, None] - utility[:, None, :]
    delta_s = score[:, :, None] - score[:, None, :]
    pairs = valid[:, :, None] & valid[:, None, :]
    pairs &= ~torch.eye(actions, dtype=torch.bool, device=score.device)[None]
    target = torch.sigmoid(delta_u / PROTOCOL["pair_temperature"])
    weight = (delta_u.abs() / PROTOCOL["pair_temperature"]).clamp(0.25, 4.0)
    element = F.binary_cross_entropy_with_logits(delta_s, target, reduction="none") * weight
    loss = element[pairs].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("pairwise loss")
    with torch.no_grad():
        alternatives = valid.clone(); alternatives[rows, anchor] = False
        candidate = score.masked_fill(~alternatives, -torch.inf).argmax(-1)
        active = alternatives.any(-1) & score[rows, candidate].gt(0)
        chosen = torch.where(active, candidate, anchor)
        chosen_utility = utility[rows, chosen]
        oracle = utility.masked_fill(~alternatives, -torch.inf).argmax(-1)
        oracle_utility = utility[rows, oracle].clamp_min(0)
        active_count = active.double().sum().clamp_min(1)
        stats = {
            "loss": loss.detach(),
            "predicted_keep_rate": chosen.eq(anchor).double().mean(),
            "harmful_deploy_rate": ((chosen_utility <= 0) & active).double().sum() / active_count,
            "mean_deployed_utility": chosen_utility.masked_select(active).double().sum() / active_count,
            "chosen_regret": (oracle_utility - chosen_utility).clamp_min(0).double().mean(),
            "pair_accuracy": ((delta_s.sign() == delta_u.sign()) & pairs).double().sum() / pairs.double().sum().clamp_min(1),
            "mean_score": score.masked_select(alternatives).double().mean(),
        }
    return loss, stats

