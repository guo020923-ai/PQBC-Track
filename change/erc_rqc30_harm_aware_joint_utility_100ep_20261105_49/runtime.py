"""Output-only runtime for harm-aware joint utility ERC."""

from copy import deepcopy

import torch

from .model import HarmAwareJointUtilityERC, MIN_DEPLOY_GAIN


def _history(source):
    result = {}
    for key in ("state", "prompt", "max_score", "pred_box_net", "out_dict"):
        if hasattr(source, key):
            value = getattr(source, key)
            result[key] = value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
    return result


def _equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class OutputOnlyHarmAwareRuntime:
    def __init__(self, source, model, provider):
        if type(model) is not HarmAwareJointUtilityERC:
            raise TypeError("exact harm-aware joint utility ERC required")
        self.source, self.model, self.provider = source, model, provider
        self.last = None

    @torch.no_grad()
    def initialize(self, image, info):
        self.last = None
        return self.source.initialize(image, info)

    @torch.no_grad()
    def track(self, image, explore=False):
        if explore:
            raise ValueError("official harm-aware inference cannot explore")
        previous_state = list(self.source.state)
        rqc_result = self.source.track(image, None)
        after_source = _history(self.source)
        batch = self.provider(self.source, image, rqc_result, previous_state)
        prediction = self.model(
            batch["evidence"], batch["executable"], batch["anchor_index"]
        )
        row = torch.arange(len(batch["boxes"]), device=batch["boxes"].device)
        chosen = batch["boxes"][row, prediction["index"]]
        if len(chosen) != 1 or not _equal(after_source, _history(self.source)):
            raise RuntimeError("invalid output-only runtime state")
        index = int(prediction["index"][0])
        box = chosen[0].detach().double().cpu().tolist()
        self.last = {
            "index": index,
            "anchor_index": int(batch["anchor_index"][0]),
            "rqc_box": list(rqc_result["target_bbox"]),
            "output_box": box,
            "predicted_gain": float(prediction["predicted_gain"][0, index]),
            "minimum_deploy_gain": MIN_DEPLOY_GAIN,
            "source_history_feedback": False,
        }
        return dict(rqc_result, target_bbox=box)

