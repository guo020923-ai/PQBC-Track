"""Output-only inference matching the fixed-RQC cache training distribution."""

from copy import deepcopy

import torch

from .model import RQC30EpochERC


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


class OutputOnlyEpochRuntime:
    def __init__(self, source, model, provider):
        if type(model) is not RQC30EpochERC:
            raise TypeError("exact epoch-supervised ERC required")
        self.source, self.model, self.provider = source, model, provider
        self.last = None

    @torch.no_grad()
    def initialize(self, image, info):
        self.last = None
        return self.source.initialize(image, info)

    @torch.no_grad()
    def track(self, image, explore=False):
        if explore:
            raise ValueError("official epoch-supervised inference cannot explore")
        previous_state = list(self.source.state)
        rqc_result = self.source.track(image, None)
        after_source = _history(self.source)
        batch = self.provider(self.source, image, rqc_result, previous_state)
        if set(batch) != {"boxes", "evidence", "executable", "anchor_index"}:
            raise ValueError("exact provider schema required")
        prediction = self.model(
            batch["evidence"], batch["executable"], batch["anchor_index"]
        )
        row = torch.arange(len(batch["boxes"]), device=batch["boxes"].device)
        chosen = batch["boxes"][row, prediction["index"]]
        if len(chosen) != 1:
            raise ValueError("official runtime is batch-one")
        if not _equal(after_source, _history(self.source)):
            raise RuntimeError("ERC/provider modified frozen RQC history")
        box = chosen[0].detach().double().cpu().tolist()
        self.last = {
            "index": int(prediction["index"][0]),
            "anchor_index": int(batch["anchor_index"][0]),
            "rqc_box": list(rqc_result["target_bbox"]),
            "output_box": box,
            "source_history_feedback": False,
        }
        return dict(rqc_result, target_bbox=box)
