"""Output-only runtime using one ordinal score for selection and gating."""

from copy import deepcopy
import json
from pathlib import Path

import torch

from .model import OrdinalRiskUtilityERC


def _history(source):
    snapshot = {}
    for key in ("state", "prompt", "max_score", "pred_box_net", "out_dict"):
        if hasattr(source, key):
            value = getattr(source, key)
            snapshot[key] = value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
    return snapshot


def _equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_equal(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_equal(x, y) for x, y in zip(left, right))
        )
    return left == right


class OutputOnlyOrdinalRiskRuntime:
    """Preserve native RQC state and modify only the emitted box."""

    def __init__(self, source, model, provider, calibration_path):
        if type(model) is not OrdinalRiskUtilityERC:
            raise TypeError("exact ordinal-risk model required")
        calibration = json.loads(Path(calibration_path).read_text())
        if (
            calibration.get("status") != "TRAIN_ORDINAL_CALIBRATION_VERIFIED"
            or calibration.get("test_data_read") is not False
        ):
            raise ValueError("verified TRAIN-only ordinal calibration required")
        self.threshold = float(calibration["threshold"])
        self.source = source
        self.model = model
        self.provider = provider
        self.last = None

    @torch.no_grad()
    def initialize(self, image, info):
        self.last = None
        return self.source.initialize(image, info)

    @torch.no_grad()
    def track(self, image, explore=False):
        if explore:
            raise ValueError("official inference cannot explore")
        previous = list(self.source.state)
        rqc = self.source.track(image, None)
        after = _history(self.source)
        batch = self.provider(self.source, image, rqc, previous)
        prediction = self.model(
            batch["evidence"], batch["executable"], batch["anchor_index"]
        )
        index = self.model.choose(
            prediction["risk_score"],
            batch["executable"],
            batch["anchor_index"],
            self.threshold,
        )
        rows = torch.arange(len(batch["boxes"]), device=batch["boxes"].device)
        chosen = batch["boxes"][rows, index]
        if len(chosen) != 1 or not _equal(after, _history(self.source)):
            raise RuntimeError("invalid output-only state")
        action = int(index[0])
        probability = prediction["ordinal_probability"][0, action]
        box = chosen[0].detach().double().cpu().tolist()
        self.last = {
            "index": action,
            "anchor_index": int(batch["anchor_index"][0]),
            "rqc_box": list(rqc["target_bbox"]),
            "output_box": box,
            "risk_score": float(prediction["risk_score"][0, action]),
            "harmful_probability": float(probability[0]),
            "marginal_probability": float(probability[1]),
            "actionable_probability": float(probability[2]),
            "risk_threshold": self.threshold,
            "source_history_feedback": False,
        }
        return dict(rqc, target_bbox=box)
