"""Transactional quarter-shrink crop feedback for cache generation and deployment."""

from copy import deepcopy
import os

import torch

from .actions import execution_mask
from .config import PROTOCOL


def deterministic_torch():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _clone(value):
    return value.detach().clone() if torch.is_tensor(value) else deepcopy(value)


def _snapshot(source):
    result = {}
    for key in ("state", "prompt", "max_score", "pred_box_net", "out_dict", "context_bbox", "y_patch_arr"):
        result[key] = (hasattr(source, key), _clone(getattr(source, key)) if hasattr(source, key) else None)
    return result


def quarter_shrink_index(raw_index, evidence, executable, anchor):
    rows = torch.arange(len(raw_index), device=raw_index.device)
    relative = evidence[..., -8:-4]
    target = relative[rows, raw_index] * 0.25
    distance = (relative - target[:, None]).square().sum(-1)
    distance = distance.masked_fill(~executable, torch.inf)
    mapped = distance.argmin(-1)
    return torch.where(raw_index.eq(anchor), anchor, mapped)


def feedback_geometry_valid(box, image_shape):
    """Return whether a candidate can safely become the next tracker state."""
    value = torch.as_tensor(box, dtype=torch.double)
    height, width = map(float, image_shape[:2])
    return bool(
        value.shape == (4,)
        and torch.isfinite(value).all()
        and value[2:].gt(0).all()
        and value[:2].ge(0).all()
        and value[0] + value[2] <= width
        and value[1] + value[3] <= height
    )


def commit_feedback(source, box, before, image_shape):
    if not feedback_geometry_valid(box, image_shape):
        raise ValueError("invalid feedback geometry")
    value = torch.as_tensor(box, dtype=torch.double)
    source.state = value.tolist()
    prompt_exists, prompt = before["prompt"]
    if prompt_exists:
        source.prompt = _clone(prompt)
    if hasattr(source, "max_score"):
        source.max_score = 0
    for key in ("pred_box_net", "out_dict"):
        source.__dict__.pop(key, None)
    for key in ("context_bbox", "y_patch_arr"):
        existed, old = before[key]
        if existed:
            setattr(source, key, _clone(old))
        else:
            source.__dict__.pop(key, None)


class ExecutionAlignedPairwiseRuntime:
    def __init__(self, source, model, provider, threshold):
        if not hasattr(model, "choose"):
            raise TypeError("pairwise model required")
        self.source, self.model, self.provider = source, model, provider
        self.threshold = float(threshold); self.last = None; self.commits = 0
        self.cooldown_frames = int(PROTOCOL["temporal_cooldown_frames"])
        self.search_index = 0; self.last_commit_search = -10**9
        self.suppressed_nonkeep_proposals = 0
        self.invalid_feedback_fallbacks = 0

    @torch.no_grad()
    def initialize(self, image, info):
        self.last = None; self.commits = 0
        self.search_index = 0; self.last_commit_search = -10**9
        self.suppressed_nonkeep_proposals = 0
        self.invalid_feedback_fallbacks = 0
        return self.source.initialize(image, info)

    @torch.no_grad()
    def track(self, image, explore=False):
        if explore:
            raise ValueError("feedback inference cannot explore")
        previous = list(self.source.state); before = _snapshot(self.source)
        rqc = self.source.track(image, None)
        packet = self.provider(self.source, image, rqc, previous)
        executable = execution_mask(packet["evidence"], packet["executable"])
        prediction = self.model(packet["evidence"], executable, packet["anchor_index"])
        raw = self.model.choose(
            prediction["utility_score"], executable,
            packet["anchor_index"], self.threshold,
        )
        mapped = raw
        proposed = int(raw[0]) != int(packet["anchor_index"][0])
        suppressed = proposed and (
            self.search_index - self.last_commit_search <= self.cooldown_frames
        )
        if suppressed:
            mapped = packet["anchor_index"].clone()
            self.suppressed_nonkeep_proposals += 1
        rows = torch.arange(len(packet["boxes"]), device=packet["boxes"].device)
        chosen = packet["boxes"][rows, mapped]
        if len(chosen) != 1:
            raise ValueError("batch-one runtime required")
        requested_commit = int(mapped[0]) != int(packet["anchor_index"][0])
        box = chosen[0].detach().double().cpu().tolist()
        invalid_feedback_fallback = requested_commit and not feedback_geometry_valid(
            box, image.shape
        )
        if invalid_feedback_fallback:
            # Invalid candidates cannot define a legal next state. Preserve the
            # source tracker's native KEEP transition instead of crashing or
            # clipping to a different, untrained action.
            mapped = packet["anchor_index"].clone()
            box = list(rqc["target_bbox"])
            self.invalid_feedback_fallbacks += 1
        committed = requested_commit and not invalid_feedback_fallback
        if committed:
            commit_feedback(self.source, box, before, image.shape)
            self.commits += 1
            self.last_commit_search = self.search_index
        self.last = {
            "index": int(mapped[0]), "raw_index": int(raw[0]),
            "anchor_index": int(packet["anchor_index"][0]),
            "rqc_box": list(rqc["target_bbox"]), "output_box": box,
            "feedback_committed": committed, "source_history_feedback": True,
            "execution_aligned_action_basis": True,
            "temporal_cooldown_frames": self.cooldown_frames,
            "nonkeep_proposal_suppressed": bool(suppressed),
            "suppressed_nonkeep_proposals_total": self.suppressed_nonkeep_proposals,
            "feedback_commit_requested": bool(requested_commit),
            "invalid_feedback_fallback": bool(invalid_feedback_fallback),
            "invalid_feedback_fallbacks_total": self.invalid_feedback_fallbacks,
            "invalid_feedback_policy": "post_selection_keep_fallback",
            "prompt_update_discarded": committed, "threshold": self.threshold,
            "packet": packet,
        }
        self.search_index += 1
        return dict(rqc, target_bbox=box)

