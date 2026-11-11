import json
from types import SimpleNamespace

import numpy as np
import torch

from .actions import execution_mask, quarter_mapping
from .config import PROTOCOL, calibration_mask, validate_protocol
from .model import ExecutionAlignedPairwiseERC
from .objective import pairwise_utility_loss
from .runtime import ExecutionAlignedPairwiseRuntime, feedback_geometry_valid


class FakeSource:
    def __init__(self):
        self.state = [10.0, 10.0, 20.0, 20.0]
        self.prompt = torch.tensor([1.0])
        self.max_score = 1

    def initialize(self, image, info):
        self.state = list(info["init_bbox"])
        return {"target_bbox": list(self.state)}

    def track(self, image, _):
        return {"target_bbox": list(self.state)}


class AlwaysActionOne:
    def __call__(self, evidence, executable, anchor):
        return {"utility_score": torch.zeros_like(executable, dtype=torch.float32)}

    @staticmethod
    def choose(score, executable, anchor, threshold):
        result = anchor.clone()
        result[:] = 1
        return result


def provider(source, image, rqc, previous):
    evidence = torch.zeros(1, 25, 48)
    evidence[0, :, 40] = torch.arange(25, dtype=torch.float32)
    executable = torch.ones(1, 25, dtype=torch.bool)
    boxes = torch.tensor([[[10.0 + i, 10.0, 20.0, 20.0] for i in range(25)]])
    return {
        "evidence": evidence,
        "executable": executable,
        "anchor_index": torch.zeros(1, dtype=torch.long),
        "boxes": boxes,
    }


def invalid_provider(source, image, rqc, previous):
    packet = provider(source, image, rqc, previous)
    packet["boxes"][0, 1] = torch.tensor([-1.0, 10.0, 20.0, 20.0])
    return packet


def main():
    validate_protocol()
    assert PROTOCOL["temporal_cooldown_frames"] == 8
    assert not torch.cuda.is_initialized()
    model = ExecutionAlignedPairwiseERC()
    evidence = torch.randn(8, 25, 48)
    source_executable = torch.ones(8, 25, dtype=torch.bool)
    executable = execution_mask(evidence, source_executable)
    anchor = torch.zeros(8, dtype=torch.long)
    utility = torch.randn(8, 25) * 0.05
    utility[:, 0] = 0
    prediction = model(evidence, executable, anchor)
    loss, _ = pairwise_utility_loss(prediction, executable, anchor, utility)
    loss.backward()
    assert torch.isfinite(loss)
    relative = torch.zeros(2, 25, 48)
    relative[0, :, -8:-4] = torch.arange(25).float()[:, None]
    mapped = quarter_mapping(relative, source_executable[:2])
    direct = execution_mask(relative, source_executable[:2])
    assert int(mapped[0, 4]) == 1 and bool(direct[0, 1]) and bool(direct[:, 0].all())

    source = FakeSource()
    runtime = ExecutionAlignedPairwiseRuntime(source, AlwaysActionOne(), provider, 0.0)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    runtime.initialize(image, {"init_bbox": [10.0, 10.0, 20.0, 20.0]})
    indices, suppressed = [], []
    for _ in range(10):
        runtime.track(image, explore=False)
        indices.append(runtime.last["index"])
        suppressed.append(runtime.last["nonkeep_proposal_suppressed"])
    assert indices == [1] + [0] * 8 + [1]
    assert suppressed == [False] + [True] * 8 + [False]
    assert runtime.commits == 2 and runtime.suppressed_nonkeep_proposals == 8

    assert feedback_geometry_valid([0.0, 0.0, 100.0, 100.0], image.shape)
    assert not feedback_geometry_valid([-1.0, 0.0, 20.0, 20.0], image.shape)
    assert not feedback_geometry_valid([0.0, 0.0, 101.0, 20.0], image.shape)
    assert not feedback_geometry_valid([0.0, 0.0, float("nan"), 20.0], image.shape)

    guarded_source = FakeSource()
    guarded = ExecutionAlignedPairwiseRuntime(
        guarded_source, AlwaysActionOne(), invalid_provider, 0.0
    )
    guarded.initialize(image, {"init_bbox": [10.0, 10.0, 20.0, 20.0]})
    guarded_output = guarded.track(image, explore=False)
    assert guarded_output["target_bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert guarded_source.state == [10.0, 10.0, 20.0, 20.0]
    assert guarded.last["raw_index"] == 1 and guarded.last["index"] == 0
    assert guarded.last["feedback_commit_requested"]
    assert guarded.last["invalid_feedback_fallback"]
    assert guarded.last["invalid_feedback_fallbacks_total"] == 1
    assert not guarded.last["feedback_committed"] and guarded.commits == 0

    held = calibration_mask(np.arange(30000))
    print(json.dumps({
        "status": "ROUND124_POST_SELECTION_GEOMETRY_FALLBACK_CPU_PASS",
        "training_rows": int((~held).sum()),
        "calibration_rows": int(held.sum()),
        "direct_execution_basis_verified": True,
        "cooldown_frames": 8,
        "cooldown_schedule_verified": indices,
        "suppressed_proposals_verified": 8,
        "invalid_feedback_keep_fallback_verified": True,
        "gpu_operations": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
