import json
from types import SimpleNamespace

import numpy as np
import torch

from .config import calibration_mask, validate_protocol
from .model import FeedbackPairwiseERC
from .objective import pairwise_utility_loss
from .runtime import _snapshot, commit_feedback, quarter_shrink_index


def main():
    validate_protocol()
    assert not torch.cuda.is_initialized()
    model = FeedbackPairwiseERC()
    evidence = torch.randn(8, 25, 48); executable = torch.ones(8, 25, dtype=torch.bool)
    anchor = torch.zeros(8, dtype=torch.long); utility = torch.randn(8, 25) * 0.05; utility[:, 0] = 0
    prediction = model(evidence, executable, anchor)
    loss, _ = pairwise_utility_loss(prediction, executable, anchor, utility)
    loss.backward(); assert torch.isfinite(loss)
    relative = torch.zeros(2, 25, 48)
    relative[0, :, -8:-4] = torch.arange(25).float()[:, None]
    raw = torch.tensor([4, 0]); mapped = quarter_shrink_index(raw, relative, executable[:2], anchor[:2])
    assert mapped.tolist() == [1, 0]
    source = SimpleNamespace(
        state=[1.0, 2.0, 3.0, 4.0], prompt=torch.tensor([3.0]), max_score=7,
        pred_box_net=torch.tensor([1.0]), out_dict={"x": 1},
        context_bbox=torch.tensor([5.0]), y_patch_arr=[6],
    )
    before = _snapshot(source)
    commit_feedback(source, [10.0, 11.0, 12.0, 13.0], before, (100, 100, 3))
    assert source.state == [10.0, 11.0, 12.0, 13.0]
    assert torch.equal(source.prompt, before["prompt"][1]) and source.max_score == 0
    assert not hasattr(source, "pred_box_net") and not hasattr(source, "out_dict")
    assert torch.equal(source.context_bbox, before["context_bbox"][1]) and source.y_patch_arr == [6]
    held = calibration_mask(np.arange(30000))
    print(json.dumps({
        "status": "FEEDBACK_PAIRWISE_CPU_PASS", "training_rows": int((~held).sum()),
        "calibration_rows": int(held.sum()), "transaction_rollback_verified": True,
        "quarter_mapping_verified": True, "gpu_operations": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

