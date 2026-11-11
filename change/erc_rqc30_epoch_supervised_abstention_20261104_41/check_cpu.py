"""CPU-only admission for model, objective, and fixed protocol."""

import json
import torch

from .config import PROTOCOL, validate_protocol
from .model import RQC30EpochERC
from .objective import MIN_ACTIONABLE_GAIN, action_utility, structured_regret_loss


def main():
    validate_protocol()
    torch.manual_seed(20261104)
    model = RQC30EpochERC()
    evidence = torch.randn(8, 25, 48)
    executable = torch.ones(8, 25, dtype=torch.bool)
    anchor = torch.zeros(8, dtype=torch.long)
    initial = model(evidence, executable, anchor)
    assert initial["relative_score"][:, 0].eq(0).all()
    assert initial["index"].eq(0).all()
    boxes = torch.rand(8, 25, 4)
    boxes[..., 2:] = boxes[..., 2:] * 0.4 + 0.05
    boxes[:, 0] = torch.tensor([0.2, 0.2, 0.3, 0.3])
    truth = torch.rand(8, 4)
    truth[:, 2:] = truth[:, 2:] * 0.4 + 0.05
    utility = action_utility(boxes, truth).float()
    loss, stats = structured_regret_loss(
        initial["relative_score"], executable, anchor, utility
    )
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    relative = torch.zeros(2, 25)
    relative[0, 3] = 0.1
    relative[1, 4] = -0.1
    choice = model.strict_positive_choice(relative, executable[:2], anchor[:2])
    assert choice.tolist() == [3, 0]
    gate_utility = torch.zeros(2, 25)
    gate_utility[0, 1] = MIN_ACTIONABLE_GAIN / 2
    gate_utility[1, 1] = MIN_ACTIONABLE_GAIN + 1e-4
    _, gate_stats = structured_regret_loss(
        torch.zeros(2, 25), executable[:2], anchor[:2], gate_utility
    )
    assert float(gate_stats["actionable_rate"]) == 0.5
    assert float(gate_stats["label_keep_rate"]) == 0.5
    report = {
        "status": "CPU_ADMISSION_PASS",
        "protocol": PROTOCOL,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "zero_initial_exact_RQC": True,
        "strict_positive_fallback": True,
        "minimum_actionable_gain": MIN_ACTIONABLE_GAIN,
        "metric_aligned_abstention_gate": True,
        "complete_action_structured_regret": True,
        "finite_nonzero_gradient": True,
        "optimizer_steps": 0,
        "gpu_operations": 0,
    }
    print("RQC_STYLE_ERC_CPU_PASS=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
