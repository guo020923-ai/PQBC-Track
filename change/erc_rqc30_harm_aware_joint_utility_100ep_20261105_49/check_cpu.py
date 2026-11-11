"""CPU-only contracts; no training, checkpoint selection, test, or CUDA use."""

import json
from pathlib import Path
import tempfile

import torch

from .config import PROTOCOL, validate_protocol
from .checkpoint import export_numeric, load_numeric
from .model import HarmAwareJointUtilityERC, MIN_DEPLOY_GAIN
from .objective import GAIN_STEP, harm_aware_utility_loss


def main():
    validate_protocol()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU admission must not initialize CUDA")
    torch.manual_seed(20261105)
    model = HarmAwareJointUtilityERC().cpu()
    evidence = torch.randn(6, 25, 48)
    executable = torch.ones(6, 25, dtype=torch.bool)
    executable[2, 7] = False
    anchor = torch.zeros(6, dtype=torch.long)
    initial = model(evidence, executable, anchor)
    assert initial["predicted_gain"].eq(0).all()
    assert initial["index"].eq(0).all()

    utility = torch.zeros(6, 25)
    utility[0, 3] = GAIN_STEP + 0.02
    utility[1, 24] = GAIN_STEP + 0.01
    utility[2, 7] = 0.5  # invalid action must never be selected
    utility[3, 5] = GAIN_STEP / 2  # below the metric-aligned deployment floor
    utility[4, 9] = -0.20
    utility[5, 12] = GAIN_STEP + 0.03
    loss, stats = harm_aware_utility_loss(
        initial["predicted_gain"], executable, anchor, utility
    )
    loss.backward()
    gradient = model.utility_readout[-1].weight.grad
    assert torch.isfinite(loss) and gradient is not None
    assert torch.isfinite(gradient).all() and gradient.abs().sum().gt(0)

    manual = torch.zeros(4, 25)
    manual[0, 3] = MIN_DEPLOY_GAIN + 1e-4
    manual[1, 24] = MIN_DEPLOY_GAIN - 1e-4
    manual[2, 7] = 1.0
    manual[3, 8] = MIN_DEPLOY_GAIN
    manual_executable = torch.ones(4, 25, dtype=torch.bool)
    manual_executable[2, 7] = False
    chosen = model.choose(manual, manual_executable, torch.zeros(4, dtype=torch.long))
    assert chosen.tolist() == [3, 0, 0, 8]

    safe_prediction = torch.zeros(1, 25)
    harmful_utility = torch.zeros(1, 25)
    harmful_utility[0, 24] = -0.2
    bad_prediction = safe_prediction.clone()
    bad_prediction[0, 24] = GAIN_STEP
    one_exec = torch.ones(1, 25, dtype=torch.bool)
    one_anchor = torch.zeros(1, dtype=torch.long)
    safe_loss, _ = harm_aware_utility_loss(
        safe_prediction, one_exec, one_anchor, harmful_utility
    )
    bad_loss, _ = harm_aware_utility_loss(
        bad_prediction, one_exec, one_anchor, harmful_utility
    )
    assert bad_loss > safe_loss

    # Exercise the exact fixed-Epoch100 export/load path that spawn workers
    # will use after training.  This prevents a repeat of the prior checkpoint
    # identity mismatch while performing no optimization.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "synthetic_epoch100.pth.tar"
        torch.save({
            "kind": "rqc30_harm_aware_joint_utility_100ep_v1",
            "model_version": model.version,
            "protocol": PROTOCOL,
            "epoch": 100,
            "optimizer_steps": 187500,
            "model": model.state_dict(),
            "optimizer": {},
        }, checkpoint)
        numeric = root / "numeric"
        manifest = export_numeric(checkpoint, numeric)
        restored, restored_manifest = load_numeric(numeric, device="cpu")
        restored.eval()
        restored_output = restored(evidence, executable, anchor)
        assert manifest == restored_manifest
        torch.testing.assert_close(
            restored_output["predicted_gain"],
            model(evidence, executable, anchor)["predicted_gain"],
            rtol=0, atol=0,
        )

    report = {
        "status": "CPU_STATIC_ADMISSION_PASS",
        "protocol_version": PROTOCOL["version"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "zero_initial_exact_keep": True,
        "joint_keep_action_competition": True,
        "metric_aligned_threshold": MIN_DEPLOY_GAIN,
        "harm_false_positive_penalty_verified": True,
        "finite_nonzero_gradient": True,
        "epoch100_numeric_roundtrip": True,
        "synthetic_actionable_rate": float(stats["actionable_rate"]),
        "optimizer_steps": 0,
        "gpu_operations": 0,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

