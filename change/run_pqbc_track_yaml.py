"""Validate and launch the sealed PQBC-Track pipeline from its YAML config.

The final historical experiment was executed from a sealed Python package.
This adapter makes ``experiments/uvltrack/PQBC-Track.yaml`` operational
without modifying that sealed implementation or its recorded hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


DEFAULT_CONFIG = Path("experiments/uvltrack/PQBC-Track.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(mapping: Dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing YAML field: {path}")
        value = value[key]
    return value


def _resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def _check_identity(path: Path, expected_sha256: str, hash_artifacts: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if hash_artifacts:
        actual = _sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"SHA256 mismatch: {path}: {actual} != {expected_sha256}")


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("PQBC-Track YAML root must be a mapping")
    if _require(data, "EXPERIMENT.NAME") != "PQBC-Track":
        raise ValueError("EXPERIMENT.NAME must be PQBC-Track")
    if _require(data, "EXPERIMENT.SCHEMA") != "pqbc_track_reproducibility_v1":
        raise ValueError("unsupported PQBC-Track YAML schema")
    return data


def _protocol_expectations(config: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    return (
        ("cache_seed", _require(config, "CACHE.SEED")),
        ("training_seed", _require(config, "TRAIN.SEED")),
        ("source_rqc_epoch", _require(config, "CHECKPOINTS.RQC.EPOCH")),
        (
            "source_rqc_checkpoint_sha256",
            _require(config, "CHECKPOINTS.RQC.SOURCE_SHA256"),
        ),
        ("source_cache_manifest_sha256", _require(config, "CACHE.MANIFEST_SHA256")),
        ("cache_examples", _require(config, "CACHE.EXAMPLES")),
        ("epochs", _require(config, "TRAIN.EPOCHS")),
        ("batch_size", _require(config, "TRAIN.BATCH_SIZE")),
        ("steps_per_epoch", _require(config, "TRAIN.STEPS_PER_EPOCH")),
        ("optimizer_steps", _require(config, "TRAIN.OPTIMIZER_STEPS")),
        ("peak_learning_rate", _require(config, "TRAIN.PEAK_LEARNING_RATE")),
        ("minimum_learning_rate", _require(config, "TRAIN.MINIMUM_LEARNING_RATE")),
        ("warmup_epochs", _require(config, "TRAIN.WARMUP_EPOCHS")),
        ("weight_decay", _require(config, "TRAIN.WEIGHT_DECAY")),
        ("gradient_clip", _require(config, "TRAIN.GRADIENT_CLIP")),
        ("candidate_actions", _require(config, "ERC.CANDIDATE_ACTIONS")),
        ("evidence_dim", _require(config, "ERC.EVIDENCE_DIM")),
        ("hidden_dim", _require(config, "ERC.HIDDEN_DIM")),
        ("pair_temperature", _require(config, "ERC.PAIR_TEMPERATURE")),
        ("action_basis", _require(config, "ERC.ACTION_BASIS")),
        ("shrink_factor", _require(config, "ERC.SHRINK_FACTOR")),
        ("feedback_mode", _require(config, "ERC.FEEDBACK_MODE")),
        ("temporal_cooldown_frames", _require(config, "ERC.TEMPORAL_COOLDOWN_FRAMES")),
        ("temporal_cooldown_selection", _require(config, "CALIBRATION.SELECTION")),
        ("calibration_block_rows", _require(config, "CALIBRATION.BLOCK_ROWS")),
        ("calibration_block_modulus", _require(config, "CALIBRATION.BLOCK_MODULUS")),
        ("calibration_block_remainder", _require(config, "CALIBRATION.BLOCK_REMAINDER")),
        ("checkpoint_selection", _require(config, "CHECKPOINTS.ERC.SELECTION")),
        ("test_data_in_training_or_calibration", False),
    )


def validate_config(
    config_path: Path,
    project_root: Path | None = None,
    hash_artifacts: bool = False,
) -> Dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    project = (project_root or Path(_require(config, "EXECUTION.PROJECT_ROOT"))).resolve()
    if not project.is_dir():
        raise FileNotFoundError(project)

    for key in ("BASE_CONFIG.BASELINE_YAML", "BASE_CONFIG.RQC_YAML", "BASE_CONFIG.RQC_PROTOCOL"):
        path = _resolve(project, str(_require(config, key)))
        if not path.is_file():
            raise FileNotFoundError(path)

    package_name = str(_require(config, "ERC.IMPLEMENTATION_PACKAGE"))
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    protocol_module = importlib.import_module(package_name + ".config")
    protocol = protocol_module.validate_protocol()
    for key, expected in _protocol_expectations(config):
        actual = protocol.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-15:
                raise ValueError(f"protocol mismatch: {key}: {actual!r} != {expected!r}")
        elif actual != expected:
            raise ValueError(f"protocol mismatch: {key}: {actual!r} != {expected!r}")

    manifest = _resolve(project, str(_require(config, "PROVENANCE.IMPLEMENTATION_MANIFEST")))
    original_config = _resolve(project, str(_require(config, "PROVENANCE.ORIGINAL_CONFIG_PY")))
    if not manifest.is_file() or not original_config.is_file():
        raise FileNotFoundError(manifest if not manifest.is_file() else original_config)
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_data.get("status") != "IMPLEMENTATION_SEALED_AND_AUTHORIZED":
        raise ValueError("sealed implementation manifest is not authorized")

    for name in ("BASELINE", "RQC", "ERC"):
        record = _require(config, f"CHECKPOINTS.{name}")
        _check_identity(
            _resolve(project, str(record["PATH"])),
            str(record["SHA256"]),
            hash_artifacts,
        )
    erc = _require(config, "CHECKPOINTS.ERC")
    _check_identity(
        _resolve(project, str(erc["NUMERIC_PATH"])),
        str(erc["NUMERIC_SHA256"]),
        hash_artifacts,
    )
    _check_identity(
        _resolve(project, str(_require(config, "CACHE.PATH"))) / "manifest.json",
        str(_require(config, "CACHE.MANIFEST_SHA256")),
        hash_artifacts,
    )

    if _require(config, "CALIBRATION.SCOPE") != "TRAIN_only":
        raise ValueError("calibration must remain TRAIN_only")
    if _require(config, "TRAIN.TEST_DATA_USED") is not False:
        raise ValueError("training must not use test data")
    if _require(config, "CALIBRATION.TEST_DATA_USED") is not False:
        raise ValueError("calibration must not use test data")
    if _require(config, "TEST.FULL700") is not True:
        raise ValueError("final evaluation must remain FULL700")

    return {
        "status": "PQBC_TRACK_CONFIG_VALID",
        "config": str(config_path),
        "project_root": str(project),
        "implementation_package": package_name,
        "hash_artifacts": hash_artifacts,
        "checkpoint_epoch": _require(config, "CHECKPOINTS.ERC.SCIENTIFIC_EPOCH"),
        "threshold": _require(config, "CALIBRATION.THRESHOLD"),
        "cooldown": _require(config, "ERC.TEMPORAL_COOLDOWN_FRAMES"),
        "full700": _require(config, "TEST.FULL700"),
    }


def _pipeline_command(config: Dict[str, Any], project: Path, output: Path) -> list[str]:
    package = str(_require(config, "ERC.IMPLEMENTATION_PACKAGE"))
    cache = _resolve(project, str(_require(config, "CACHE.PATH")))
    return [
        sys.executable,
        "-B",
        "-m",
        package + ".run_execution_pipeline",
        str(project),
        str(cache),
        str(output),
    ]


def _evaluation_command(
    config_path: Path,
    project: Path,
    output: Path,
    result_name: str,
    run_id: int,
    processes: int,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "change.run_pqbc_track_yaml",
        "evaluate",
        "--config",
        str(config_path),
        "--project-root",
        str(project),
        "--output",
        str(output),
        "--result-name",
        result_name,
        "--run-id",
        str(run_id),
        "--processes",
        str(processes),
    ]


def _evaluate_existing(
    config: Dict[str, Any],
    project: Path,
    output: Path,
    result_name: str,
    run_id: int,
    processes: int,
) -> int:
    from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.run_authorized_pipeline import (
        require_idle_gpu,
    )
    from change.erc_rqc30_temporal_cooldown_pairwise_seed20261109_geomfallback_20261109_124 import (
        run_test_recovery as recovery_runner,
        stage as pairwise_stage,
    )

    if output.exists():
        raise FileExistsError(output)
    if processes < 1:
        raise ValueError("processes must be positive")
    require_idle_gpu()
    recovery_package = project / "change" / recovery_runner.PACKAGE.rsplit(".", 1)[-1]
    authorization = json.loads(
        (recovery_package / "authorization_124.json").read_text(encoding="utf-8")
    )
    if authorization.get("status") != "AUTHORIZED_ENGINEERING_RECOVERY_TEST":
        raise PermissionError("Round124 recovery evaluation is not authorized")
    recovery_runner.verify_manifest(project, recovery_package)
    final_job = _resolve(project, str(_require(config, "PROVENANCE.FINAL_TRAINING_JOB")))
    training_audit, numeric_dir, _, _ = recovery_runner.verify_source(final_job)

    # The sealed stage uses module-level names to construct result roots. Set
    # unique names for this manual replay so the historical official outputs
    # remain read-only and cannot be overwritten.
    pairwise_stage.inherited.RESULT_NAME = result_name
    pairwise_stage.inherited.RUNID = int(run_id)
    pairwise_stage.inherited.NATIVE_NAME = result_name + "_native_same_run"
    pairwise_stage.inherited.NATIVE_RUNID = int(run_id) + 1
    result = pairwise_stage.run_stage(
        training_audit,
        numeric_dir,
        output,
        processes=int(processes),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or run PQBC-Track from YAML")
    parser.add_argument(
        "action",
        choices=("check", "command", "run", "evaluate-command", "evaluate"),
        nargs="?",
        default="check",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output", type=Path, help="new output directory")
    parser.add_argument("--hash-artifacts", action="store_true")
    parser.add_argument("--result-name", help="unique tracker result name for evaluation")
    parser.add_argument("--run-id", type=int, help="unique evaluation run id; native shadow uses run-id + 1")
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args(argv)

    report = validate_config(
        args.config,
        args.project_root,
        args.hash_artifacts or args.action == "evaluate",
    )
    config = load_config(args.config)
    project = Path(report["project_root"])
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.action == "check":
        return 0
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.action in ("evaluate-command", "evaluate"):
        if not args.result_name or args.run_id is None:
            parser.error("--result-name and --run-id are required for evaluation")
        command = _evaluation_command(
            args.config.resolve(), project, output, args.result_name, args.run_id, args.processes
        )
        print(json.dumps({"evaluation_command": command}, indent=2), flush=True)
        if args.action == "evaluate-command":
            return 0
        return _evaluate_existing(
            config, project, output, args.result_name, args.run_id, args.processes
        )
    command = _pipeline_command(config, project, output)
    print(json.dumps({"pipeline_command": command}, indent=2), flush=True)
    if args.action == "command":
        return 0
    return subprocess.call(command, cwd=str(project))


if __name__ == "__main__":
    raise SystemExit(main())
