"""Build a fixed 30k TRAIN-only cache on feedback-consistent RQC trajectories."""

import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from change.erc_rqc30_epoch_supervised_20261104_39.objective import action_utility
from .config import PROTOCOL, validate_protocol
from .io_utils import atomic_json, digest, write_new_json
from .runtime import FeedbackPairwiseRuntime, deterministic_torch


def _read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("missing TRAIN frame: " + str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FeedbackRQCStream:
    def __init__(self, pools, params, behavior_numeric, seed):
        from change.erc_natural_reuse_fresh_20261021_01.bridge import collector_class
        from change.erc_rqc30_epoch_supervised_abstention_20261104_41.provider import causal_candidate_provider
        from change.erc_rqc30_pairwise_utility_100ep_20261105_60.checkpoint import load_numeric

        self.pools, self.params, self.rng = pools, params, random.Random(seed)
        self.collector = collector_class()(params, "tnl2k")
        self.behavior, manifest = load_numeric(behavior_numeric, device="cuda")
        self.behavior.eval().requires_grad_(False)
        if (manifest.get("source_checkpoint_sha256") != PROTOCOL["behavior_checkpoint_sha256"] or
                manifest.get("numeric_npz_sha256") != PROTOCOL["behavior_numeric_sha256"]):
            raise ValueError("fixed behavior identity")
        calibration_root = Path(behavior_numeric)
        calibration_path = calibration_root / "pairwise_evaluation_fallback.json"
        if not calibration_path.exists():
            calibration_path = calibration_root / "pairwise_calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("test_data_read", False) or float(calibration["threshold"]) != PROTOCOL["behavior_threshold"]:
            raise ValueError("fixed behavior threshold identity")
        self.runtime = FeedbackPairwiseRuntime(
            self.collector, self.behavior, causal_candidate_provider,
            PROTOCOL["behavior_threshold"],
        )
        self.clip = None; self.cursor = 0

    def _sample_clip(self):
        warmup, sample = PROTOCOL["cache_warmup_frames"], PROTOCOL["cache_frames_per_clip"]
        total = 1 + warmup + sample
        for _ in range(2000):
            source = self.rng.choices(
                self.pools.vl, [self.pools.weights[index] for index in self.pools.vl]
            )[0]
            pool = self.pools.datasets[source]
            sid = self.rng.randrange(pool.get_num_sequences())
            info = pool.get_sequence_info(sid)
            valid = (info["valid"].bool() & info["visible"].bool()).cpu()
            if len(valid) < total:
                continue
            starts = [begin for begin in range(len(valid) - total + 1)
                      if bool(valid[begin:begin + total].all())]
            if not starts:
                continue
            begin = self.rng.choice(starts)
            frames, annotation, meta = pool.get_frames(sid, [begin], info)
            language = meta.get("language")
            if not isinstance(language, str) or not language.strip():
                continue
            sequence_path = Path(pool._get_sequence_path(sid)).resolve()
            frame_dir = sequence_path / ("imgs" if pool.get_name() == "tnl2k" else "img")
            paths = sorted(path for path in frame_dir.iterdir()
                           if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if len(paths) < begin + total:
                continue
            return {
                "source": int(source), "domain": pool.get_name(),
                "sequence": pool.sequence_list[sid], "begin": begin,
                "paths": paths, "annotations": info["bbox"],
                "initial_image": frames[0], "initial_box": annotation["bbox"][0].tolist(),
                "language": language,
            }
        raise RuntimeError("no contiguous original-TRAIN feedback cache clip after 2000 draws")

    def _new_clip(self):
        self.clip = self._sample_clip()
        self.runtime.initialize(self.clip["initial_image"], {
            "init_bbox": self.clip["initial_box"], "language": self.clip["language"],
        })
        for offset in range(1, PROTOCOL["cache_warmup_frames"] + 1):
            image = _read_rgb(self.clip["paths"][self.clip["begin"] + offset])
            self.runtime.track(image, explore=False)
        self.cursor = self.clip["begin"] + 1 + PROTOCOL["cache_warmup_frames"]

    @torch.no_grad()
    def next_example(self):
        end = None if self.clip is None else (
            self.clip["begin"] + 1 + PROTOCOL["cache_warmup_frames"] +
            PROTOCOL["cache_frames_per_clip"]
        )
        if self.clip is None or self.cursor >= end:
            self._new_clip()
        image = _read_rgb(self.clip["paths"][self.cursor])
        self.runtime.track(image, explore=False)
        packet = self.runtime.last["packet"]
        truth = self.clip["annotations"][self.cursor].to(
            packet["boxes"].device, dtype=torch.double
        )[None]
        utility = action_utility(packet["boxes"], truth)
        row = {
            "evidence": packet["evidence"][0].float().cpu().numpy(),
            "executable": packet["executable"][0].cpu().numpy(),
            "utility": utility[0].float().cpu().numpy(),
            "source": self.clip["source"],
            "behavior_action": self.runtime.last["index"],
            "feedback_committed": self.runtime.last["feedback_committed"],
        }
        self.cursor += 1
        return row

    def close(self):
        close = getattr(self.collector, "close", None)
        if close is not None:
            close()


def _write_shard(cache_dir, index, rows):
    final = cache_dir / ("shard_%04d.npz" % index)
    temporary = cache_dir / ("shard_%04d.partial" % index)
    arrays = {
        "evidence": np.stack([row["evidence"] for row in rows]).astype(np.float16),
        "executable": np.stack([row["executable"] for row in rows]).astype(np.bool_),
        "utility": np.stack([row["utility"] for row in rows]).astype(np.float32),
        "source": np.asarray([row["source"] for row in rows], dtype=np.int16),
        "behavior_action": np.asarray([row["behavior_action"] for row in rows], dtype=np.int16),
        "feedback_committed": np.asarray([row["feedback_committed"] for row in rows], dtype=np.bool_),
    }
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, final)
    return {"name": final.name, "examples": len(rows), "sha256": digest(final)}


def build_cache(output_dir, behavior_numeric):
    validate_protocol(); deterministic_torch()
    output_dir = Path(output_dir); cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    from change.erc_natural_reuse_fresh_20261021_01.bridge import original_training_components

    legacy, _ = original_training_components(); legacy.idle()
    spec, cfg, plan, params = legacy.settings(); del spec
    pools = legacy.FitPools(cfg, plan)
    stream = FeedbackRQCStream(pools, params, behavior_numeric, PROTOCOL["cache_seed"])
    started = time.monotonic(); total = PROTOCOL["cache_examples"]
    shards, rows, intervals = [], [], []
    interval_started, interval_completed = started, 0
    actions = [0] * PROTOCOL["candidate_actions"]; commits = 0
    try:
        for completed in range(1, total + 1):
            row = stream.next_example(); rows.append(row)
            actions[row["behavior_action"]] += 1; commits += int(row["feedback_committed"])
            if len(rows) == 1024 or completed == total:
                shards.append(_write_shard(cache_dir, len(shards), rows)); rows = []
            if completed == 1 or completed % 100 == 0 or completed == total:
                now = time.monotonic(); elapsed = now - started
                if completed > interval_completed:
                    intervals.append((now - interval_started) / (completed - interval_completed))
                    intervals = intervals[-20:]; interval_started, interval_completed = now, completed
                remaining = total - completed
                atomic_json(output_dir / "progress.json", {
                    "phase": "CACHE_EXTRACTION", "completed_examples": completed,
                    "total_examples": total, "elapsed_seconds": elapsed,
                    "examples_per_second": completed / max(elapsed, 1e-9),
                    "eta_seconds": remaining * elapsed / max(completed, 1),
                    "eta_seconds_low": remaining * float(np.quantile(intervals, 0.2)),
                    "eta_seconds_high": remaining * float(np.quantile(intervals, 0.8)),
                    "feedback_commits": commits, "updated_unix": time.time(),
                })
        manifest = {
            "schema": "rqc30_feedback_pairwise_cache_v1", "protocol": PROTOCOL,
            "examples": total, "shards": shards, "seconds": time.monotonic() - started,
            "train_only": True, "fixed_rqc_trajectory": False,
            "feedback_consistent": True, "feedback_from_first_post_init_frame": True,
            "warmup_uses_same_feedback": True,
            "deterministic_source_execution": PROTOCOL["deterministic_source_execution"],
            "determinism_mode": PROTOCOL["determinism_mode"],
            "behavior_checkpoint_sha256": PROTOCOL["behavior_checkpoint_sha256"],
            "behavior_numeric_sha256": PROTOCOL["behavior_numeric_sha256"],
            "behavior_threshold": PROTOCOL["behavior_threshold"],
            "behavior_action_totals": actions, "feedback_commits": commits,
            "test_data_read": False,
        }
        write_new_json(cache_dir / "manifest.json", manifest)
        return manifest
    finally:
        stream.close()

