# PQBC-Track core-code archive

This archive contains the source code needed to inspect and run the final
PQBC-Track implementation, including the UVLTrack base library, final RQC/ERC
modules, inherited runtime dependencies, experiment YAML files, and evaluation
entry points.

It intentionally excludes:

- datasets and dataset images;
- pretrained models and trained checkpoints;
- prediction/result directories;
- feature caches (`.npz`, `.npy`);
- training transactions (`.jsonl`);
- logs, TensorBoard data, Python bytecode, and temporary files.

Important entry points:

- `experiments/uvltrack/PQBC-Track.yaml`
- `change/run_pqbc_track_yaml.py`
- `change/recheck_pqbc_tnl2k_metrics.py`
- `change/erc_rqc30_temporal_cooldown_pairwise_seed20261109_100ep_20261109_123/`
- `change/erc_rqc30_temporal_cooldown_pairwise_seed20261109_geomfallback_20261109_124/`
- `change/erc_rqc30_output_only_ablation_20261109_130/`
- `change/erc_rqc30_unary_objective_ablation_20261109_132/`

The YAML records the server-side locations and SHA256 values of the final
checkpoints. The checkpoint files themselves are not included in this
code-only archive.

Basic source validation:

```bash
python -B -m change.run_pqbc_track_yaml check
```

The full evaluation additionally requires the datasets and checkpoint files
referenced by the YAML configuration.
