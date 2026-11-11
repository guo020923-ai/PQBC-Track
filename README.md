# PQBC-Track

Official code release for **PQBC-Track**, a quality-aware extension of
[UVLTrack](https://arxiv.org/abs/2401.11228) for vision-language object
tracking.

PQBC-Track retains the unified visual-language tracker and introduces two
additional components:

- **RQC**, which produces a quality-controlled candidate box from the shared
  tracker representation;
- **ERC**, which compares executable candidate actions against `KEEP`, applies
  TRAIN-only calibration, and uses guarded feedback to prevent invalid state
  updates.

The final ERC configuration uses 25 candidate actions, a 48-dimensional action
descriptor, KEEP-relative pairwise utility learning, a TRAIN-only threshold of
`0.36725926399230957`, and an eight-frame cooldown.

> **Release status.** Source code and reproducibility metadata are available in
> this repository. Datasets and trained checkpoints are not stored in Git.
> Checkpoint downloads will be provided through the
> [GitHub Releases](https://github.com/guo020923-ai/PQBC-Track/releases) page.
> Public checkpoint copies retain the trained tensors while replacing local
> machine paths in metadata with repository-relative paths. The consolidated
> YAML records both public-file and original scientific-source SHA256 values.

## Results

### Contextual benchmark comparison

| Method | TNL2K AUC | TNL2K P | LaSOT AUC | LaSOT P | OTB-Lang AUC | OTB-Lang P |
|---|---:|---:|---:|---:|---:|---:|
| Reproduced UVLTrack-B | 62.73 | 66.21 | 68.39 | 74.49 | 69.39 | 89.89 |
| **PQBC-Track** | **63.27** | **66.83** | **68.60** | **74.52** | **69.79** | **90.18** |

`P` denotes center-location precision under the protocol used by each
benchmark.

### Final TNL2K FULL700 evaluation

| AUC | OP50 | OP75 | Precision | Normalized Precision |
|---:|---:|---:|---:|---:|
| 63.2737 | 74.1890 | 59.3355 | 66.8281 | 80.7188 |

The final run contains 700 sequences, 516,038 frames, and 515,338 search
frames. No automatic retry was used.

## Repository layout

```text
PQBC-Track/
├── change/        # RQC/ERC implementation and reproducibility entry points
├── experiments/   # UVLTrack, RQC, and consolidated PQBC-Track configurations
├── lib/           # UVLTrack base framework plus PQBC-Track modifications
├── tracking/      # training, testing, profiling, and analysis entry points
├── environment.yml
├── LICENSE
├── install.sh
└── README.md
```

Important entry points:

- `experiments/uvltrack/PQBC-Track.yaml`
- `change/run_pqbc_track_yaml.py`
- `change/recheck_pqbc_tnl2k_metrics.py`
- `change/erc_rqc30_temporal_cooldown_pairwise_seed20261109_100ep_20261109_123/`
- `change/erc_rqc30_temporal_cooldown_pairwise_seed20261109_geomfallback_20261109_124/`
- `change/erc_rqc30_output_only_ablation_20261109_130/`
- `change/erc_rqc30_unary_objective_ablation_20261109_132/`

## Installation

Linux with a CUDA-capable GPU is recommended.

```bash
conda env create -f environment.yml
conda activate pqbc-track
```

The environment file reflects the software stack used for the final
experiments: Python 3.8, PyTorch 2.0.1, and CUDA 11.8. As an equivalent
shortcut, run `bash install.sh`, then activate the environment with the command
printed by the script.

## Data preparation

Datasets are not included. Download them from their official sources and place
them under `data/`:

```text
data/
├── got10k/
│   └── train/
├── lasot/
├── lasotext/
├── otb99/
└── tnl2k/
    ├── train/
    └── test/
```

Update the dataset locations in the UVLTrack local environment files under
`lib/train/admin/` and `lib/test/evaluation/` when your directory layout is
different.

## Configuration validation

After configuring the artifact paths, validate the release configuration:

```bash
python -B -m change.run_pqbc_track_yaml check \
  --config experiments/uvltrack/PQBC-Track.yaml \
  --project-root . \
  --hash-artifacts
```

A valid setup prints:

```text
PQBC_TRACK_CONFIG_VALID
```

## TNL2K evaluation

Preview the complete evaluation command without starting a run:

```bash
python -B -m change.run_pqbc_track_yaml evaluate-command \
  --config experiments/uvltrack/PQBC-Track.yaml \
  --project-root . \
  --output outputs/pqbc_tnl2k \
  --result-name pqbc_track_tnl2k \
  --run-id 110925 \
  --processes 8
```

Run the evaluation by replacing `evaluate-command` with `evaluate`:

```bash
python -B -m change.run_pqbc_track_yaml evaluate \
  --config experiments/uvltrack/PQBC-Track.yaml \
  --project-root . \
  --output outputs/pqbc_tnl2k \
  --result-name pqbc_track_tnl2k \
  --run-id 110925 \
  --processes 8
```

All output directories and run identifiers must be new; the reproducibility
entry point does not overwrite an existing run.

## Recompute metrics from saved predictions

Place the completed tracker predictions in the UVLTrack result directory and
configure the local test environment. Then run:

```bash
python -B -m change.recheck_pqbc_tnl2k_metrics \
  --report outputs/metric_recheck
```

This command evaluates the saved predictions only; it does not run network
inference.

## Reproducibility controls

- final pairwise checkpoint: fixed Epoch 100;
- optimizer steps: 187,500;
- training seed: `20261109`;
- calibration scope: TRAIN-only;
- ERC threshold: `0.36725926399230957`;
- cooldown: eight frames;
- invalid feedback: post-selection KEEP fallback;
- final benchmark evaluation: FULL700;
- automatic retry: disabled.

See `experiments/uvltrack/PQBC-Track.yaml` and the implementation manifests in
`change/` for the complete protocol.

## License and acknowledgements

PQBC-Track is built on UVLTrack, the official implementation of
*Unifying Visual and Vision-Language Tracking via Contrastive Learning*
(AAAI 2024). We thank the UVLTrack authors for releasing their code.

The inherited UVLTrack components are redistributed and modified under the MIT
License. The original copyright notice is retained in `LICENSE`:

```text
Copyright (c) 2023 University of Science and Technology of China
```

The repository also inherits components or ideas acknowledged by UVLTrack,
including JointNLT, MixFormer, OSTrack, Timm, and
`pytorch_pretrained_bert`. Please follow the licenses of the corresponding
upstream projects.

## Citation

The PQBC-Track citation will be added after the paper metadata is finalized.
If you use the UVLTrack base implementation, please also cite:

```bibtex
@misc{ma2024unifying,
  title={Unifying Visual and Vision-Language Tracking via Contrastive Learning},
  author={Yinchao Ma and Yuyang Tang and Wenfei Yang and Tianzhu Zhang and
          Jinpeng Zhang and Mengxue Kang},
  year={2024},
  eprint={2401.11228},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Contact

Please use the repository's
[Issues](https://github.com/guo020923-ai/PQBC-Track/issues) page for questions
about the released code and reproducibility protocol.
