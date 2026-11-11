"""CPU-only, saved-prediction paired diagnosis; no tracker/model execution."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust
from directional_center_joint_contract_20261001 import NAME, file_hash

KEYS = ('AUC', 'OP50', 'OP75', 'Precision', 'Norm Precision')
RUNS = {'tnl2k': (101230, 700), 'lasot': (101231, 280), 'otb99': (101232, 48)}
EXPECTED_BASE = {'tnl2k': [62.73, 73.73, 58.90, 66.21, 80.28],
                 'lasot': [68.39, 79.68, 67.59, 74.49, 78.11],
                 'otb99': [69.39, 86.97, 56.67, 89.89, 83.93]}
EXPECTED_FULL = {'tnl2k': [63.02, 73.85, 59.14, 66.49, 80.40],
                 'lasot': [68.51, 79.46, 67.80, 74.38, 77.96],
                 'otb99': [69.46, 86.81, 56.78, 89.68, 83.68]}


def measure(box, gt, seq):
    visible = None if seq.target_visible is None else torch.tensor(seq.target_visible)
    iou, err, norm, valid = calc_seq_err_robust(box, gt, seq.dataset, visible)
    # Same official 21-point success grid, full sequence denominator and <= center thresholds.
    curve = (iou[:, None] > torch.arange(0, 1.0001, .05, dtype=torch.float64)).sum(0).float() / len(gt)
    values = dict(zip(KEYS, [float(curve.mean() * 100), float(curve[10] * 100),
                            float(curve[15] * 100), float((err <= 20).sum().float() / len(gt) * 100),
                            float((norm <= .2).sum().float() / len(gt) * 100)]))
    return values, iou, err, norm, valid


def sequence_summary(values, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    boot = np.concatenate([values[rng.randint(len(values), size=(500, len(values)))].mean(1)
                           for _ in range(20)])
    eps = 1e-7
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                positive=int((values > eps).sum()), negative=int((values < -eps).sum()),
                tied=int((abs(values) <= eps).sum()), positive_ratio=float((values > eps).mean()),
                paired_sequence_bootstrap_95=list(map(float, np.percentile(boot, [2.5, 97.5]))))


def describe(values):
    values = np.concatenate(values)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.percentile(values, 10)), p90=float(np.percentile(values, 90)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    dest = Path(args.output_dir)
    if dest.exists():
        raise FileExistsError(str(dest))
    baseline = Path(args.baseline_root)
    report = dict(config=NAME, epoch=30, mode='NLBBOX', datasets={}, optimizer_steps=0,
                  model_executed=False, model_changed=False,
                  limitations=[
                      'FULL minus official combines trained Head, its trajectory, and output refiner; not module-only attribution.',
                      'Bootstrap resamples paired sequences from one run, not training seeds; exploratory, not independent significance proof.',
                      'Test sets have informed prior development; no untouched-test or causal-mechanism claim.',
                      'Frame diagnostics exclude initialization and invalid/invisible frames; official metrics retain official denominator.'])
    for domain, (runid, expected) in RUNS.items():
        result = ROOT / 'test/tracking_results/uvltrack' / f'{NAME}_{runid}'
        manifest = json.loads((result / 'center_joint_run_manifest.json').read_text())
        assert (manifest['config'], manifest['dataset'], manifest['epoch'], manifest['runid']) == (NAME, domain, 30, runid)
        assert manifest['checkpoint_sha256'] == '1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd'
        assert file_hash(manifest['checkpoint']) == manifest['checkpoint_sha256']
        eval_root = ROOT / 'change/directional_center_joint_evaluation_20261001' / f'{NAME}_{domain}_e0030_run{runid}'
        assert json.loads((eval_root / 'input_manifest.json').read_text()) == manifest
        dataset = get_dataset(domain)
        assert len(dataset) == expected
        records, counts, distributions, buckets = [], {}, {}, {}
        total = 0
        for seq in dataset:
            gt = torch.tensor(seq.ground_truth_rect, dtype=torch.float64)
            boxes = [torch.tensor(np.loadtxt(str(path), ndmin=2), dtype=torch.float64) for path in
                     [baseline / f'{domain}_NLBBOX_300' / (seq.name + '.txt'),
                      result / f'{seq.dataset}_NLBBOX_030' / (seq.name + '.txt')]]
            assert all(b.shape == gt.shape and torch.isfinite(b).all() and (b[:, 2:] > 0).all() for b in boxes), seq.name
            measures = [measure(b, gt, seq) for b in boxes]
            bm, bi, be, bn, bv = measures[0]
            fm, fi, fe, fn, fv = measures[1]
            assert torch.equal(bv, fv)
            valid = bv.clone(); valid[0] = False
            total += int(valid.sum())
            gain = fi - bi
            masks = dict(iou_gain_gt_001=gain > .01, iou_loss_gt_001=gain < -.01,
                         precision_pass_to_fail=(be <= 20) & (fe > 20),
                         precision_fail_to_pass=(be > 20) & (fe <= 20),
                         norm_pass_to_fail=(bn <= .2) & (fn > .2),
                         norm_fail_to_pass=(bn > .2) & (fn <= .2),
                         iou_gain_but_precision_crossing_loss=(gain > .01) & (be <= 20) & (fe > 20),
                         iou_gain_but_norm_crossing_loss=(gain > .01) & (bn <= .2) & (fn > .2))
            for key, mask in masks.items(): counts[key] = counts.get(key, 0) + int((mask & valid).sum())
            diagnostic = dict(iou_delta=gain, center_error_delta_pixels=fe-be,
                              center_error_delta_normalized=fn-bn)
            for label, b in zip(('baseline', 'full'), boxes):
                diagnostic[label + '_center_error_pixels'] = measures[0 if label == 'baseline' else 1][2]
                diagnostic[label + '_center_error_normalized'] = measures[0 if label == 'baseline' else 1][3]
                diagnostic[label + '_absolute_log_area_error'] = torch.abs(torch.log(b[:, 2:].prod(1) / gt[:, 2:].prod(1).clamp_min(1e-9)))
                diagnostic[label + '_absolute_log_aspect_error'] = torch.abs(torch.log((b[:, 2] / b[:, 3]) / (gt[:, 2] / gt[:, 3].clamp_min(1e-9)).clamp_min(1e-9)))
            for key, value in diagnostic.items(): distributions.setdefault(key, []).append(value[valid].numpy())
            for label, mask in [('base_iou_ge_075', bi >= .75), ('base_iou_050_075', (bi >= .5) & (bi < .75)),
                                ('base_iou_010_050', (bi >= .1) & (bi < .5)), ('base_iou_lt_010', bi < .1)]:
                mask = mask & valid
                bucket = buckets.setdefault(label, dict(count=0, iou_gain_sum=0., center_error_delta_sum=0.))
                bucket['count'] += int(mask.sum())
                bucket['iou_gain_sum'] += float(gain[mask].sum())
                bucket['center_error_delta_sum'] += float((fe-be)[mask].sum())
            records.append(dict(sequence=seq.name, frames=len(gt), baseline=bm, full=fm,
                                delta={key: fm[key] - bm[key] for key in KEYS}))
        macro = {label: {key: float(np.mean([r[label][key] for r in records])) for key in KEYS}
                 for label in ('baseline', 'full')}
        for label, expected_values in [('baseline', EXPECTED_BASE[domain]), ('full', EXPECTED_FULL[domain])]:
            assert all(abs(macro[label][key] - value) < .011 for key, value in zip(KEYS, expected_values)), (domain, label, macro[label])
        summaries = {key: sequence_summary([r['delta'][key] for r in records], 20261002) for key in KEYS}
        auc_delta = np.asarray([r['delta']['AUC'] for r in records])
        auc_abs = abs(auc_delta).sum()
        top5_share = float(np.sort(abs(auc_delta))[-5:].sum() / auc_abs) if auc_abs else 0.
        for bucket in buckets.values():
            n = max(bucket['count'], 1)
            bucket['mean_iou_delta'] = bucket['iou_gain_sum'] / n
            bucket['mean_center_error_delta_pixels'] = bucket['center_error_delta_sum'] / n
        outcome = dict(runid=runid, manifest=manifest, sequences=len(records), valid_tracking_frames=total,
                       macro_metrics=macro, paired_sequence_deltas=summaries, top5_absolute_auc_change_share=top5_share,
                       frame_counts=counts, frame_rates={k: v / max(total, 1) for k, v in counts.items()},
                       frame_distributions={key: describe(values) for key, values in distributions.items()},
                       baseline_difficulty_buckets=buckets, sequence_details=records)
        report['datasets'][domain] = outcome
        print('PAIRED_AUDIT_COMPLETE', domain, json.dumps({k: outcome[k] for k in
              ['sequences', 'valid_tracking_frames', 'macro_metrics', 'paired_sequence_deltas', 'frame_counts', 'top5_absolute_auc_change_share']}), flush=True)
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / 'report.json').open('x') as f: json.dump(report, f, indent=2)
    lines = ['# Directional Center E30: saved-result paired audit', '',
             'CPU only; no model execution, training, or new tracker run. Changes are percentage points.', '',
             '| Dataset | AUC delta | Median sequence AUC delta | Positive sequences | Paired bootstrap 95% | P delta | NormP delta |',
             '|---|---:|---:|---:|---|---:|---:|']
    for domain, outcome in report['datasets'].items():
        s = outcome['paired_sequence_deltas']; a = s['AUC']; lo, hi = a['paired_sequence_bootstrap_95']
        lines.append(f"| {domain} | {a['mean']:+.4f} | {a['median']:+.4f} | {a['positive']}/{outcome['sequences']} | [{lo:+.4f}, {hi:+.4f}] | {s['Precision']['mean']:+.4f} | {s['Norm Precision']['mean']:+.4f} |")
    lines += ['', '## Interpretation boundaries', ''] + ['- ' + x for x in report['limitations']]
    with (dest / 'report.md').open('x', encoding='utf-8') as f: f.write('\n'.join(lines) + '\n')
    print('AUDIT_REPORT', str(dest), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
