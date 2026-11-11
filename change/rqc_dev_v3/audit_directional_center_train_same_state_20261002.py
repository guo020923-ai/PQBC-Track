"""Fixed training-domain prefixes: native/refined pairs from one natural trajectory.

No optimizer, fitting, inference thresholds, model changes or official test writes.
GT is supplied to the tracker ONLY for frame-zero initialization; later GT is
read strictly for offline diagnostics after prediction.
"""
import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from directional_center_joint_contract_20261001 import NAME, file_hash
from directional_center_joint_tracker_20261001 import JointCenterEvaluationTracker
from lib.test.analysis.extract_results import calc_seq_err_robust

CHECKPOINT_SHA = '1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd'
_WRAPPERS = {}
_CHECKPOINT = None


def worker_init(checkpoint):
    global _CHECKPOINT
    _CHECKPOINT = Path(checkpoint)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)


def read_frames(row):
    path = Path(row['path'])
    tnl = row['domain'] == 'tnl2k'
    frames = sorted(p for p in (path / ('imgs' if tnl else 'img')).iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    gt = np.loadtxt(str(path / 'groundtruth.txt'), delimiter=',', ndmin=2)
    assert gt.shape == (len(frames), 4) and len(frames) > 1, row['name']
    assert np.isfinite(gt[0]).all() and (gt[0, 2:] > 0).all(), row['name']
    language = (path / ('language.txt' if tnl else 'nlp.txt')).read_text().strip()
    assert language
    visible = np.ones(len(gt), dtype=bool)
    if not tnl:
        for filename in ('full_occlusion.txt', 'out_of_view.txt'):
            flag = np.loadtxt(str(path / filename), delimiter=',').reshape(-1)
            assert len(flag) == len(gt)
            visible &= flag == 0
    return frames, gt, language.splitlines()[0], visible


def prediction_errors(box, gt, domain, visible):
    return calc_seq_err_robust(torch.tensor(box, dtype=torch.float64),
                               torch.tensor(gt, dtype=torch.float64), domain,
                               torch.tensor(visible) if domain == 'lasot' else None)


def one_sequence(task):
    row, output, max_frames = task
    frames, gt, language, visible = read_frames(row)
    end = min(len(frames), max_frames)
    domain = row['domain']
    if domain not in _WRAPPERS:
        _WRAPPERS[domain] = JointCenterEvaluationTracker(domain, _CHECKPOINT, 102030, epoch=30)
    wrapper = _WRAPPERS[domain]
    tracker = wrapper.tracker_class(wrapper.params, domain)
    start = time.time()
    native, refined, deltas = [gt[0].tolist()], [gt[0].tolist()], [[0., 0.]]
    state_checks = 0

    def read_image(path):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError('Unreadable frame: ' + str(path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        tracker.initialize(read_image(frames[0]), dict(init_bbox=gt[0].tolist(), language=language))
        assert int(tracker.flag.item()) == 2
        for frame in range(1, end):
            # No GT, sequence identity or dataset-specific tuning is passed here.
            result = tracker.track(read_image(frames[frame]))
            n = np.asarray(tracker.last_native_box, dtype=np.float64)
            f = np.asarray(result['target_bbox'], dtype=np.float64)
            assert np.array_equal(np.asarray(tracker.state), n), 'Corrected output fed back to state'
            assert n.shape == f.shape == (4,) and np.isfinite(n).all() and np.isfinite(f).all()
            assert (n[2:] > 0).all() and (f[2:] > 0).all()
            native.append(n.tolist()); refined.append(f.tolist())
            deltas.append(tracker._center_prediction['delta'][0].cpu().tolist())
            state_checks += 1
    finally:
        tracker.close()
        del tracker
        gc.collect()
        torch.cuda.empty_cache()
    arrays = [np.asarray(native), np.asarray(refined)]
    assert all(a.shape == (end, 4) for a in arrays)
    results = {}
    for format_name, predictions in [('float', arrays), ('saved_integer', [np.trunc(a) for a in arrays])]:
        measures = [prediction_errors(a, gt[:end], domain, visible[:end]) for a in predictions]
        ni, ne, nn, valid = measures[0]
        fi, fe, fn, v2 = measures[1]
        assert torch.equal(valid, v2)
        valid = valid.clone(); valid[0] = False
        if not valid.any():
            raise RuntimeError('No valid diagnostic frames: ' + row['name'])
        gain = fi - ni
        curve = torch.arange(0., 1.0001, .05, dtype=torch.float64)
        metrics = []
        for iou, err, norm, _ in measures:
            metrics.append(dict(AUC=float((iou[:, None] > curve).float().mean() * 100),
                                Precision=float((err <= 20).float().mean() * 100),
                                Norm_Precision=float((norm <= .2).float().mean() * 100)))
        results[format_name] = dict(
            valid_frames=int(valid.sum()), iou_native=ni[valid].tolist(), iou_refined=fi[valid].tolist(),
            iou_gain=gain[valid].tolist(), center_error_delta=(fe-ne)[valid].tolist(),
            norm_error_delta=(fn-nn)[valid].tolist(),
            beneficial_count=int(((gain > .01) & valid).sum()),
            harmful_count=int(((gain < -.01) & valid).sum()),
            severe_harmful_count=int(((gain < -.2) & valid).sum()),
            precision_pass_to_fail=int(((ne <= 20) & (fe > 20) & valid).sum()),
            precision_fail_to_pass=int(((ne > 20) & (fe <= 20) & valid).sum()),
            norm_pass_to_fail=int(((nn <= .2) & (fn > .2) & valid).sum()),
            norm_fail_to_pass=int(((nn > .2) & (fn <= .2) & valid).sum()),
            native_metrics=metrics[0], refined_metrics=metrics[1],
            metric_delta={key: metrics[1][key]-metrics[0][key] for key in metrics[0]})
    record = dict(sequence=row, tracked_frames=end-1, state_equal_native_checks=state_checks,
                  seconds=time.time()-start, native_boxes=native, refined_boxes=refined,
                  gt=gt[:end].tolist(), visible=visible[:end].tolist(), delta=deltas,
                  output_changed_float=int(np.any(arrays[0][1:] != arrays[1][1:], axis=1).sum()),
                  output_changed_integer=int(np.any(np.trunc(arrays[0][1:]) != np.trunc(arrays[1][1:]), axis=1).sum()),
                  formats=results)
    dest = Path(output) / ('sequence_' + str(row['id']).zfill(3) + '.json')
    temporary = dest.with_suffix('.partial')
    with temporary.open('x') as f:
        json.dump(record, f)
    if dest.exists():
        raise FileExistsError(str(dest))
    temporary.rename(dest)
    return dict(id=row['id'], domain=domain, name=row['name'], frames=end-1,
                seconds=record['seconds'], valid=results['float']['valid_frames'])


def bootstrap(values):
    values = np.asarray(values)
    rng = np.random.RandomState(20261002)
    draws = values[rng.randint(len(values), size=(10000, len(values)))].mean(1)
    return np.percentile(draws, [2.5, 97.5]).tolist()


def summarize(records):
    summary = {}
    for name in ['float', 'saved_integer']:
        rows = [r['formats'][name] for r in records]
        gains = np.concatenate([r['iou_gain'] for r in rows])
        n = len(gains)
        per_sequence = [float(np.mean(r['iou_gain'])) for r in rows]
        entry = dict(valid_frames=n, frame_mean_iou_gain=float(gains.mean()),
                     sequence_mean_iou_gain=float(np.mean(per_sequence)),
                     sequence_median_iou_gain=float(np.median(per_sequence)),
                     positive_sequences=int((np.asarray(per_sequence) > 1e-8).sum()),
                     paired_sequence_bootstrap_95=bootstrap(per_sequence),
                     mean_center_error_delta_pixels=float(np.concatenate([r['center_error_delta'] for r in rows]).mean()),
                     mean_norm_error_delta=float(np.concatenate([r['norm_error_delta'] for r in rows]).mean()),
                     prefix_macro_metric_delta={key: float(np.mean([r['metric_delta'][key] for r in rows]))
                                                for key in rows[0]['metric_delta']})
        for key in ['beneficial_count', 'harmful_count', 'severe_harmful_count',
                    'precision_pass_to_fail', 'precision_fail_to_pass', 'norm_pass_to_fail', 'norm_fail_to_pass']:
            count = sum(r[key] for r in rows)
            entry[key] = count
            entry[key.replace('_count', '_rate') if key.endswith('_count') else key + '_rate'] = count/n
        buckets = {}
        native_iou = np.concatenate([r['iou_native'] for r in rows])
        for label, mask in [('native_iou_ge_075', native_iou >= .75),
                            ('native_iou_050_075', (native_iou >= .5) & (native_iou < .75)),
                            ('native_iou_lt_050', native_iou < .5)]:
            buckets[label] = dict(frames=int(mask.sum()), mean_iou_gain=float(gains[mask].mean()) if mask.any() else None)
        entry['native_quality_buckets'] = buckets
        summary[name] = entry
    delta = np.concatenate([np.asarray(r['delta'])[1:] for r in records])
    shifts = np.concatenate([(np.asarray(r['refined_boxes'])[:, :2]+np.asarray(r['refined_boxes'])[:, 2:]/2
                              -np.asarray(r['native_boxes'])[:, :2]-np.asarray(r['native_boxes'])[:, 2:]/2)[1:]
                             for r in records])
    total = sum(r['tracked_frames'] for r in records)
    summary.update(sequences=len(records), tracked_frames=total,
                   state_equal_native_checks=sum(r['state_equal_native_checks'] for r in records),
                   output_changed_float_rate=sum(r['output_changed_float'] for r in records)/total,
                   output_changed_integer_rate=sum(r['output_changed_integer'] for r in records)/total,
                   delta_mean=delta.mean(0).tolist(), delta_abs=float(abs(delta).mean()),
                   saturation_ratio=float((abs(delta) > .095).mean()),
                   center_shift_mean_pixels=shifts.mean(0).tolist())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--per-domain', type=int, default=20)
    parser.add_argument('--frames', type=int, default=512)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8 and args.per_domain >= 2 and args.frames >= 2
    os.chdir(ROOT)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    dest = Path(args.output).resolve()
    if dest.exists():
        raise FileExistsError(str(dest))
    checkpoint = ROOT/'change/directional_center_joint_training_20261001'/NAME/'DirectionalCenterJoint_ep0030.pth.tar'
    saved = torch.load(str(checkpoint), map_location='cpu')
    assert saved['epoch'] == 30 and saved['variant'] == 'directional_center_guard_joint' and not saved.get('synthetic_check')
    assert file_hash(checkpoint) == CHECKPOINT_SHA
    identity = saved['identity']
    assert file_hash(ROOT/'experiments/uvltrack'/(NAME+'.yaml')) == identity['config_sha256']
    for name, digest in identity['source_files'].items():
        assert file_hash(Path(__file__).parent/name) == digest, name
    assert file_hash(identity['source_checkpoint']) == identity['source_checkpoint_sha256']
    del saved
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    from lib.train.admin.settings import Settings
    from lib.train.base_functions import names2datasets
    from lib.train.data import opencv_loader
    from lib.test.evaluation import get_dataset
    datasets = names2datasets(['LASOT', 'TNL2K'], Settings(), opencv_loader)
    rng = random.Random(20261002)
    manifest = []
    for dataset in datasets:
        domain = dataset.get_name()
        test_dirs = {str(Path(seq.frames[0]).resolve().parent.parent) for seq in get_dataset(domain)}
        pool = sorted((name, str(Path(dataset._get_sequence_path(i)).resolve()))
                      for i, name in enumerate(dataset.sequence_list))
        assert not {path for _, path in pool} & test_dirs
        selected = rng.sample(pool, args.per_domain)
        for name, path in selected:
            row = dict(id=len(manifest), domain=domain, name=name, path=path, role='training_domain_development')
            frames, gt, language, visible = read_frames(row)
            row.update(total_frames=len(frames), audited_frames=min(len(frames), args.frames)-1,
                       groundtruth_sha256=file_hash(Path(path)/'groundtruth.txt'),
                       language_sha256=file_hash(Path(path)/('language.txt' if domain == 'tnl2k' else 'nlp.txt')))
            manifest.append(row)
    dest.mkdir(parents=True, exist_ok=False)
    protocol = dict(config=NAME, checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA,
                    seed=20261002, selection='Uniform training-sequence sample before predictions; no selection by gain or difficulty',
                    workers=args.workers, max_frames=args.frames, manifest=manifest,
                    no_test_video_overlap=True, optimizer_steps=0, full_off_test=False,
                    model_changed=False, gt_usage='Initialization box only; later labels used only after prediction',
                    limitations=['Training-domain development sequences may have been seen during official/Head training; not held-out generalization.',
                                 'Only early prefixes, not full benchmark AUC or module-independent cross-dataset proof.',
                                 'Same trained Head and same history; comparison isolates output correction, not Head-vs-official causality.',
                                 'Bootstrap over sequences of one model is exploratory, not repeated-seed significance.'])
    with (dest/'protocol.json').open('x') as f: json.dump(protocol, f, indent=2)
    print('SAME_STATE_PREFLIGHT_PASS', json.dumps(dict(config=NAME, sequences=len(manifest),
          total_tracking_frames=sum(r['audited_frames'] for r in manifest), workers=args.workers,
          no_test_video_overlap=True, output=str(dest))), flush=True)
    start = time.time()
    tasks = [(row, str(dest), args.frames) for row in manifest]
    with mp.get_context('spawn').Pool(args.workers, initializer=worker_init, initargs=(str(checkpoint),)) as pool:
        for done, result in enumerate(pool.imap_unordered(one_sequence, tasks), 1):
            print('SAME_STATE_SEQUENCE_COMPLETE', done, len(tasks), json.dumps(result), flush=True)
    records = [json.loads((dest/f'sequence_{row["id"]:03d}.json').read_text()) for row in manifest]
    assert all(r['state_equal_native_checks'] == r['tracked_frames'] for r in records)
    assert file_hash(checkpoint) == CHECKPOINT_SHA
    summaries = {domain: summarize([r for r in records if r['sequence']['domain'] == domain])
                 for domain in ['lasot', 'tnl2k']}
    report = dict(protocol=protocol, seconds=time.time()-start, domains=summaries,
                  sequences=len(records), optimizer_steps=0, model_changed=False)
    with (dest/'report.json').open('x') as f: json.dump(report, f, indent=2)
    with (dest/'completed.json').open('x') as f:
        json.dump(dict(sequences=len(records), tracked_frames=sum(r['tracked_frames'] for r in records),
                       seconds=report['seconds'], checkpoint_sha256=CHECKPOINT_SHA, optimizer_steps=0), f, indent=2)
    print('SAME_STATE_AUDIT_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
