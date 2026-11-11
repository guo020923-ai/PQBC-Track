"""Read-only natural-state feasibility check. No fitting and no GT-driven tracking."""
import sys, json, math, time, itertools, os
from pathlib import Path
import numpy as np
import torch
import cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit_jq_natural_20260930 import NAME
from audit_bq_contract_20260929 import ious
from head_quality_features_20260930 import HeadQualityCapture
from local_box_refiner_20260930 import LocalBoxRefiner, residual_target
from lib.test.parameter.uvltrack import parameters
from lib.test.tracker.uvltrack import UVLTrack
from lib.utils.box_ops import clip_box


def main():
    os.chdir(str(ROOT))
    dest = ROOT/'change/rqc_dev_v3/local_refiner_audit_20260930_checked'
    dest.mkdir(exist_ok=False)
    source = ROOT/'change/rqc_dev_v3/jq_natural_20260930/manifest.json'
    chosen = json.loads(source.read_text())
    torch.set_num_threads(1); cv2.setNumThreads(1); torch.manual_seed(42)
    params = parameters(NAME, epoch=30); params.debug = 0
    tracker = UVLTrack(params, 'tnl2k')
    model = LocalBoxRefiner().cuda().eval()
    capture = {}; rows = []; cache = []; start = time.time(); max_error = 0.
    cap = HeadQualityCapture(tracker.network.box_head); cap.__enter__()
    def hook(module, inputs, out):
        batch = inputs[0]
        feature = cap.gather(batch['indices'][:, :1])[:, 0]
        box = batch['boxes'][:, 0]
        refined = model(feature, box)['boxes']
        assert torch.equal(refined, box)
        capture.update(box=box[0].cpu(), refined=refined[0].cpu(), feature=feature[0].cpu())
        changed = dict(out)
        changed['decision_score'] = torch.zeros_like(out['decision_score'])
        changed['decision_score'][:, 0] = 1
        return changed
    handle = tracker.network.rqc.register_forward_hook(hook)
    read = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    limits = np.array([.1, .1, math.log(1.2), math.log(1.2)])
    grid = np.array(list(itertools.product((-1., 0., 1.), repeat=4))) * limits
    try:
        for m in chosen:
            path = Path(m['path']); tnl = m['domain'] == 'tnl2k_train'
            if tnl: assert path.resolve().parent == (ROOT/'data/tnl2k/train').resolve()
            else: assert m['name'] in (ROOT/'lib/train/data_specs/lasot_train_split.txt').read_text().splitlines()
            frames = sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
            gt = np.loadtxt(path/'groundtruth.txt', delimiter=',', ndmin=2)
            assert len(gt) == len(frames)
            visible = np.ones(len(gt), bool)
            if not tnl:
                for fn in ('full_occlusion.txt', 'out_of_view.txt'):
                    visible &= np.loadtxt(path/fn, delimiter=',').reshape(-1) == 0
            language = (path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
            tracker.max_score = 0; tracker.out_dict = None; tracker.pred_box_net = None
            tracker.initialize(read(frames[0]), dict(init_bbox=gt[0].tolist(), language=language))
            for t in range(1, min(401, len(frames))):
                old = np.asarray(tracker.state)
                side = math.ceil(math.sqrt(old[2]*old[3])*params.search_factor)
                origin = old[:2] + old[2:]/2 - side/2
                image = read(frames[t]); output = tracker.track(image)['target_bbox']
                b = capture['box'].numpy().astype(float)
                def world(boxes):
                    return np.array([clip_box(np.r_[origin+(x[:2]-x[2:]/2)*side, x[2:]*side].tolist(), *image.shape[:2], margin=10) for x in boxes])
                zero = world([capture['refined'].numpy()])[0]
                max_error = max(max_error, float(np.max(np.abs(zero-output))))
                np.testing.assert_allclose(zero, output, rtol=0, atol=.002)
                if not visible[t] or not np.isfinite(gt[t]).all() or min(gt[t, 2:]) <= 0: continue
                target = np.r_[((gt[t,:2]+gt[t,2:]/2-origin)/side-b[:2])/b[2:], np.log(gt[t,2:]/side/b[2:])]
                # Finite local-action oracle, including zero and projected GT action.
                # Used only for diagnostics; none of these actions update tracker.state.
                delta = np.concatenate((grid, np.clip(target, -limits, limits)[None]), axis=0)
                boxes = np.concatenate((b[:2]+delta[:,:2]*b[2:], b[2:]*np.exp(delta[:,2:])), axis=-1)
                scores = ious(world(boxes), gt[t]); baseline = float(ious([output], gt[t])[0])
                row = dict(sequence=m['name'], domain=m['domain'], frame=t, baseline=baseline,
                           oracle_gain=float(scores.max()-baseline), projected_gt_gain=float(scores[-1]-baseline),
                           reachable=bool((np.abs(target)<=limits).all()), clipped_coordinates=int((np.abs(target)>limits).sum()))
                rows.append(row)
                if t % 5 == 0:
                    cache.append(dict(**row, features=capture['feature'].clone(), box=capture['box'].clone(),
                                      gt=torch.tensor(gt[t], dtype=torch.float32), origin=torch.tensor(origin, dtype=torch.float32),
                                      side=side, image_hw=image.shape[:2], raw_target=torch.tensor(target, dtype=torch.float32)))
            print('LOCAL_SEQUENCE_DONE', m['name'], len(rows), round(time.time()-start, 1), flush=True)
    finally:
        handle.remove(); cap.__exit__()
    def summarize(group):
        return dict(n=len(group), mean_oracle_gain=float(np.mean([r['oracle_gain'] for r in group])),
                    correctable_005=sum(r['oracle_gain']>.05 for r in group),
                    mean_projected_gt_gain=float(np.mean([r['projected_gt_gain'] for r in group])),
                    projected_gt_harm=sum(r['projected_gt_gain']<-.01 for r in group),
                    reachable_rate=float(np.mean([r['reachable'] for r in group])))
    summary = dict(all=summarize(rows), domains={d:summarize([r for r in rows if r['domain']==d]) for d in sorted(set(r['domain'] for r in rows))},
                   max_zero_pixel_error=max_error, elapsed=time.time()-start, cached=len(cache), optimizer_steps=0,
                   note='8 previously inspected train-source sequences; finite GT oracle is not learnability or closed-loop AUC evidence. Original JQ Head response trajectory.')
    torch.save(dict(rows=cache, summary=summary), dest/'local_features.pt')
    (dest/'report.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print('LOCAL_AUDIT_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    with torch.no_grad(): main()
