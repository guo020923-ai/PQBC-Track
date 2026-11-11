"""Recollect exactly the previous frames; fail closed on trajectory mismatch."""
import os, sys, json, time, math, hashlib
from pathlib import Path
import numpy as np
import torch
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from audit_jq_natural_20260930 import NAME
from head_quality_features_20260930 import HeadQualityCapture
from boundary_local_refiner_20260930 import sample_box_boundary
from lib.test.parameter.uvltrack import parameters
from lib.test.tracker.uvltrack import UVLTrack


def main():
    os.chdir(str(ROOT));torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(42)
    source=ROOT/'change/rqc_dev_v3/local_refiner_audit_20260930_checked/local_features.pt'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    old=torch.load(str(source),map_location='cpu')['rows']
    lookup={(r['sequence'],r['frame']):r for r in old}
    assert len(lookup)==len(old)==570
    chosen=json.loads((ROOT/'change/rqc_dev_v3/jq_natural_20260930/manifest.json').read_text())
    dest=ROOT/'change/rqc_dev_v3/boundary_local_cache_20260930';dest.mkdir(exist_ok=False)
    params=parameters(NAME,epoch=30);params.debug=0;tracker=UVLTrack(params,'tnl2k')
    cap=HeadQualityCapture(tracker.network.box_head);cap.__enter__();capture={}
    def hook(module,inputs,out):
        batch=inputs[0];box=batch['boxes'][:,:1]
        boundary=sample_box_boundary(cap.maps,box[:,0])
        capture.update(box=box[0,0].cpu(),features=cap.gather(batch['indices'][:,:1])[0,0].cpu(),
                       samples=boundary['samples'][0].cpu(),valid=boundary['valid'][0].cpu())
        answer=dict(out);answer['decision_score']=torch.zeros_like(out['decision_score']);answer['decision_score'][:,0]=1
        return answer
    handle=tracker.network.rqc.register_forward_hook(hook)
    rows=[];seen=set();start=time.time();maxdiff={k:0. for k in ('box','features','origin','gt')}
    read=lambda p:cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
    try:
        for m in chosen:
            path=Path(m['path']);tnl=m['domain']=='tnl2k_train'
            if tnl:assert path.resolve().parent==(ROOT/'data/tnl2k/train').resolve()
            else:assert m['name'] in (ROOT/'lib/train/data_specs/lasot_train_split.txt').read_text().splitlines()
            frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
            gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2);assert len(frames)==len(gt)
            language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
            tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
            tracker.initialize(read(frames[0]),dict(init_bbox=gt[0].tolist(),language=language))
            for t in range(1,min(401,len(frames))):
                previous=np.asarray(tracker.state)
                side=math.ceil(math.sqrt(previous[2]*previous[3])*params.search_factor)
                origin=previous[:2]+previous[2:]/2-side/2
                image=read(frames[t]);tracker.track(image)
                key=(m['name'],t)
                if key not in lookup:continue
                reference=lookup[key]
                assert reference['domain']==m['domain'] and reference['side']==side and tuple(reference['image_hw'])==image.shape[:2]
                current=dict(box=capture['box'],features=capture['features'],origin=torch.tensor(origin,dtype=torch.float32),gt=torch.tensor(gt[t],dtype=torch.float32))
                for k,v in current.items():
                    difference=float((v-reference[k]).abs().max());maxdiff[k]=max(maxdiff[k],difference)
                    if not torch.equal(v,reference[k]):raise AssertionError(f'{key}: {k} differs by {difference}; no paired cache emitted')
                row=dict(reference);row.update(boundary_samples=capture['samples'].clone(),boundary_valid=capture['valid'].clone())
                assert torch.isfinite(row['boundary_samples']).all()
                rows.append(row);seen.add(key)
            print('BOUNDARY_SEQUENCE_DONE',m['name'],len(rows),round(time.time()-start,1),flush=True)
    finally:
        handle.remove();cap.__exit__()
    assert seen==set(lookup) and len(rows)==len(old)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    # Match original order for paired training/evaluation.
    mapped={(r['sequence'],r['frame']):r for r in rows};rows=[mapped[(r['sequence'],r['frame'])] for r in old]
    valid=torch.stack([r['boundary_valid'] for r in rows])
    summary=dict(samples=len(rows),sequences=len(chosen),source_sha256=before,source_unchanged=True,
        exact_pairing=True,max_differences=maxdiff,valid_point_rate=float(valid.float().mean()),
        missing_edge_rate=float((~valid.flatten(2).any(-1)).float().mean()),
        frames_missing_any_edge=int((~valid.flatten(2).any(-1)).any(-1).sum()),
        elapsed=time.time()-start,optimizer_steps=0,trained=False,
        note='Same previously inspected development frames, not new generalization data.')
    target=dest/'boundary_features.pt';torch.save(dict(rows=rows,summary=summary),target)
    summary.update(bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (dest/'completed.json').write_text(json.dumps(summary,indent=2))
    print('BOUNDARY_CACHE_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
