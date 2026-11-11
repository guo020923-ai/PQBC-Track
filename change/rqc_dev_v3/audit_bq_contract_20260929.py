"""Bounded train-source diagnostic; no fitting or production file changes."""
import argparse, copy, inspect, json, math, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import lib.test.tracker.uvltrack as tracker_module
from lib.test.parameter.uvltrack import parameters
from lib.utils.box_ops import clip_box
NAME='baseline_base_frozen_head_rqc_bq_bs16_30ep_20260929'

def ious(boxes, gt, crop=False):
    a=np.asarray(boxes,dtype=float).copy();b=np.asarray(gt,dtype=float).copy()
    a[:,2:]+=a[:,:2];b[2:]+=b[:2]
    if crop:a=np.clip(a,0,1);b=np.clip(b,0,1)
    inter=np.maximum(0,np.minimum(a[:,2:],b[2:])-np.maximum(a[:,:2],b[:2])).prod(-1)
    area=np.maximum(0,a[:,2:]-a[:,:2]).prod(-1)
    return inter/np.maximum(area+np.maximum(0,b[2:]-b[:2]).prod()-inter,1e-12)

def fork(t):
    other=copy.copy(t)
    shared={'network','params','cfg','tokenizer','preprocessor'}
    for k,v in t.__dict__.items():
        if k not in shared:setattr(other,k,copy.deepcopy(v))
    return other

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(42)
    manifest=json.loads(Path(args.manifest).read_text())
    selected=[]
    for domain in ('lasot_train','tnl2k_train'):
        group=sorted([m for m in manifest if m['domain']==domain and m['split']=='probe_calibration'],key=lambda m:m['name'])
        assert len(group)>=2
        selected.extend(group[:2])
    (dest/'manifest.json').write_text(json.dumps(selected,indent=2))
    # Counterfactual variant in memory ONLY: update reference box and confidence
    # from deployed RQC winner. No GT used; interval/threshold are unchanged.
    source=inspect.getsource(tracker_module)
    old='prompt_box_net = pred_boxes[baseline_index]\n        score = (pred_cls * pred_cont)[baseline_index]'
    assert source.count(old)==1
    namespace=dict(tracker_module.__dict__)
    exec(compile(source.replace(old,'prompt_box_net = pred_boxes[selected_index]\n        score = (pred_cls * pred_cont)[selected_index]'),'<bq_prompt_counterfactual>','exec'),namespace)
    aligned_class=namespace['UVLTrack']
    params=parameters(NAME,epoch=30);params.debug=0
    tracker=tracker_module.UVLTrack(params,'tnl2k');capture={}
    def hook(module,inputs,out):
        capture.clear();capture.update(boxes=inputs[0]['boxes'][0].cpu().numpy(),winner=int(out['decision_score'][0].argmax()))
    handle=tracker.network.rqc.register_forward_hook(hook)
    read=lambda f:cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB)
    rows=[];events=[];start=time.time()
    for m in selected:
        path=Path(m['path']);tnl=m['domain']=='tnl2k_train'
        if tnl:assert path.resolve().parent==(ROOT/'data/tnl2k/train').resolve()
        else:assert m['name'] in (ROOT/'lib/train/data_specs/lasot_train_split.txt').read_text().splitlines()
        frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.png','.jpeg'))
        gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2);assert len(gt)==len(frames)
        language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
        visible=np.ones(len(gt),dtype=bool)
        if not tnl:
            for fn in ('full_occlusion.txt','out_of_view.txt'):visible &= np.loadtxt(path/fn,delimiter=',').reshape(-1)==0
        tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
        tracker.initialize(read(frames[0]),dict(init_bbox=gt[0].tolist(),language=language))
        event_done=False
        for t in range(1,min(201,len(frames))):
            before=fork(tracker) if not event_done else None
            previous=np.asarray(tracker.state);side=math.ceil(math.sqrt(previous[2]*previous[3])*params.search_factor)
            origin=previous[:2]+previous[2:]/2-side/2
            image=read(frames[t]);result=tracker.track(image)['target_bbox']
            boxes=capture['boxes'];winner=capture['winner']
            local=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,2:]),-1)
            world=np.array([clip_box(np.r_[origin+b[:2]*side,b[2:]*side].tolist(),*image.shape[:2],margin=10) for b in local])
            np.testing.assert_allclose(world[winner],result,atol=.002,rtol=0)
            if visible[t] and np.isfinite(gt[t]).all() and min(gt[t,2:])>0:
                target=np.r_[(gt[t,:2]-origin)/side,gt[t,2:]/side]
                q=ious(world,gt[t]);c=ious(local,target,True);u=ious(local,target)
                rows.append(dict(sequence=m['name'],domain=m['domain'],frame=t,winner=winner,
                    world_gain=float(q[winner]-q[0]),crop_gain=float(c[winner]-c[0]),
                    raw_gain=float(u[winner]-u[0]),oracle_flip=int(q.argmax())!=int(c.argmax()),
                    label_mae=float(np.abs(c-q).mean()),world_iou=q.tolist(),crop_iou=c.tolist()))
            if not event_done and winner!=0 and t+21<len(frames):
                event_done=True;a=fork(before);b=fork(before);b.__class__=aligned_class
                gains=[]
                for j in range(t,t+22):
                    frame=read(frames[j]);aa=a.track(frame)['target_bbox'];bb=b.track(frame)['target_bbox']
                    if visible[j] and np.isfinite(gt[j]).all() and min(gt[j,2:])>0:
                        gains.append(float(ious([bb],gt[j])[0]-ious([aa],gt[j])[0]))
                events.append(dict(sequence=m['name'],frame=t,horizon=22,valid_frames=len(gains),
                    cumulative_aligned_minus_current=float(sum(gains)),per_frame=gains))
                del a,b
        print('SEQUENCE_DONE',m['name'],'rows',len(rows),'seconds',round(time.time()-start,1),flush=True)
    handle.remove()
    assert rows
    summary=dict(samples=len(rows),sequences=len(selected),elapsed=time.time()-start,
        oracle_flip_rate=float(np.mean([r['oracle_flip'] for r in rows])),
        label_mae=float(np.mean([r['label_mae'] for r in rows])),
        crop_positive_world_negative=sum(r['crop_gain']>.01 and r['world_gain']<-.01 for r in rows),
        crop_gain=float(np.mean([r['crop_gain'] for r in rows])),world_gain=float(np.mean([r['world_gain'] for r in rows])),
        prompt_events=len(events),prompt_cumulative_gains=[e['cumulative_aligned_minus_current'] for e in events],
        note='Train-source calibration subset, possibly seen during BQ training; diagnostic only, no generalization claim. No training or policy deployment.')
    (dest/'report.json').write_text(json.dumps(dict(summary=summary,rows=rows,events=events),indent=2,allow_nan=False))
    print('AUDIT_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
