"""Fixed natural-state ranking/trajectory diagnostic; no training or tuning."""
import sys,json,math,time
from pathlib import Path
import numpy as np,torch,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from audit_bq_contract_20260929 import ious,fork
from lib.test.parameter.uvltrack import parameters
from lib.test.tracker.uvltrack import UVLTrack
from lib.utils.box_ops import clip_box
NAME='baseline_base_head_rqc_jq_bs16_30ep_20260929'

def main(output_name='jq_natural_20260930',observer=None):
    dest=ROOT/'change/rqc_dev_v3'/output_name;dest.mkdir(exist_ok=False)
    manifest=Path('/root/autodl-tmp/UVLTrack-master/change/rqc_pr_20260928_isolated/change/rqc_dev_v3/coverage_plan_20260929/manifest.json')
    allseq=json.loads(manifest.read_text());chosen=[]
    for domain in ('lasot_train','tnl2k_train'):
        group=sorted([m for m in allseq if m['domain']==domain and m['split']=='probe_calibration'],key=lambda m:m['name'])
        chosen.extend(group[i] for i in np.linspace(0,len(group)-1,4,dtype=int))
    (dest/'manifest.json').write_text(json.dumps(chosen,indent=2))
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(20260930)
    params=parameters(NAME,epoch=30);params.debug=0
    tracker=UVLTrack(params,'tnl2k');capture={};control={'full':False}
    if observer is not None:observer.attach(tracker)
    def hook(m,inputs,out):
        b=inputs[0];capture.clear()
        capture.update(boxes=b['boxes'][0].cpu().numpy(),scores=out['decision_score'][0].cpu().numpy(),
            quality=out['quality'][0].cpu().numpy(),prior=b['scores'][0].cpu().numpy())
        if not control['full']:
            out=dict(out);out['decision_score']=torch.zeros_like(out['decision_score']);out['decision_score'][:,0]=1
            return out
    handle=tracker.network.rqc.register_forward_hook(hook)
    read=lambda p:cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
    rows=[];events=[];start=time.time()
    for m in chosen:
        path=Path(m['path']);tnl=m['domain']=='tnl2k_train'
        if tnl:assert path.resolve().parent==(ROOT/'data/tnl2k/train').resolve()
        else:assert m['name'] in (ROOT/'lib/train/data_specs/lasot_train_split.txt').read_text().splitlines()
        frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
        gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2);assert len(frames)==len(gt)
        visible=np.ones(len(gt),bool)
        if not tnl:
            for fn in ('full_occlusion.txt','out_of_view.txt'):visible &= np.loadtxt(path/fn,delimiter=',').reshape(-1)==0
        language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
        tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
        tracker.initialize(read(frames[0]),dict(init_bbox=gt[0].tolist(),language=language))
        event_done=False
        for t in range(1,min(401,len(frames))):
            old=np.asarray(tracker.state);side=math.ceil(math.sqrt(old[2]*old[3])*params.search_factor)
            origin=old[:2]+old[2:]/2-side/2;image=read(frames[t]);control['full']=False
            output=tracker.track(image)['target_bbox']
            b=capture['boxes'];win=int(capture['scores'].argmax());quality=capture['quality'].copy();prior=capture['prior'].copy()
            boxes=np.array([clip_box(np.r_[origin+(x[:2]-x[2:]/2)*side,x[2:]*side].tolist(),*image.shape[:2],margin=10) for x in b])
            np.testing.assert_allclose(boxes[0],output,rtol=0,atol=.002)
            valid=visible[t] and np.isfinite(gt[t]).all() and min(gt[t,2:])>0
            if not valid:continue
            q=ious(boxes,gt[t]);qwin=int(quality.argmax())
            if observer is not None:observer.record(m,t,q)
            rows.append(dict(sequence=m['name'],domain=m['domain'],frame=t,winner=win,
                iou=q.tolist(),quality=quality.tolist(),prior=prior.tolist(),
                gain=float(q[win]-q[0]),quality_only_gain=float(q[qwin]-q[0]),
                oracle_gain=float(q.max()-q[0]),quality_mae=float(np.abs(quality-q).mean())))
            if win!=0 and not event_done and t+21<len(frames):
                event_done=True;base=fork(tracker);single=fork(tracker);full=fork(tracker)
                single.state=boxes[win].tolist();full.state=boxes[win].tolist()
                sg=[];fg=[]
                for j in range(t+1,t+22):
                    frame=read(frames[j]);control['full']=False
                    a=base.track(frame)['target_bbox'];s=single.track(frame)['target_bbox']
                    control['full']=True;f=full.track(frame)['target_bbox']
                    if visible[j] and np.isfinite(gt[j]).all() and min(gt[j,2:])>0:
                        sg.append(float(ious([s],gt[j])[0]-ious([a],gt[j])[0]));fg.append(float(ious([f],gt[j])[0]-ious([a],gt[j])[0]))
                events.append(dict(sequence=m['name'],frame=t,initial_gain=float(q[win]-q[0]),
                    single_future=sg,full_future=fg,cumulative_single=sum(sg),cumulative_full=sum(fg)))
                del base,single,full
        print('SEQUENCE_DONE',m['name'],'frames',len(rows),'seconds',round(time.time()-start,1),flush=True)
    handle.remove()
    if observer is not None:observer.finish(dest)
    changed=[r for r in rows if r['winner']!=0];harm=[r for r in changed if r['gain']<-.01]
    mean=lambda items:float(np.mean(items)) if items else None
    summary=dict(samples=len(rows),sequences=len(chosen),elapsed=time.time()-start,
        changes=len(changed),beneficial=sum(r['gain']>.01 for r in changed),harmful=len(harm),
        severe=sum(r['gain']<-.2 for r in changed),mean_gain=mean([r['gain'] for r in rows]),
        oracle_gain=mean([r['oracle_gain'] for r in rows]),quality_only_gain=mean([r['quality_only_gain'] for r in rows]),
        correctable_frames=sum(r['oracle_gain']>.05 for r in rows),
        harmful_with_oracle_gain=sum(r['oracle_gain']>.05 for r in harm),
        quality_mae=mean([r['quality_mae'] for r in rows]),events=len(events),
        future_single_gains=[e['cumulative_single'] for e in events],future_full_gains=[e['cumulative_full'] for e in events],
        limitations='8 train-source calibration sequences, first400frames; prior model training may have seen these. Current Head baseline trajectory, not official baseline, not final benchmark or policy tuning.')
    (dest/'report.json').write_text(json.dumps(dict(summary=summary,rows=rows,events=events),indent=2,allow_nan=False))
    print('NATURAL_AUDIT_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
