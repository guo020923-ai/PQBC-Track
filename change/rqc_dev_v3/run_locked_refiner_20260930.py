"""Executable paired-input experiment. Final holdout is never loaded here."""
import argparse,os,sys,json,time,math,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prepare_locked_refiner_protocol_20260930 import load_role
from local_box_refiner_20260930 import LocalBoxRefiner,decode_image_boxes,localization_loss
from boundary_local_refiner_20260930 import BoundaryLocalRefiner,sample_box_boundary


def collect(args):
    import cv2
    from head_quality_features_20260930 import HeadQualityCapture
    from audit_jq_natural_20260930 import NAME
    from lib.test.parameter.uvltrack import parameters
    from lib.test.tracker.uvltrack import UVLTrack
    manifest=load_role(args.protocol,'fit')+load_role(args.protocol,'dev')
    if args.smoke:
        manifest=[next(r for r in manifest if r['role']==s and r['domain']==d)
                  for s in ('fit','dev') for d in ('lasot_train','tnl2k_train')]
    checkpoint=ROOT/'checkpoints/train/uvltrack'/NAME/'UVLTrack_ep0030.pth.tar'
    identity=dict(manifest=manifest,smoke=args.smoke,samples_per_sequence=40,tracker=NAME,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_source_hashes={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                              for n in ('head_quality_features_20260930.py','boundary_local_refiner_20260930.py')})
    dest=Path(args.output)
    if args.resume:assert json.loads((dest/'identity.json').read_text())==identity
    else:dest.mkdir(parents=True,exist_ok=False);(dest/'identity.json').write_text(json.dumps(identity,indent=2))
    if (dest/'completed.json').exists():raise FileExistsError('Already complete')
    cv2.setNumThreads(1);torch.manual_seed(42)
    params=parameters(NAME,epoch=30);params.debug=0;tracker=UVLTrack(params,'tnl2k')
    cap=HeadQualityCapture(tracker.network.box_head);cap.__enter__();capture={}
    def hook(module,inputs,out):
        batch=inputs[0];box=batch['boxes'][:,0]
        capture.update(box=box[0].cpu(),features=cap.gather(batch['indices'][:,:1])[0,0].cpu())
        boundary=sample_box_boundary(cap.maps,box)
        capture.update(boundary_samples=boundary['samples'][0].cpu(),boundary_valid=boundary['valid'][0].cpu())
        result=dict(out);result['decision_score']=torch.zeros_like(out['decision_score']);result['decision_score'][:,0]=1
        return result
    handle=tracker.network.rqc.register_forward_hook(hook);allrows=[];tracked=0;start=time.time()
    read=lambda p:cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
    try:
        for i,m in enumerate(manifest):
            shard=dest/f'sequence_{i:03d}.pt'
            if shard.exists():
                obj=torch.load(str(shard),map_location='cpu');assert obj['sequence']==m
                allrows.extend(obj['rows']);tracked+=obj['tracked'];print('RESUME_SKIP',i,flush=True);continue
            path=Path(m['path']);tnl=m['domain']=='tnl2k_train'
            if tnl:assert path.resolve().parent==(ROOT/'data/tnl2k/train').resolve()
            else:assert m['name'] in (ROOT/'lib/train/data_specs/lasot_train_split.txt').read_text().splitlines()
            frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
            gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2);assert len(gt)==len(frames)
            visible=np.ones(len(gt),bool)
            if not tnl:
                for n in ('full_occlusion.txt','out_of_view.txt'):visible &= np.loadtxt(path/n,delimiter=',').reshape(-1)==0
            end=min(25,len(frames)) if args.smoke else len(frames)
            schedule=set(np.linspace(1,end-1,min(40,end-1),dtype=int).tolist())
            language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
            tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
            tracker.initialize(read(frames[0]),dict(init_bbox=gt[0].tolist(),language=language));rows=[]
            for t in range(1,end):
                previous=np.asarray(tracker.state);side=math.ceil(math.sqrt(previous[2]*previous[3])*params.search_factor)
                origin=previous[:2]+previous[2:]/2-side/2
                image=read(frames[t]);actual=tracker.track(image)['target_bbox']
                if t%500==0:print('FRAME',i,t,end-1,'elapsed',round(time.time()-start,1),flush=True)
                if t not in schedule or not visible[t] or not np.isfinite(gt[t]).all() or min(gt[t,2:])<=0:continue
                row={k:v.clone() for k,v in capture.items()}
                row.update(sequence=m['name'],domain=m['domain'],role=m['role'],frame=t,
                    gt=torch.tensor(gt[t],dtype=torch.float32),origin=torch.tensor(origin,dtype=torch.float32),
                    side=side,image_hw=image.shape[:2])
                decoded=decode_image_boxes(row['box'][None],row['origin'][None],torch.tensor([side]),torch.tensor([image.shape[:2]]))[0]
                np.testing.assert_allclose(decoded.numpy(),actual,atol=.002,rtol=0)
                rows.append(row)
            if not rows:raise RuntimeError('No valid scheduled frames: '+m['name'])
            obj=dict(sequence=m,rows=rows,tracked=end-1);tmp=shard.with_suffix('.partial')
            torch.save(obj,tmp);tmp.replace(shard)
            tracked+=end-1;allrows.extend(rows)
            print('SEQUENCE_DONE',i+1,len(manifest),m['name'],'samples',len(rows),'tracked',tracked,'elapsed',round(time.time()-start,1),flush=True)
    finally:handle.remove();cap.__exit__()
    torch.save(dict(rows=allrows,identity=identity),dest/'paired_cache.pt')
    report=dict(sequences=len(manifest),samples=len(allrows),tracked_frames=tracked,elapsed=time.time()-start,trained=False,smoke=args.smoke)
    (dest/'completed.json').write_text(json.dumps(report,indent=2));print('CACHE_COMPLETE',json.dumps(report),flush=True)


def train(args):
    source=Path(args.cache);cache=torch.load(str(source),map_location='cpu');assert not cache['identity']['smoke']
    rows=cache['rows'];allowed=load_role(args.protocol,'fit')+load_role(args.protocol,'dev')
    assert {(r['sequence'],r['role']) for r in rows}=={(r['name'],r['role']) for r in allowed}
    fit=[i for i,r in enumerate(rows) if r['role']=='fit'];dev=torch.tensor([i for i,r in enumerate(rows) if r['role']=='dev'])
    groups={s:[i for i in fit if rows[i]['sequence']==s] for s in sorted(set(rows[i]['sequence'] for i in fit))}
    assert len(groups)==80 and len(set(rows[i]['sequence'] for i in dev.tolist()))==20
    x={k:torch.stack([r[k] for r in rows]) for k in ('features','boundary_samples','boundary_valid','box','gt','origin')}
    side=torch.tensor([r['side'] for r in rows]);hw=torch.tensor([r['image_hw'] for r in rows])
    base=decode_image_boxes(x['box'],x['origin'],side,hw)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    protocol=dict(epochs=20,lr=1e-4,batch=64,batches_per_epoch=50,seed=20260930,
        sampling='uniform sequence then uniform cached frame',loss='2 GIoU + 5 image-normalized L1',
        checkpoint='last epoch only',final_evaluated=False,cache_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2));results={}
    for variant in ('point','boundary'):
        torch.manual_seed(20260930)
        model=LocalBoxRefiner() if variant=='point' else BoundaryLocalRefiner()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
        rng=np.random.RandomState(20260930);names=list(groups);logs=[]
        def decode(idx):
            if variant=='point':out=model(x['features'][idx],x['box'][idx])
            else:out=model(dict(samples=x['boundary_samples'][idx],valid=x['boundary_valid'][idx],boxes=x['box'][idx]))
            return decode_image_boxes(out['boxes'],x['origin'][idx],side[idx],hw[idx])
        for epoch in range(1,21):
            for step in range(50):
                idx=torch.tensor([rng.choice(groups[names[j]]) for j in rng.randint(len(names),size=64)])
                opt.zero_grad(set_to_none=True);loss,_=localization_loss(decode(idx),base[idx],x['gt'][idx],hw[idx],torch.ones(len(idx),dtype=torch.bool))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                opt.step()
            with torch.no_grad():
                prediction=decode(dev);loss,stats=localization_loss(prediction,base[dev],x['gt'][dev],hw[dev],torch.ones(len(dev),dtype=torch.bool))
            record=dict(variant=variant,epoch=epoch,loss=float(loss),**{k:float(v) for k,v in stats.items()})
            logs.append(record);print('DEV',json.dumps(record),flush=True)
        bysequence={}
        with torch.no_grad():
            for name in sorted(set(rows[i]['sequence'] for i in dev.tolist())):
                idx=torch.tensor([i for i in dev.tolist() if rows[i]['sequence']==name])
                _,stats=localization_loss(decode(idx),base[idx],x['gt'][idx],hw[idx],torch.ones(len(idx),dtype=torch.bool))
                bysequence[name]={k:float(v) for k,v in stats.items()}
        torch.save(dict(model=model.state_dict(),protocol=protocol,variant=variant),dest/(variant+'_ep20.pt'))
        results[variant]=dict(logs=logs,sequences=bysequence,macro_gain=sum(v['iou_gain'] for v in bysequence.values())/len(bysequence))
    (dest/'report.json').write_text(json.dumps(results,indent=2));print('TRAIN_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','train']);p.add_argument('--protocol',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cache');p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();os.chdir(str(ROOT));torch.set_num_threads(1)
    if args.action=='collect':
        with torch.no_grad():collect(args)
    else:train(args)
