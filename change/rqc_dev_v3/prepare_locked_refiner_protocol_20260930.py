"""Lock development/final roles; no inference, fitting, or metric calculation."""
import argparse,json,random,hashlib
from pathlib import Path
import numpy as np


def load_role(folder,role,allow_final=False):
    if role not in ('fit','dev','final'):raise ValueError('Unknown role')
    if role=='final' and not allow_final:
        raise PermissionError('Final evaluation requires explicit authorization; never use for training/tuning')
    folder=Path(folder);contract=json.loads((folder/'protocol.json').read_text())
    path=folder/(role+'_manifest.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==contract['hashes'][role]
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.root).resolve();dest=Path(args.output)
    if dest.exists():raise FileExistsError(str(dest))
    previous=root/'change/rqc_pr_20260928_isolated/change/rqc_dev_v3/coverage_plan_20260929/manifest.json'
    old=json.loads(previous.read_text())
    fit=[dict(r,role='fit') for r in old if r['split']=='probe_fit']
    dev=[dict(r,role='dev') for r in old if r['split']=='probe_calibration']
    assert len(fit)==80 and len(dev)==20
    excluded=set();sources=[]
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('name'),str) and ('domain' in obj or 'path' in obj):excluded.add(obj['name'])
            if isinstance(obj.get('sequence'),str):excluded.add(obj['sequence'])
            for value in obj.values():walk(value)
        elif isinstance(obj,list):
            for value in obj:walk(value)
    for path in sorted((root/'change').rglob('*manifest*.json')):
        raw=path.read_bytes()
        try:obj=json.loads(raw.decode('utf-8-sig'))
        except (ValueError,UnicodeError) as e:
            raise RuntimeError('Cannot inspect historical manifest '+str(path)) from e
        walk(obj);sources.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    names=(root/'lib/train/data_specs/lasot_train_split.txt').read_text().splitlines()
    pools={'lasot_train':[(n,root/'data/lasot'/n.split('-')[0]/n) for n in names],
           'tnl2k_train':[(p.name,p) for p in (root/'data/tnl2k/train').iterdir() if p.is_dir()]}
    rng=random.Random(20260930);final=[];rejected=[]
    for domain,pool in pools.items():
        pool=sorted((n,p) for n,p in pool if n not in excluded);rng.shuffle(pool);accepted=[]
        for name,path in pool:
            if len(accepted)==20:break
            try:
                tnl=domain=='tnl2k_train';lang=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip()
                gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2)
                count=sum(f.suffix.lower() in ('.jpg','.jpeg','.png') for f in (path/('imgs' if tnl else 'img')).iterdir())
                assert lang and count==len(gt) and len(gt)>1 and gt.shape[1]==4
                assert np.isfinite(gt[0]).all() and (gt[0,2:]>0).all()
                if not tnl:
                    for filename in ('full_occlusion.txt','out_of_view.txt'):
                        assert np.loadtxt(path/filename,delimiter=',').size==len(gt)
                accepted.append(dict(name=name,path=str(path),domain=domain,frames=count,role='final'))
            except (OSError,ValueError,AssertionError) as e:rejected.append(dict(name=name,reason=str(e) or 'integrity failure'))
        assert len(accepted)==20
        final.extend(accepted)
    splits=dict(fit=fit,dev=dev,final=final)
    sets={role:set(str(Path(r['path']).resolve()) for r in rows) for role,rows in splits.items()}
    assert all(len(sets[k])==len(splits[k]) for k in splits)
    assert not sets['fit']&sets['dev'] and not sets['fit']&sets['final'] and not sets['dev']&sets['final']
    assert not set(r['name'] for r in final)&excluded
    dest.mkdir(parents=True);hashes={}
    for role,rows in splits.items():
        path=dest/(role+'_manifest.json');path.write_text(json.dumps(rows,indent=2))
        hashes[role]=hashlib.sha256(path.read_bytes()).hexdigest()
    contract=dict(counts={k:len(v) for k,v in splits.items()},hashes=hashes,seed=20260930,
        excluded_names=len(excluded),history_sources=sources,rejected=rejected,
        official_training_yaml_modified=False,inference_runs=0,optimizer_steps=0,
        policies=['Fit/dev reused; prior holdout retired from final role.',
                  'No final metrics before architecture/training protocol frozen and user approval.',
                  'Any later change informed by final outcomes makes this set development, not fresh final.',
                  'Final is absent from discovered manifests, NOT guaranteed absent from all historical training or undocumented analysis.',
                  'This does not fix the model or guarantee gain.'],
        comparison='Fixed old-local vs boundary-local; no third new structure or hyperparameter sweep',
        backbone_source='JQ E30 for paired representation diagnosis, not official baseline init experiment')
    (dest/'protocol.json').write_text(json.dumps(contract,indent=2))
    for role in ('fit','dev'):assert len(load_role(dest,role))==len(splits[role])
    try:load_role(dest,'final')
    except PermissionError:pass
    else:raise AssertionError('Final guard failed')
    print('LOCKED_PROTOCOL_READY',json.dumps(dict(counts=contract['counts'],excluded_names=len(excluded),
        historical_manifests=len(sources),split_disjoint=True,final_guard_pass=True,inference_runs=0,optimizer_steps=0)),flush=True)


if __name__=='__main__':main()
