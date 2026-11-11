"""Official metric functions on development trajectories, NOT official test scores."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust
BASE=Path(__file__).resolve().parent
manifest=json.loads((BASE/'locked_refiner_protocol_20260930/dev_manifest.json').read_text())
names={m['name']:m for m in manifest};rows=[]
for variant,folder in (('point','output_only_local_20260930'),('relative','relative_output_only_20260930'),('center_safe','center_safe_output_only_20260930')):
    dest=BASE/folder
    assert (dest/'report.json').exists(),'Full development evaluation incomplete'
    seen=set()
    for shard in range(8):
        report=json.loads((dest/f'shard_{shard}/report.json').read_text())
        for i,row in enumerate(report['results']):
            name=row['sequence'];assert name not in seen;seen.add(name)
            data=np.load(dest/f'shard_{shard}/sequence_{i+1:02d}.npz')
            dataset='lasot' if row['domain']=='lasot_train' else 'tnl2k'
            truth=torch.tensor(data['gt'],dtype=torch.float64)
            visible=torch.tensor(data['valid']) if dataset=='lasot' else None
            scores={}
            for kind in ('baseline','refined'):
                pred=torch.tensor(data[kind],dtype=torch.float64)
                overlap,center,norm,valid=calc_seq_err_robust(pred,truth,dataset,visible)
                # Exactly the official default: include invalid/occluded frames in denominator.
                bins=torch.arange(0.,1.05,.05,dtype=torch.float64)
                curve=(overlap[:,None]>bins).double().mean(0)
                scores[kind]=dict(AUC=float(curve.mean()*100),OP50=float(curve[10]*100),
                    OP75=float(curve[15]*100),Precision=float((center<=20).double().mean()*100),
                    NormPrecision=float((norm<=.2).double().mean()*100))
            rows.append(dict(variant=variant,sequence=name,domain=row['domain'],scores=scores))
    assert seen==set(names)
for variant in ('point','relative','center_safe'):
    for domain in ('all','lasot_train','tnl2k_train'):
        group=[r for r in rows if r['variant']==variant and (domain=='all' or r['domain']==domain)]
        result={kind:{metric:float(np.mean([r['scores'][kind][metric] for r in group]))
                      for metric in group[0]['scores'][kind]} for kind in ('baseline','refined')}
        delta={k:result['refined'][k]-result['baseline'][k] for k in result['baseline']}
        print('DEV_OFFICIAL_DEFINITION',json.dumps(dict(variant=variant,domain=domain,n=len(group),
            scores=result,delta=delta,note='Training-source development sequences; not benchmark test AUC.')))
for domain in ('all','lasot_train','tnl2k_train'):
    group=[r for r in rows if r['variant']=='center_safe' and (domain=='all' or r['domain']==domain)]
    gains=np.array([r['scores']['refined']['AUC']-r['scores']['baseline']['AUC'] for r in group])
    rng=np.random.RandomState(42);boot=gains[rng.randint(len(gains),size=(5000,len(gains)))].mean(1)
    print('DEV_AUC_INTERVAL',json.dumps(dict(domain=domain,positive_sequences=int((gains>0).sum()),
        bootstrap95=np.quantile(boot,[.025,.975]).tolist())))
    pd=np.array([r['scores']['refined']['Precision']-r['scores']['baseline']['Precision'] for r in group])
    print('PRECISION_SPREAD',json.dumps(dict(domain=domain,negative_sequences=int((pd<0).sum()),
        worse_than_half_point=int((pd<-.5).sum()),median=float(np.median(pd)),minimum=float(pd.min()),
        maximum=float(pd.max()),note='Diagnostic only, no sequence-specific inference rules.')))
