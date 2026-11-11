"""Post-clip image-coordinate diagnostic, not independent trained ablations."""
import sys,json
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust
BASE=Path(__file__).resolve().parent;dest=BASE/'relative_output_only_20260930';rows=[]
for shard in range(8):
    report=json.loads((dest/f'shard_{shard}/report.json').read_text())
    for j,row in enumerate(report['results']):
        a=np.load(dest/f'shard_{shard}/sequence_{j+1:02d}.npz');b=a['baseline'];r=a['refined'];gt=a['gt']
        cb=b[:,:2]+b[:,2:]/2;cr=r[:,:2]+r[:,2:]/2
        variants=dict(baseline=b,full=r,
            center_only=np.concatenate((cr-b[:,2:]/2,b[:,2:]),1),
            size_only=np.concatenate((cb-r[:,2:]/2,r[:,2:]),1))
        scores={};domain=row['domain'];dataset='lasot' if domain=='lasot_train' else 'tnl2k'
        for kind,boxes in variants.items():
            ov,c,n,v=calc_seq_err_robust(torch.tensor(boxes),torch.tensor(gt),dataset,
                torch.tensor(a['valid']) if dataset=='lasot' else None)
            curve=(ov[:,None]>torch.arange(0,1.05,.05,dtype=torch.float64)).double().mean(0)
            scores[kind]=dict(AUC=float(curve.mean()*100),OP50=float(curve[10]*100),OP75=float(curve[15]*100),
                Precision=float((c<=20).double().mean()*100),NormPrecision=float((n<=.2).double().mean()*100))
        rows.append(dict(sequence=row['sequence'],domain=domain,scores=scores))
for domain in ('all','lasot_train','tnl2k_train'):
    group=[r for r in rows if domain=='all' or r['domain']==domain]
    for kind in ('full','center_only','size_only'):
        delta={k:float(np.mean([r['scores'][kind][k]-r['scores']['baseline'][k] for r in group])) for k in group[0]['scores'][kind]}
        print('POSTCLIP_ACTION_DECOMPOSITION',json.dumps(dict(domain=domain,action=kind,n=len(group),delta=delta)))
print('NOTE: Post-clip coordinate factorization only; size-only center equality is by construction. No retraining, no new oracle, no inference gate.')
