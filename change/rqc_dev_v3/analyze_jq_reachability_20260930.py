"""Read saved natural audit only. GT is oracle analysis, never inference input."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'change/rqc_dev_v3/jq_natural_20260930/report.json'
report=json.loads(source.read_text());rows=report['rows']
def summarize(items):
    q=np.array([r['iou'] for r in items]);p=np.array([r['prior'] for r in items]);pred=np.array([r['quality'] for r in items])
    current=np.array([r['winner'] for r in items]);idx=np.arange(len(items))
    oracle=q.argmax(-1);product=(p*q).argmax(-1)
    base=q[:,0];best=q[idx,oracle]-base;prod=q[idx,product]-base;actual=q[idx,current]-base
    correct=best>.05
    changed=current!=0;harm=actual<-.01;good=actual>.01
    # Compare ideal quality under current product rule versus unrestricted oracle.
    return dict(frames=len(items),correctable_gt005=int(correct.sum()),
        oracle_mean_gain=float(best.mean()),perfect_quality_product_mean_gain=float(prod.mean()),
        perfect_quality_product_changed=int((product!=0).sum()),
        perfect_quality_product_harmful=int((prod<-.01).sum()),
        perfect_quality_product_recovers_gt005=int((prod>.05).sum()),
        correctable_but_product_stays_top1=int((correct&(product==0)).sum()),
        actual_mean_gain=float(actual.mean()),actual_changed=int(changed.sum()),
        actual_beneficial=int(good.sum()),actual_harmful=int(harm.sum()),
        harmed_when_ideal_product_stays_top1=int((harm&(product==0)).sum()),
        harmed_when_ideal_product_changes=int((harm&(product!=0)).sum()),
        quality_pair_sign_accuracy_on_meaningful_pairs=float(((pred[:,1:]-pred[:,:1]>0)==(q[:,1:]-q[:,:1]>0))[np.abs(q[:,1:]-q[:,:1])>.05].mean()))
out=dict(overall=summarize(rows),domains={d:summarize([r for r in rows if r['domain']==d]) for d in sorted(set(r['domain'] for r in rows))},
    limitations='Saved 8 train-source trajectories only; perfect quality is GT oracle, not a learned model or AUC estimate. No coefficient search or fitting.')
dest=source.parent/'reachability.json'
with dest.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out),flush=True)
