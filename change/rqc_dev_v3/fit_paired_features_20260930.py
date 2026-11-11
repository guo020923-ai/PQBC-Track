"""Fixed four-fold sequence-held-out linear quality probe; no tracker training.

Both views: train-only standardized PCA32 + identical standardized scalar18,
ridge lambda=1 with unpenalized intercept, clipped IoU prediction. No tuning.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]

def transform(train,test,dim=32):
    mean=train.mean(0);std=train.std(0);std=np.maximum(std,1e-6)
    x=(train-mean)/std;y=(test-mean)/std
    _,_,vh=np.linalg.svd(x,full_matrices=False)
    basis=vh[:dim].T
    xp=x@basis;yp=y@basis
    scale=np.maximum(xp.std(0),1e-6)
    return xp/scale,yp/scale

def metrics(q,pred,prior):
    baseline=q[:,0];r={}
    for name,score in [('quality_only',pred),('fixed_product',pred*prior)]:
        w=score.argmax(-1);gain=q[np.arange(len(q)),w]-baseline
        r[name]=dict(mean_gain=float(gain.mean()),changed=int((w!=0).sum()),
            beneficial=int((gain>.01).sum()),harmful=int((gain<-.01).sum()),severe=int((gain<-.2).sum()))
    r['mae']=float(np.abs(pred-q).mean())
    return r

def main():
    start=time.time();torch.set_num_threads(1)
    source=ROOT/'change/rqc_dev_v3/jq_paired_features_20260930/paired_features.pt'
    dest=ROOT/'change/rqc_dev_v3/jq_paired_probe_20260930';dest.mkdir(exist_ok=False)
    rows=torch.load(str(source),map_location='cpu')['rows']
    seq=np.array([r['sequence'] for r in rows]);domain=np.array([r['domain'] for r in rows])
    groups={d:sorted(set(seq[domain==d])) for d in sorted(set(domain))}
    assert all(len(v)==4 for v in groups.values())
    arrays={k:torch.stack([r[k] for r in rows]).numpy().astype(np.float64) for k in ('old','new','scalar','iou','prior')}
    assert np.allclose(arrays['old'][...,-18:],arrays['scalar'])
    predictions={name:np.zeros_like(arrays['iou']) for name in ('old','head')}
    folds=[]
    for fold in range(4):
        held=[v[fold] for v in groups.values()];test=np.isin(seq,held);train=~test
        scalar_train=arrays['scalar'][train].reshape(-1,18);scalar_test=arrays['scalar'][test].reshape(-1,18)
        sm=scalar_train.mean(0);ss=np.maximum(scalar_train.std(0),1e-6)
        st=(scalar_train-sm)/ss;sv=(scalar_test-sm)/ss
        y=arrays['iou'][train].reshape(-1)
        # Equal sequence weight, avoiding long-sequence dominance in regression.
        counts={s:int((seq[train]==s).sum()) for s in set(seq[train])}
        weight=np.repeat([1/counts[s] for s in seq[train]],10);weight/=weight.mean()
        for name,features in [('old',arrays['old'][...,:-18]),('head',arrays['new'])]:
            tr=features[train].reshape(-1,features.shape[-1]);te=features[test].reshape(-1,features.shape[-1])
            x,z=transform(tr,te)
            x=np.column_stack((x,st,np.ones(len(x))));z=np.column_stack((z,sv,np.ones(len(z))))
            assert x.shape[1]==51
            reg=np.eye(51);reg[-1,-1]=0
            coef=np.linalg.solve(x.T@(x*weight[:,None])+reg,x.T@(y*weight))
            pred=np.clip(z@coef,0,1).reshape(-1,10)
            predictions[name][test]=pred
            folds.append(dict(fold=fold,view=name,held_sequences=held,train_frames=int(train.sum()),
                held_frames=int(test.sum()),metrics=metrics(arrays['iou'][test],pred,arrays['prior'][test])))
    result={}
    for name,pred in predictions.items():
        byseq={s:metrics(arrays['iou'][seq==s],pred[seq==s],arrays['prior'][seq==s]) for s in sorted(set(seq))}
        result[name]=dict(overall=metrics(arrays['iou'],pred,arrays['prior']),sequences=byseq,
            domains={d:metrics(arrays['iou'][domain==d],pred[domain==d],arrays['prior'][domain==d]) for d in groups},
            sequence_macro={rule:float(np.mean([v[rule]['mean_gain'] for v in byseq.values()])) for rule in ('quality_only','fixed_product')})
    delta={rule:[result['head']['sequences'][s][rule]['mean_gain']-result['old']['sequences'][s][rule]['mean_gain'] for s in sorted(set(seq))] for rule in ('quality_only','fixed_product')}
    report=dict(results=result,folds=folds,paired_sequence_deltas=delta,elapsed_seconds=time.time()-start,
        protocol='PCA32+same18 scalars; ridge1; four fixed leave-two-sequence-out folds; no parameter search',
        limitations='Small previously inspected train-source subset; old projection pretrained by JQ, Head features also pretrained; linear probe does not establish nonlinear quality capacity or closed-loop AUC. Not formal training.')
    (dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('PAIRED_PROBE_COMPLETE',json.dumps({k:dict(overall=v['overall'],macro=v['sequence_macro']) for k,v in result.items()}),flush=True)
    print('PAIRED_SEQUENCE_DELTAS',json.dumps(delta),flush=True)

if __name__=='__main__':main()
