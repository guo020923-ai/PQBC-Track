"""Fixed final weights, fitting vs held sequences; no optimizer or policy tuning."""
import json, hashlib
from pathlib import Path
import torch
from local_box_refiner_20260930 import LocalBoxRefiner, decode_image_boxes, localization_loss
ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    folder=ROOT/'change/rqc_dev_v3/local_refiner_probe_20260930'
    output=folder/'mechanism_diagnosis.json'
    if output.exists():raise FileExistsError(str(output))
    rows=torch.load(str(ROOT/'change/rqc_dev_v3/local_refiner_audit_20260930_checked/local_features.pt'),map_location='cpu')['rows']
    x={k:torch.stack([r[k] for r in rows]).float() for k in ('features','box','origin','gt','raw_target')}
    side=torch.tensor([r['side'] for r in rows],dtype=torch.float32)
    hw=torch.tensor([r['image_hw'] for r in rows],dtype=torch.float32)
    baseline=decode_image_boxes(x['box'],x['origin'],side,hw)
    limit=torch.tensor([.1,.1,__import__('math').log(1.2),__import__('math').log(1.2)])
    def decode(delta):
        boxes=torch.cat((x['box'][:,:2]+delta[:,:2]*x['box'][:,2:],x['box'][:,2:]*delta[:,2:].exp()),-1)
        return decode_image_boxes(boxes,x['origin'],side,hw)
    reports=[]
    for fold in range(4):
        path=folder/f'fold{fold}_final.pt';before=hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint=torch.load(str(path),map_location='cpu');model=LocalBoxRefiner().eval()
        model.load_state_dict(checkpoint['model'],strict=True)
        delta=model(x['features'],x['box'])['delta']
        center=delta.clone();center[:,2:]=0
        size=delta.clone();size[:,:2]=0
        variants={'full':decode(delta),'center_only':decode(center),'size_only':decode(size)}
        held=checkpoint['held_sequences']
        for split in ('fit','held'):
            mask=torch.tensor([(r['sequence'] in held)==(split=='held') for r in rows])
            for domain in ('all','lasot_train','tnl2k_train'):
                keep=mask & torch.tensor([domain=='all' or r['domain']==domain for r in rows])
                metrics={}
                base_loss,_=localization_loss(baseline,baseline,x['gt'],hw,keep)
                for name,boxes in variants.items():
                    loss,stats=localization_loss(boxes,baseline,x['gt'],hw,keep)
                    metrics[name]=dict(loss=float(loss),loss_delta=float(loss-base_loss),**{k:float(v) for k,v in stats.items()})
                coordinate={}
                for j,name in enumerate(('cx','cy','logw','logh')):
                    p=delta[keep,j]/limit[j];target=x['raw_target'][keep,j]/limit[j]
                    # Diagnostic dead zone only, not an inference threshold.
                    active=target.abs()>.05
                    wrong=(p*target<0)&active
                    corr=None
                    if p.std()>1e-8 and target.clamp(-1,1).std()>1e-8:
                        corr=float(torch.corrcoef(torch.stack((p,target.clamp(-1,1))))[0,1])
                    coordinate[name]=dict(pred_mean=float(p.mean()),pred_abs=float(p.abs().mean()),
                        saturation_rate=float((p.abs()>=.95).float().mean()),
                        target_outside_limit=float((target.abs()>1).float().mean()),
                        wrong_direction_rate=float(wrong.sum()/active.sum()) if active.any() else None,
                        direction_samples=int(active.sum()),correlation_to_bounded_target=corr)
                reports.append(dict(fold=fold,split=split,domain=domain,n=int(keep.sum()),metrics=metrics,coordinates=coordinate))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    summary={}
    for split in ('fit','held'):
        summary[split]={}
        for domain in ('all','lasot_train','tnl2k_train'):
            group=[r for r in reports if r['split']==split and r['domain']==domain];n=sum(r['n'] for r in group)
            aggregate={}
            for variant in variants:
                aggregate[variant]={k:sum(r['metrics'][variant][k]*r['n'] for r in group)/n
                    for k in ('iou_gain','loss_delta','beneficial_rate','harmful_rate','severe_harmful_rate')}
            coords={}
            for name in ('cx','cy','logw','logh'):
                coords[name]={k:sum(r['coordinates'][name][k]*r['n'] for r in group)/n
                    for k in ('pred_mean','pred_abs','saturation_rate','target_outside_limit')}
                count=sum(r['coordinates'][name]['direction_samples'] for r in group)
                coords[name]['wrong_direction_rate']=sum((r['coordinates'][name]['wrong_direction_rate'] or 0)*r['coordinates'][name]['direction_samples'] for r in group)/max(count,1)
            summary[split][domain]=dict(n_evaluations=n,metrics=aggregate,coordinates=coords)
    result=dict(summary=summary,folds=reports,optimizer_steps=0,weights_unchanged=True,
        limitations='Fit includes each sequence three times across different models; held once. Previously inspected 8-sequence development subset. Center/size removal is posthoc diagnostic, not a separately trained model or selection recommendation.')
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('LOCAL_DIAGNOSIS_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
