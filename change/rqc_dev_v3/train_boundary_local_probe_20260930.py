"""Fixed paired boundary-input experiment. No sweep or best-epoch selection."""
import json,time,hashlib
from pathlib import Path
import torch
from local_box_refiner_20260930 import LocalBoxRefiner,decode_image_boxes,localization_loss
from boundary_local_refiner_20260930 import BoundaryLocalRefiner
ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    source=ROOT/'change/rqc_dev_v3/boundary_local_cache_20260930/boundary_features.pt'
    olddir=ROOT/'change/rqc_dev_v3/local_refiner_probe_20260930'
    dest=ROOT/'change/rqc_dev_v3/boundary_local_probe_20260930'
    original=json.loads((olddir/'report.json').read_text())
    cache=torch.load(str(source),map_location='cpu');rows=cache['rows']
    assert cache['summary']['exact_pairing'] and len(rows)==570
    groups={d:sorted(set(r['sequence'] for r in rows if r['domain']==d)) for d in sorted(set(r['domain'] for r in rows))}
    assert len(groups)==2 and all(len(v)==4 for v in groups.values())
    data={k:torch.stack([r[k] for r in rows]).float() for k in ('boundary_samples','box','origin','gt')}
    data['valid']=torch.stack([r['boundary_valid'] for r in rows]).bool()
    data['side']=torch.tensor([r['side'] for r in rows],dtype=torch.float32)
    data['hw']=torch.tensor([r['image_hw'] for r in rows],dtype=torch.float32)
    assert all(torch.isfinite(v).all() for v in data.values())
    protocol=dict(epochs=20,lr=1e-4,batch_size=64,folds=4,seed=42,checkpoint_rule='final epoch only',
                  parameters=20076,old_parameters=19332,action_loss_limits_unchanged=True,
                  paired_batch_order=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  warning='Previously inspected development data. Input projection and 3.85% capacity difference prevent pure feature-only causal attribution.')
    for k in ('epochs','lr','batch_size','folds','seed','checkpoint_rule'):
        assert protocol[k]==original['protocol'][k]
    dest.mkdir(exist_ok=False);(dest/'protocol.json').write_text(json.dumps(protocol,indent=2))
    base=decode_image_boxes(data['box'],data['origin'],data['side'],data['hw'])
    def decode(model,idx):
        result=model(dict(samples=data['boundary_samples'][idx],valid=data['valid'][idx],boxes=data['box'][idx]))
        return decode_image_boxes(result['boxes'],data['origin'][idx],data['side'][idx],data['hw'][idx])
    def evaluate(boxes,idx):
        loss,stats=localization_loss(boxes,base[idx],data['gt'][idx],data['hw'][idx],torch.ones(len(idx),dtype=torch.bool))
        return dict(loss=float(loss),**{k:float(v) for k,v in stats.items()})
    predictions=torch.empty_like(base);logs=[];fits=[];start=time.time();steps=0
    for fold in range(4):
        held=[s[fold] for s in groups.values()]
        old=torch.load(str(olddir/f'fold{fold}_final.pt'),map_location='cpu')
        assert old['held_sequences']==held
        train=torch.tensor([i for i,r in enumerate(rows) if r['sequence'] not in held])
        val=torch.tensor([i for i,r in enumerate(rows) if r['sequence'] in held])
        # Preserve the old sampler RNG state after its model initialization.
        torch.manual_seed(42+fold);reference=LocalBoxRefiner();sampler_state=torch.get_rng_state();del reference
        torch.manual_seed(42+fold);model=BoundaryLocalRefiner();torch.set_rng_state(sampler_state)
        assert sum(p.numel() for p in model.parameters())==20076
        with torch.no_grad():assert torch.equal(decode(model,val),base[val])
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
        for epoch in range(1,21):
            model.train();order=train[torch.randperm(len(train))]
            for idx in order.split(64):
                opt.zero_grad(set_to_none=True)
                loss,_=localization_loss(decode(model,idx),base[idx],data['gt'][idx],data['hw'][idx],torch.ones(len(idx),dtype=torch.bool))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
                opt.step();steps+=1
            model.eval()
            with torch.no_grad():record=dict(fold=fold,epoch=epoch,held=held,**evaluate(decode(model,val),val))
            logs.append(record);print('BOUNDARY_PROBE_EPOCH',json.dumps(record),flush=True)
        with torch.no_grad():
            predictions[val]=decode(model,val)
            fits.append(dict(fold=fold,n=len(train),**evaluate(decode(model,train),train)))
        torch.save(dict(model=model.state_dict(),protocol=protocol,held_sequences=held),dest/f'fold{fold}_final.pt')
    domains={};sequences={}
    for d in groups:
        idx=torch.tensor([i for i,r in enumerate(rows) if r['domain']==d]);domains[d]=evaluate(predictions[idx],idx)
    for s in sorted(set(r['sequence'] for r in rows)):
        idx=torch.tensor([i for i,r in enumerate(rows) if r['sequence']==s]);sequences[s]=evaluate(predictions[idx],idx)
    overall=evaluate(predictions,torch.arange(len(rows)))
    report=dict(protocol=protocol,domains=domains,sequences=sequences,overall=overall,fit=fits,logs=logs,
        sequence_macro_gain=sum(s['iou_gain'] for s in sequences.values())/len(sequences),
        old_domains=original['domains'],elapsed=time.time()-start,optimizer_steps=steps)
    (dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('BOUNDARY_PROBE_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('logs','protocol')}),flush=True)


if __name__=='__main__':main()
