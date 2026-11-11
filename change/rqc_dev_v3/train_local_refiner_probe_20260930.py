"""Explicitly launched, fixed four-fold prototype experiment; not production train.

No checkpoint selection, no hyperparameter sweep. Every fold starts at zero
residual; splits by sequence. The previously inspected subset is development
data, NOT an untouched benchmark. This file is not launched by any monitor.
"""
import argparse, json, time
from pathlib import Path
import torch
from local_box_refiner_20260930 import LocalBoxRefiner, decode_image_boxes, localization_loss


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    rows=torch.load(args.cache,map_location='cpu')['rows']
    groups={d:sorted(set(r['sequence'] for r in rows if r['domain']==d)) for d in sorted(set(r['domain'] for r in rows))}
    assert len(groups)==2 and all(len(v)==4 for v in groups.values())
    data={k:torch.stack([r[k] for r in rows]).float() for k in ('features','box','origin','gt')}
    data['side']=torch.tensor([r['side'] for r in rows],dtype=torch.float32)
    data['hw']=torch.tensor([r['image_hw'] for r in rows],dtype=torch.float32)
    assert all(torch.isfinite(v).all() for v in data.values())
    protocol=dict(epochs=20,lr=1e-4,batch_size=64,folds=4,seed=42,
                  checkpoint_rule='final epoch only',source=str(Path(args.cache).resolve()),
                  frozen='cached JQ E30 native-response Head/Backbone; ERC not used',
                  warning='Development feasibility, not official-init full training or closed-loop evaluation')
    if args.check_only:
        print('LOCAL_TRAIN_PREFLIGHT_PASS',json.dumps(dict(samples=len(rows),groups=groups,protocol=protocol,optimizer_steps=0)))
        return
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2))
    logs=[];predictions=torch.zeros(len(rows),4);start=time.time()
    def decoded(model,idx):
        box=model(data['features'][idx],data['box'][idx])['boxes']
        return decode_image_boxes(box,data['origin'][idx],data['side'][idx],data['hw'][idx])
    base=decode_image_boxes(data['box'],data['origin'],data['side'],data['hw'])
    for fold in range(4):
        held=[s[fold] for s in groups.values()]
        train=torch.tensor([i for i,r in enumerate(rows) if r['sequence'] not in held])
        val=torch.tensor([i for i,r in enumerate(rows) if r['sequence'] in held])
        assert not (set(rows[i]['sequence'] for i in train.tolist()) & set(held))
        torch.manual_seed(42+fold);model=LocalBoxRefiner()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
        for epoch in range(1,21):
            model.train();order=train[torch.randperm(len(train))]
            for idx in order.split(64):
                opt.zero_grad(set_to_none=True)
                loss,_=localization_loss(decoded(model,idx),base[idx],data['gt'][idx],data['hw'][idx],torch.ones(len(idx),dtype=torch.bool))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                opt.step()
            model.eval()
            with torch.no_grad():
                output=decoded(model,val)
                loss,stats=localization_loss(output,base[val],data['gt'][val],data['hw'][val],torch.ones(len(val),dtype=torch.bool))
            record=dict(fold=fold,epoch=epoch,held=held,loss=float(loss),**{k:float(v) for k,v in stats.items()})
            logs.append(record);print('LOCAL_PROBE_EPOCH',json.dumps(record),flush=True)
        predictions[val]=output
        torch.save(dict(model=model.state_dict(),protocol=protocol,held_sequences=held),dest/f'fold{fold}_final.pt')
    results={}
    for name in groups:
        idx=torch.tensor([i for i,r in enumerate(rows) if r['domain']==name])
        _,stats=localization_loss(predictions[idx],base[idx],data['gt'][idx],data['hw'][idx],torch.ones(len(idx),dtype=torch.bool))
        results[name]={k:float(v) for k,v in stats.items()}
    sequences={}
    for name in sorted(set(r['sequence'] for r in rows)):
        idx=torch.tensor([i for i,r in enumerate(rows) if r['sequence']==name])
        _,stats=localization_loss(predictions[idx],base[idx],data['gt'][idx],data['hw'][idx],torch.ones(len(idx),dtype=torch.bool))
        sequences[name]={k:float(v) for k,v in stats.items()}
    (dest/'report.json').write_text(json.dumps(dict(protocol=protocol,logs=logs,domains=results,sequences=sequences,
        sequence_macro_gain=sum(s['iou_gain'] for s in sequences.values())/len(sequences),elapsed=time.time()-start),indent=2))
    print('LOCAL_PROBE_COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':main()
