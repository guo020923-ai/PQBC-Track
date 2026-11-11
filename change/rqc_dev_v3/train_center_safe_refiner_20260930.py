"""One fixed center-safe development comparison, never load final/test trajectories."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from prepare_locked_refiner_protocol_20260930 import load_role
from center_safe_refiner_20260930 import CenterSafeRefiner,center_safe_loss
from local_box_refiner_20260930 import decode_image_boxes,localization_loss
BASE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20260930)
    source=BASE/'locked_refiner_parallel_20260930/paired_cache.pt'
    cache=torch.load(str(source),map_location='cpu');rows=cache['rows'];assert not cache['identity']['smoke']
    allowed=load_role(BASE/'locked_refiner_protocol_20260930','fit')+load_role(BASE/'locked_refiner_protocol_20260930','dev')
    assert {(r['sequence'],r['role']) for r in rows}=={(r['name'],r['role']) for r in allowed}
    x={k:torch.stack([r[k] for r in rows]) for k in ('features','box','gt','origin')}
    side=torch.tensor([r['side'] for r in rows]);hw=torch.tensor([r['image_hw'] for r in rows])
    base=decode_image_boxes(x['box'],x['origin'],side,hw)
    fit=[i for i,r in enumerate(rows) if r['role']=='fit'];dev=torch.tensor([i for i,r in enumerate(rows) if r['role']=='dev'])
    groups={s:[i for i in fit if rows[i]['sequence']==s] for s in sorted(set(rows[i]['sequence'] for i in fit))}
    assert len(groups)==80 and len(set(rows[i]['sequence'] for i in dev.tolist()))==20
    model=CenterSafeRefiner()
    def decode(idx):
        prediction=model(x['features'][idx],x['box'][idx])['boxes']
        assert torch.equal(prediction[:,2:],x['box'][idx,2:])
        return decode_image_boxes(prediction,x['origin'][idx],side[idx],hw[idx])
    idx=torch.tensor(fit[:64]);pred=decode(idx);assert torch.equal(pred,base[idx])
    loss,stats=center_safe_loss(pred,base[idx],x['gt'][idx],hw[idx],torch.ones(len(idx),dtype=torch.bool))
    loss.backward();assert torch.isfinite(loss) and model.net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Zero-point preservation derivative must vanish; localization still learns.
    r=base[idx].clone().requires_grad_(True)
    total,_=center_safe_loss(r,base[idx],x['gt'][idx],hw[idx],torch.ones(len(idx),dtype=torch.bool))
    original,_=localization_loss(r,base[idx],x['gt'][idx],hw[idx],torch.ones(len(idx),dtype=torch.bool))
    extra=total-original;assert extra.item()==0
    assert torch.equal(torch.autograd.grad(extra,r)[0],torch.zeros_like(r))
    g=torch.tensor([[10.,10.,20.,20.]]);b=torch.tensor([[12.,10.,20.,20.]])
    h=torch.tensor([[100.,100.]]);valid=torch.tensor([True])
    def penalty(box):
        _,s=center_safe_loss(box,b,g,h,valid)
        return float(s['center_harm_penalty'])
    assert penalty(b)==0 and penalty(g)==0
    assert penalty(torch.tensor([[16.,10.,20.,20.]]))>0
    # Gradient may reach the refinement only, never cached features/base/GT.
    feature=x['features'][idx].clone().requires_grad_(True)
    box=x['box'][idx].clone().requires_grad_(True)
    truth=x['gt'][idx].clone().requires_grad_(True)
    prediction=decode_image_boxes(model(feature,box)['boxes'],x['origin'][idx],side[idx],hw[idx])
    testloss,_=center_safe_loss(prediction,base[idx],truth,hw[idx],torch.ones(len(idx),dtype=torch.bool))
    testloss.backward();assert feature.grad is None and box.grad is None and truth.grad is None
    if args.check_only:
        print('CENTER_SAFE_CHECK_PASS',json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
            optimizer_steps=0,size_exact=True,finite_gradient=True,zero_extra_gradient=True)));return
    model.zero_grad(set_to_none=True)
    dest=BASE/'center_safe_pilot_20260930';dest.mkdir(exist_ok=False)
    protocol=dict(epochs=20,batches_per_epoch=50,batch=64,lr=1e-4,seed=20260930,
        checkpoint='last epoch only',source='JQ E30 natural cache, not official baseline',
        harm_weight=1,center_harm_weight=1,output_feedback=False,
        cache_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),final_read=False)
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0);rng=np.random.RandomState(20260930);names=list(groups);logs=[]
    for epoch in range(1,21):
        for _ in range(50):
            idx=torch.tensor([rng.choice(groups[names[j]]) for j in rng.randint(len(names),size=64)])
            opt.zero_grad(set_to_none=True);loss,_=center_safe_loss(decode(idx),base[idx],x['gt'][idx],hw[idx],torch.ones(len(idx),dtype=torch.bool))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            opt.step()
        with torch.no_grad():
            value,stats=localization_loss(decode(dev),base[dev],x['gt'][dev],hw[dev],torch.ones(len(dev),dtype=torch.bool))
        row=dict(epoch=epoch,loss=float(value),**{k:float(v) for k,v in stats.items()});logs.append(row);print('DEV',json.dumps(row),flush=True)
    bysequence={}
    with torch.no_grad():
        for name in sorted(set(rows[i]['sequence'] for i in dev.tolist())):
            idx=torch.tensor([i for i in dev.tolist() if rows[i]['sequence']==name])
            _,stats=localization_loss(decode(idx),base[idx],x['gt'][idx],hw[idx],torch.ones(len(idx),dtype=torch.bool))
            bysequence[name]={k:float(v) for k,v in stats.items()}
    torch.save(dict(model=model.state_dict(),variant='center_safe',protocol=protocol),dest/'center_safe_ep20.pt')
    (dest/'report.json').write_text(json.dumps(dict(logs=logs,sequences=bysequence),indent=2))
    print('TRAIN_COMPLETE',flush=True)

if __name__=='__main__':main()
