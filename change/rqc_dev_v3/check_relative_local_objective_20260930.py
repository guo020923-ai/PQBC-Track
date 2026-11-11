"""Synthetic and real-fit-cache backward checks only; no optimizer steps."""
import json
from pathlib import Path
import torch
from relative_local_objective_20260930 import relative_localization_loss
from local_box_refiner_20260930 import LocalBoxRefiner,decode_image_boxes,localization_loss


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    gt=torch.tensor([[0.,0.,10.,10.]]).repeat(3,1).requires_grad_()
    baseline=torch.tensor([[1.,0.,10.,10.]]).repeat(3,1).requires_grad_()
    pred=torch.tensor([[1.,0.,10.,10.],[0.,0.,10.,10.],[2.,0.,10.,10.]],requires_grad=True)
    hw=torch.tensor([[100.,100.]]).repeat(3,1);mask=torch.ones(3,dtype=torch.bool)
    _,stats=relative_localization_loss(pred,baseline,gt,hw,mask)
    for i in (0,1):
        _,s=relative_localization_loss(pred[i:i+1],baseline[i:i+1],gt[i:i+1],hw[i:i+1],mask[:1])
        assert float(s['relative_harm'])==0
    loss,_=relative_localization_loss(pred,baseline,gt,hw,mask)
    old,_=localization_loss(pred,baseline,gt,hw,mask)
    added_gradient=torch.autograd.grad(loss-old,pred,retain_graph=True)[0]
    assert added_gradient[2,0]>0 and torch.allclose(added_gradient[:2],torch.zeros_like(added_gradient[:2]),atol=1e-7,rtol=0)
    loss.backward();assert baseline.grad is None and gt.grad is None
    zero,_=relative_localization_loss(pred,baseline,gt,hw,mask,harm_weight=0)
    assert torch.equal(zero,old)
    empty,_=relative_localization_loss(pred,baseline,gt,hw,torch.zeros(3,dtype=torch.bool));assert float(empty)==0
    root=Path(__file__).resolve().parents[2]
    cache=torch.load(str(root/'change/rqc_dev_v3/locked_refiner_parallel_20260930/paired_cache.pt'),map_location='cpu')
    rows=[r for r in cache['rows'] if r['role']=='fit'][:64];assert len(rows)==64
    x={k:torch.stack([r[k] for r in rows]).float() for k in ('features','box','gt','origin')}
    side=torch.tensor([r['side'] for r in rows]);hw=torch.tensor([r['image_hw'] for r in rows])
    model=LocalBoxRefiner();before={k:v.clone() for k,v in model.state_dict().items()}
    output=model(x['features'],x['box'])
    base=decode_image_boxes(x['box'],x['origin'],side,hw)
    refined=decode_image_boxes(output['boxes'],x['origin'],side,hw)
    loss,metrics=relative_localization_loss(refined,base,x['gt'],hw,torch.ones(len(rows),dtype=torch.bool))
    assert float(metrics['relative_harm'])==0
    loss.backward()
    assert model.net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    print('RELATIVE_OBJECTIVE_CHECK_PASS',json.dumps(dict(real_fit_samples=len(rows),loss=float(loss),
        unchanged_and_improved_unpenalized=True,harm_gradient_direction=True,zero_weight_legacy_exact=True,
        upstream_isolated=True,initial_gradient_nonzero=True,optimizer_steps=0,final_holdout_read=False)),flush=True)


if __name__=='__main__':main()
