"""Replay real features; backward once, no optimizer, no fitted weights."""
import sys, json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lib.utils.box_ops import clip_box
from local_box_refiner_20260930 import LocalBoxRefiner, decode_image_boxes, localization_loss


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    folder=ROOT/'change/rqc_dev_v3/local_refiner_audit_20260930_checked'
    rows=torch.load(str(folder/'local_features.pt'),map_location='cpu')['rows']
    model=LocalBoxRefiner()
    feat=torch.stack([r['features'] for r in rows]).requires_grad_()
    box=torch.stack([r['box'] for r in rows]).requires_grad_()
    origin=torch.stack([r['origin'] for r in rows])
    side=torch.tensor([r['side'] for r in rows],dtype=torch.float32)
    hw=torch.tensor([r['image_hw'] for r in rows],dtype=torch.float32)
    gt=torch.stack([r['gt'] for r in rows]);valid=torch.ones(len(rows),dtype=torch.bool)
    output=model(feat,box)
    assert torch.equal(output['boxes'],box.detach())
    real=decode_image_boxes(output['boxes'],origin,side,hw)
    base=decode_image_boxes(box.detach(),origin,side,hw)
    assert torch.equal(real,base)
    # Check official clipping, including saturated nonzero residual, not just zero.
    with torch.no_grad():
        for candidate in (box.detach(),torch.cat((box[:,:2]+.1*box[:,2:],box[:,2:]*1.2),-1)):
            direct=decode_image_boxes(candidate,origin,side,hw)
            raw=torch.cat((origin+(candidate[:,:2]-candidate[:,2:]/2)*side[:,None],candidate[:,2:]*side[:,None]),-1)
            expected=torch.tensor([clip_box(v.tolist(),*dims.tolist(),margin=10) for v,dims in zip(raw,hw)])
            assert torch.allclose(direct,expected,rtol=0,atol=.0002)
    before={k:v.clone() for k,v in model.state_dict().items()}
    loss,stats=localization_loss(real,base,gt,hw,valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert feat.grad is None and box.grad is None
    assert model.net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    report=dict(samples=len(rows),loss=float(loss),zero_output_exact=True,official_clip_match=True,
                finite_nonzero_gradient=True,parameters_unchanged=True,optimizer_steps=0,
                stats={k:float(v) for k,v in stats.items()},
                warning='Engineering feasibility only, not learned generalization or full tracker integration.')
    with (folder/'real_contract.json').open('x') as f:json.dump(report,f,indent=2)
    print('REAL_LOCAL_CONTRACT_PASS',json.dumps(report),flush=True)


if __name__=='__main__':main()
