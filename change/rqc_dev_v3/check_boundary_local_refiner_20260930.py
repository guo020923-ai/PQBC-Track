"""Synthetic spatial checks and real-model passive input smoke test; no training."""
import os,sys,json
from pathlib import Path
import torch
from boundary_local_refiner_20260930 import sample_box_boundary,BoundaryLocalRefiner


def synthetic():
    x=(torch.arange(16)+.5)/16
    yy,xx=torch.meshgrid(x,x,indexing='ij')
    ramp=torch.stack([xx,yy]*16)[None].requires_grad_()
    maps={k:ramp for k in ('conv_bbox','conv_offset','conv_cls')}
    box=torch.tensor([[.5,.5,.5,.5]],requires_grad=True)
    batch=sample_box_boundary(maps,box)
    assert batch['valid'].all()
    assert torch.allclose(batch['samples'][...,0],batch['coordinates'][...,0],atol=1e-6)
    assert torch.allclose(batch['samples'][...,1],batch['coordinates'][...,1],atol=1e-6)
    assert (batch['coordinates'][:,0,...,0] < batch['coordinates'][:,1,...,0]).all()
    assert (batch['coordinates'][:,2,...,1] < batch['coordinates'][:,3,...,1]).all()
    border=sample_box_boundary(maps,torch.tensor([[0.,0.,.5,.5]]))
    assert not border['valid'].all() and (border['samples'][~border['valid']]==0).all()
    model=BoundaryLocalRefiner()
    result=model(batch)
    assert torch.equal(result['boxes'],box.detach())
    (result['boxes']-torch.tensor([[.51,.49,.51,.49]])).square().mean().backward()
    assert ramp.grad is None and box.grad is None
    assert model.net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(parameters=sum(p.numel() for p in model.parameters()),ramp_coordinates=True,
                boundary_order=True,padding_mask=True,zero_residual=True,gradient_isolated=True)


def real():
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));os.chdir(str(root))
    os.environ['LOCAL_RANK']='-1'
    from lib.config.uvltrack.config import cfg,update_config_from_file
    from lib.train.train_script_mutrack import reset_training_seed
    from lib.train.base_functions import build_dataloaders,update_settings
    from lib.train.admin.settings import Settings
    from lib import registry
    import lib.models,lib.train.actors
    from audit_jq_mechanisms_20260930 import NAME,digest
    from head_quality_features_20260930 import HeadQualityCapture
    update_config_from_file(str(root/'experiments/uvltrack'/f'{NAME}.yaml'))
    cfg.TRAIN.BATCH_SIZE=4;cfg.TRAIN.NUM_WORKER=0
    settings=Settings();settings.local_rank=-1;settings.use_lmdb=False;settings.base_seed=42;settings.script_name='uvltrack'
    update_settings(settings,cfg);reset_training_seed(settings)
    net=registry.MODELS['uvltrack'](cfg).cuda()
    net.load_state_dict(torch.load(str(root/'checkpoints/train/uvltrack'/NAME/'UVLTrack_ep0030.pth.tar'),map_location='cpu')['net'])
    actor=registry.ACTORS['uvltrack'](net,cfg);actor.train(False)
    data=next(iter(build_dataloaders(cfg,settings)[-1])).to('cuda');data['epoch']=30;data['settings']=settings
    before=digest(net);captured={};model=BoundaryLocalRefiner().cuda().eval()
    with torch.no_grad():
        torch.manual_seed(42);original,stats=actor(data)
        with HeadQualityCapture(net.box_head) as cap:
            def hook(module,inputs,out):
                flag=data['flag'].reshape(-1);count=cap.maps['conv_bbox'].shape[0]
                assert count % flag.numel()==0
                active=flag.repeat(count//flag.numel()).eq(2)
                batch=sample_box_boundary({k:v[active] for k,v in cap.maps.items()},inputs[0]['boxes'][:,0])
                prediction=model(batch)
                assert torch.equal(prediction['boxes'],inputs[0]['boxes'][:,0])
                captured.update(samples=len(batch['boxes']),valid_rate=float(batch['valid'].float().mean()),
                    missing_edge_rate=float((~batch['valid'].flatten(2).any(-1)).float().mean()))
            handle=net.rqc.register_forward_hook(hook)
            try:
                torch.manual_seed(42);actual,after=actor(data)
            finally:handle.remove()
    assert torch.equal(original,actual) and all(float(stats[k])==float(after[k]) for k in stats)
    assert digest(net)==before and captured
    return dict(**captured,original_loss_unchanged=True,original_model_unchanged=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(42)
    print('BOUNDARY_LOCAL_CHECK_PASS',json.dumps(dict(synthetic=synthetic(),real=real(),optimizer_steps=0)),flush=True)
