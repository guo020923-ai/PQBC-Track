"""Real crop interface check only; no fitting or optimizer steps."""
import os,sys,json
from pathlib import Path
os.environ['LOCAL_RANK']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from lib.config.uvltrack.config import cfg,update_config_from_file
from lib.train.train_script_mutrack import reset_training_seed
from lib.train.base_functions import build_dataloaders,update_settings
from lib.train.admin.settings import Settings
from lib import registry
import lib.models,lib.train.actors
from audit_jq_mechanisms_20260930 import NAME,digest
from head_quality_features_20260930 import HeadQualityCapture

def main():
    torch.set_num_threads(1);update_config_from_file(str(ROOT/'experiments/uvltrack'/f'{NAME}.yaml'))
    cfg.TRAIN.BATCH_SIZE=4;cfg.TRAIN.NUM_WORKER=0
    settings=Settings();settings.local_rank=-1;settings.use_lmdb=False;settings.base_seed=42;settings.script_name='uvltrack'
    update_settings(settings,cfg);reset_training_seed(settings)
    net=registry.MODELS['uvltrack'](cfg).cuda()
    net.load_state_dict(torch.load(str(ROOT/'checkpoints/train/uvltrack'/NAME/'UVLTrack_ep0030.pth.tar'),map_location='cpu')['net'])
    actor=registry.ACTORS['uvltrack'](net,cfg);actor.train(False)
    data=next(iter(build_dataloaders(cfg,settings)[-1])).to('cuda');data['epoch']=30;data['settings']=settings
    before=digest(net);captured={}
    with torch.no_grad():
        torch.manual_seed(42);original,stats=actor(data)
        with HeadQualityCapture(net.box_head) as cap:
            def hook(module,inputs,out):
                flag=data['flag'].reshape(-1)
                count=cap.maps['conv_bbox'].shape[0]
                assert count % flag.numel()==0
                active=flag.repeat(count//flag.numel()).eq(2)
                captured['features']=cap.gather(inputs[0]['indices'],active)
            h=net.rqc.register_forward_hook(hook)
            torch.manual_seed(42);actual,after=actor(data);h.remove()
            assert torch.equal(original,actual)
            assert all(float(stats[k])==float(after[k]) for k in stats)
    assert digest(net)==before
    f=captured['features']
    print('HEAD_FEATURE_INTERFACE_PASS',json.dumps(dict(shape=list(f.shape),finite=True,requires_grad=f.requires_grad,
        loss_difference=float(actual-original),state_unchanged=True,optimizer_steps=0,
        adjacent_candidate_feature_difference=float((f[:,1:]-f[:,:1]).abs().mean()),
        warning='Nonzero feature differences do not prove quality separability or gains.')),flush=True)

if __name__=='__main__':main()
