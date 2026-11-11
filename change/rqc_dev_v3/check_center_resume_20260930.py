"""CPU serialization/control-flow check; synthetic data, zero optimizer updates."""
import json,time,random
from pathlib import Path
import numpy as np
import torch
from train_center_formal_20260930 import ROOT,restore_resume
from center_safe_refiner_20260930 import CenterSafeRefiner

class SyntheticNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Identity().eval()
        self.box_head=torch.nn.Identity().eval()
        self.output_refiner=CenterSafeRefiner()

def main():
    dest=ROOT/'change/rqc_dev_v3'/('center_resume_check_'+time.strftime('%Y%m%d_%H%M%S'))
    dest.mkdir(exist_ok=False)
    net=SyntheticNet()
    optimizer=torch.optim.AdamW(net.output_refiner.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
    # Synthetic completed-epoch scheduler metadata, NOT an optimization run.
    sched_state=scheduler.state_dict();sched_state['last_epoch']=1
    identity=dict(config='SYNTHETIC_RESUME_CHECK_NOT_FOR_TESTING')
    payload=dict(variant='center_safe',epoch=1,identity=identity,
        model=net.output_refiner.state_dict(),optimizer=optimizer.state_dict(),scheduler=sched_state,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate())
    cp=dest/'CenterSafe_ep0001.pth.tar';torch.save(payload,str(cp))
    (dest/'protocol.json').write_text(json.dumps(identity))
    (dest/'stats.jsonl').write_text(json.dumps(dict(epoch=1))+'\n')
    before={k:v.clone() for k,v in net.output_refiner.state_dict().items()}
    first=restore_resume(net,optimizer,scheduler,identity,dest,cp)
    assert first==2 and optimizer.state=={}
    assert all(torch.equal(before[k],v) for k,v in net.output_refiner.state_dict().items())
    try:restore_resume(net,optimizer,scheduler,dict(config='wrong'),dest,cp)
    except ValueError:pass
    else:raise AssertionError('Identity mismatch not rejected')
    newer=dest/'CenterSafe_ep0002.pth.tar';newer.touch(exist_ok=False)
    try:restore_resume(net,optimizer,scheduler,identity,dest,cp)
    except FileExistsError:pass
    else:raise AssertionError('Newer checkpoint not protected')
    print('CENTER_RESUME_CHECK_PASS',json.dumps(dict(next_epoch=first,optimizer_steps=0,
        model_unchanged=True,identity_mismatch_rejected=True,newer_checkpoint_protected=True,
        synthetic_checkpoint_bytes=cp.stat().st_size,directory=str(dest))),flush=True)

if __name__=='__main__':main()
