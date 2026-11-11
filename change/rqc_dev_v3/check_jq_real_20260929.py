"""Engineering checks only: no optimizer steps, training, or checkpoint writes."""
import os,sys,copy,json,hashlib
from pathlib import Path
os.environ['LOCAL_RANK']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from lib.config.uvltrack.config import cfg,update_config_from_file
from lib.train.train_script_mutrack import load_initial_checkpoint,configure_trainable_parameters,reset_training_seed
from lib.train.base_functions import get_optimizer_scheduler,build_dataloaders,update_settings
from lib.train.admin.settings import Settings
from lib import registry
import lib.models,lib.train.actors
from lib.models.heads.candidate_quality_jq import training_loss,decision_scores
NAME='baseline_base_head_rqc_jq_bs16_30ep_20260929'

def digest(net):
    h=hashlib.sha256()
    for n,t in net.state_dict().items():h.update(n.encode());h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    update_config_from_file(str(ROOT/'experiments/uvltrack'/f'{NAME}.yaml'))
    cfg.TRAIN.BATCH_SIZE=2;cfg.TRAIN.NUM_WORKER=0
    settings=Settings();settings.local_rank=-1;settings.use_lmdb=False
    settings.base_seed=42;settings.script_name='uvltrack'
    update_settings(settings,cfg);reset_training_seed(settings)
    net=registry.MODELS['uvltrack'](cfg).cuda()
    load_initial_checkpoint(net,cfg.TRAIN.INIT_CHECKPOINT);configure_trainable_parameters(net,cfg)
    opt,sched=get_optimizer_scheduler(net,cfg)
    actor=registry.ACTORS['uvltrack'](net,cfg);actor.train(False)
    before=digest(net);capture={}
    def hook(module,inputs,out):capture.update(batch=inputs[0],pred=out)
    handle=net.rqc.register_forward_hook(hook)
    data=next(iter(build_dataloaders(cfg,settings)[-1])).to('cuda')
    data['epoch']=1;data['settings']=settings
    loss,stats=actor(data);assert torch.isfinite(loss)
    b,p=capture['batch'],capture['pred']
    assert p['decision_score'].argmax(-1).eq(0).all()
    assert b['live_prior'].requires_grad and not b['boxes'].requires_grad
    # Independent ranking gradient check on real Head responses, without taking
    # an optimization step. Synthetic targets test connectivity, not efficacy.
    target=torch.linspace(.1,.9,p['quality'].shape[1],device='cuda').expand_as(p['quality'])
    checked,_=training_loss(p,target)
    head=[v for n,v in net.named_parameters() if n.startswith('box_head.') and v.requires_grad]
    grads=torch.autograd.grad(checked,head,retain_graph=True,allow_unused=True)
    assert any(g is not None and g.abs().sum()>0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    loss.backward()
    assert all(v.grad is None for n,v in net.named_parameters() if n.startswith('backbone.'))
    assert net.rqc.quality[-1].weight.grad is not None
    assert torch.isfinite(net.rqc.quality[-1].weight.grad).all()
    # Equivalent two/three-logit candidate path is exercised separately in the
    # dedicated test fixture by constructing identical probabilities.
    from lib.models.heads.candidate_quality_jq import candidates
    side=16;dim=768
    raw=dict(search=torch.randn(1,256,dim,device='cuda'),template=torch.randn(1,64,dim,device='cuda'),
        txt_token=torch.randn(1,1,dim,device='cuda'),prompts=torch.randn(1,3,dim,device='cuda'),
        cls_score_test=torch.rand(1,256,device='cuda'),cont_score=torch.randn(1,256,2,device='cuda'),
        bbox_map=torch.rand(1,256,4,device='cuda'))
    mask=torch.ones(1,64,device='cuda');a=candidates(raw,mask)
    raw['cont_score']=torch.cat((raw['cont_score'],torch.zeros_like(raw['cont_score'][...,:1])),-1)
    c=candidates(raw,mask)
    assert torch.equal(a['indices'],c['indices']) and torch.equal(a['live_prior'],c['live_prior'])
    assert digest(net)==before
    handle.remove()
    # Exercise deployed tracker, not only the module argmax, at initialization.
    import cv2,numpy as np
    from unittest.mock import patch
    from lib.test.parameter.uvltrack import parameters
    from lib.test.tracker.uvltrack import UVLTrack
    from lib.test.evaluation import get_dataset
    params=parameters(NAME,epoch=30);params.cfg=copy.deepcopy(params.cfg);params.debug=0
    params.checkpoint='/__jq_memory_check__'
    state={k:v.detach().cpu() for k,v in net.state_dict().items()}
    real_load=torch.load
    def memory_load(path,*a,**kw):
        return {'net':state} if str(path)==params.checkpoint else real_load(path,*a,**kw)
    with patch('torch.load',side_effect=memory_load):new=UVLTrack(params,'tnl2k')
    baseparams=copy.deepcopy(params);baseparams.cfg.MODEL.RQC.ENABLED=False
    baseparams.checkpoint=cfg.TRAIN.INIT_CHECKPOINT;base=UVLTrack(baseparams,'tnl2k')
    errors=[]
    with torch.no_grad():
        for seq in list(get_dataset('tnl2k'))[:2]:
            read=lambda path:cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            info=seq.init_info();info['language']=seq.language
            base.initialize(read(seq.frames[0]),copy.deepcopy(info));new.initialize(read(seq.frames[0]),copy.deepcopy(info))
            for path in seq.frames[1:6]:
                frame=read(path)
                errors.append(float(np.max(np.abs(np.asarray(base.track(frame)['target_bbox'])-np.asarray(new.track(frame)['target_bbox'])))))
    assert max(errors)==0,errors
    print('DEPLOYED_EQUIVALENCE_PASS',json.dumps(dict(frames=len(errors),max_box_error=max(errors))),flush=True)
    print('JQ_CHECK_PASS',json.dumps(dict(real_loss=float(loss),active=float(stats['RQC/active_samples']),
        optimizer_groups=[dict(name=g.get('name'),lr=g['lr']) for g in opt.param_groups],
        scheduler_tmax=sched.T_max,head_receives_ranking_gradient=True,optimizer_steps=0,
        initial_top1_equivalence=True,weights_and_buffers_unchanged=True)),flush=True)

if __name__=='__main__':main()
