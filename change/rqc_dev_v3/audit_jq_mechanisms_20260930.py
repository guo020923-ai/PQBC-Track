"""Fixed checkpoint/data diagnostic; no optimizer or parameter/buffer updates."""
import os,sys,json,hashlib,itertools
from pathlib import Path
os.environ['LOCAL_RANK']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from lib.config.uvltrack.config import cfg,update_config_from_file
from lib.train.train_script_mutrack import configure_trainable_parameters,reset_training_seed
from lib.train.base_functions import build_dataloaders,update_settings
from lib.train.admin.settings import Settings
from lib import registry
import lib.models,lib.train.actors
NAME='baseline_base_head_rqc_jq_bs16_30ep_20260929'

def digest(net):
    h=hashlib.sha256()
    for n,v in net.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    dest=ROOT/'change/rqc_dev_v3/jq_mechanisms_20260930'
    dest.mkdir(exist_ok=False)
    torch.set_num_threads(1);update_config_from_file(str(ROOT/'experiments/uvltrack'/f'{NAME}.yaml'))
    cfg.TRAIN.BATCH_SIZE=4;cfg.TRAIN.NUM_WORKER=0
    settings=Settings();settings.local_rank=-1;settings.use_lmdb=False
    settings.base_seed=20260930;settings.script_name='uvltrack'
    update_settings(settings,cfg);reset_training_seed(settings)
    net=registry.MODELS['uvltrack'](cfg).cuda()
    ck=ROOT/'checkpoints/train/uvltrack'/NAME/'UVLTrack_ep0030.pth.tar'
    net.load_state_dict(torch.load(str(ck),map_location='cpu')['net'],strict=True)
    configure_trainable_parameters(net,cfg)
    actor=registry.ACTORS['uvltrack'](net,cfg);actor.train(False)
    before=digest(net);capture={}
    old=actor.compute_losses
    def record(pred,gt,cls,cont,*a,**kw):
        capture.update(out=pred,gt=gt,cls=cls,cont=cont)
        return old(pred,gt,cls,cont,*a,**kw)
    actor.compute_losses=record
    handle=net.rqc.register_forward_hook(lambda m,inputs,out:capture.update(batch=inputs[0],pred=out))
    head=[p for n,p in net.named_parameters() if n.startswith('box_head.') and p.requires_grad]
    loaders=build_dataloaders(cfg,settings);rows=[];frames=[]
    for label,loader in [('train_source',loaders[0]),('vl_validation',loaders[-1])]:
        for step,data in enumerate(itertools.islice(loader,32)):
            capture.clear();data=data.to('cuda');data['epoch']=30;data['settings']=settings
            loss,stats=actor(data);assert torch.isfinite(loss)
            if 'batch' not in capture:continue
            o,b,p=capture['out'],capture['batch'],capture['pred']
            active=o['flag'].reshape(-1).eq(2)
            target=actor.candidate_iou(b['boxes'],capture['gt'][active]).detach()
            idx=b['indices'];winner=p['decision_score'].argmax(-1)
            chosen=idx.gather(1,winner[:,None]).flatten()
            # Identify exact original regression location via the selected box.
            delta=(o['bbox_map'][active]-o['pred_boxes'][active].reshape(-1,1,4)).abs().sum(-1)
            assert delta.min(-1).values.max()<1e-6
            regidx=delta.argmin(-1);in_top=idx.eq(regidx[:,None])
            qmax=target.max(-1).values
            for n in range(len(chosen)):
                frames.append(dict(split=label,changed=bool(winner[n]!=0),
                    rqc_matches_regression=bool(chosen[n]==regidx[n]),
                    baseline_matches_regression=bool(idx[n,0]==regidx[n]),
                    regression_in_topk=bool(in_top[n].any()),
                    gain=float(target[n,winner[n]]-target[n,0]),
                    oracle_gain=float(qmax[n]-target[n,0])))
            i,j=torch.triu_indices(10,10,1,device='cuda');adv=target[:,i]-target[:,j]
            s=p['decision_score']
            rank=(F.softplus(-adv.sign()*(s[:,i]-s[:,j]))*adv.abs()).mean().float()*cfg.TRAIN.RQC_LOSS_WEIGHT
            original=actor.loss_weight['cls']*actor.objective['cls'](o['cls_score'],capture['cls'])+actor.loss_weight['cont']*actor.objective['cont'](o['cont_score'].reshape(-1,2),capture['cont'].reshape(-1))
            ga=torch.autograd.grad(rank,head,retain_graph=True,allow_unused=True)
            gb=torch.autograd.grad(original,head,allow_unused=True)
            dot=0.;na=0.;nb=0.
            for x,y in zip(ga,gb):
                if x is not None:assert torch.isfinite(x).all();na+=float(x.square().sum())
                if y is not None:assert torch.isfinite(y).all();nb+=float(y.square().sum())
                if x is not None and y is not None:dot+=float((x*y).sum())
            rows.append(dict(split=label,batch=step,cosine=dot/max((na*nb)**.5,1e-30),rank_grad_norm=na**.5,original_grad_norm=nb**.5))
            del loss,rank,original,ga,gb
        print('SPLIT_DONE',label,flush=True)
    assert digest(net)==before;handle.remove()
    result={}
    mean=lambda a:sum(a)/len(a) if a else None
    for label in ('train_source','vl_validation'):
        fs=[r for r in frames if r['split']==label];rs=[r for r in rows if r['split']==label]
        changes=[r for r in fs if r['changed']]
        result[label]=dict(samples=len(fs),batches=len(rs),changed=len(changes),
            rqc_matches_regression=mean([r['rqc_matches_regression'] for r in fs]),
            baseline_matches_regression=mean([r['baseline_matches_regression'] for r in fs]),
            changed_matches_regression=mean([r['rqc_matches_regression'] for r in changes]),
            regression_in_topk=mean([r['regression_in_topk'] for r in fs]),
            mean_gain=mean([r['gain'] for r in fs]),oracle_gain=mean([r['oracle_gain'] for r in fs]),
            negative_cosine_fraction=mean([r['cosine']<0 for r in rs]),mean_cosine=mean([r['cosine'] for r in rs]),
            mean_rank_grad_norm=mean([r['rank_grad_norm'] for r in rs]),mean_original_grad_norm=mean([r['original_grad_norm'] for r in rs]))
    (dest/'report.json').write_text(json.dumps(dict(summary=result,batches=rows,frames=frames,
        checkpoint=str(ck),optimizer_steps=0,unchanged=True,mode='eval, sampled crops, not closed loop'),indent=2))
    print('MECHANISM_AUDIT_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
