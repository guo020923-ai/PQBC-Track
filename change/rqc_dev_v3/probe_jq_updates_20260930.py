"""Eight independent one-step interventions, fresh AdamW, memory only.

Eval mode freezes BN/dropout; this is NOT resuming the epoch30 optimizer and
does not estimate AUC. Initial YAML learning rates intentionally test sensitivity.
"""
import os,sys,json,gc
from pathlib import Path
os.environ['LOCAL_RANK']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from lib.config.uvltrack.config import cfg,update_config_from_file
from lib.train.train_script_mutrack import configure_trainable_parameters,reset_training_seed
from lib.train.base_functions import build_dataloaders,update_settings,get_optimizer_scheduler
from lib.train.admin.settings import Settings
from lib.models.heads.candidate_quality_jq import decision_scores
from lib import registry
import lib.models,lib.train.actors
from audit_jq_mechanisms_20260930 import digest,NAME

def main():
    dest=ROOT/'change/rqc_dev_v3/jq_update_probe_20260930';dest.mkdir(exist_ok=False)
    torch.set_num_threads(1);update_config_from_file(str(ROOT/'experiments/uvltrack'/f'{NAME}.yaml'))
    cfg.TRAIN.BATCH_SIZE=4;cfg.TRAIN.NUM_WORKER=0
    settings=Settings();settings.local_rank=-1;settings.use_lmdb=False
    settings.base_seed=20260930;settings.script_name='uvltrack'
    update_settings(settings,cfg);reset_training_seed(settings)
    net=registry.MODELS['uvltrack'](cfg).cuda()
    ck=ROOT/'checkpoints/train/uvltrack'/NAME/'UVLTrack_ep0030.pth.tar'
    state=torch.load(str(ck),map_location='cpu')['net']
    net.load_state_dict(state);configure_trainable_parameters(net,cfg)
    actor=registry.ACTORS['uvltrack'](net,cfg);actor.train(False)
    before=digest(net);capture={};control={'arm':'joint'}
    original=actor.compute_losses
    def record(*a,**kw):
        result=original(*a,**kw);capture['original']=result[0];return result
    actor.compute_losses=record
    def hook(m,inputs,out):
        if control['arm']=='detach_rank_head':
            out=dict(out);out['decision_score']=decision_scores(inputs[0]['live_prior'].detach(),out['quality_logit'])
            return out
    handle=net.rqc.register_forward_hook(hook)
    loaders=build_dataloaders(cfg,settings);train=iter(loaders[0]);valid=iter(loaders[-1])
    def prepare(data):
        data=data.to('cuda');data['epoch']=30;data['settings']=settings;return data
    def evaluate(data):
        with torch.no_grad():
            total,s=actor(data)
            return {k:float(s[k]) for k in ('IoU','Loss/cls','Loss/cont','Loss/rqc_total',
                    'RQC/selected_iou_gain','RQC/selection_change_rate','RQC/harmful_change_rate')}
    rows=[]
    for pair in range(8):
        for attempt in range(100):
            raw=next(train)
            if raw['flag'].eq(2).any():break
        else:raise RuntimeError('No active VL batch')
        fit=prepare(raw);held=prepare(next(valid))
        net.load_state_dict(state);control['arm']='joint';reference=evaluate(held)
        for arm in ('original_only','joint','detach_rank_head'):
            net.load_state_dict(state);net.zero_grad(set_to_none=True)
            control['arm']=arm;opt,_=get_optimizer_scheduler(net,cfg)
            total,stats=actor(fit)
            objective=capture['original'] if arm=='original_only' else total
            assert torch.isfinite(objective);objective.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],cfg.TRAIN.GRAD_CLIP_NORM)
            assert torch.isfinite(norm)
            opt.step();actual=evaluate(held)
            rows.append(dict(pair=pair,arm=arm,preclip_norm=float(norm),reference=reference,after=actual,
                delta={k:actual[k]-reference[k] for k in reference}))
            del opt,total,objective;capture.clear();net.zero_grad(set_to_none=True)
        print('PAIR_COMPLETE',pair+1,flush=True)
        del fit,held;gc.collect()
    net.load_state_dict(state);assert digest(net)==before;handle.remove()
    summary={}
    for arm in ('original_only','joint','detach_rank_head'):
        part=[r for r in rows if r['arm']==arm]
        summary[arm]={k:sum(r['delta'][k] for r in part)/len(part) for k in part[0]['delta']}
    report=dict(summary=summary,rows=rows,checkpoint=str(ck),independent_pairs=8,
        in_memory_optimizer_steps=24,checkpoint_writes=0,original_state_restored=True,
        conditions='Fresh AdamW, YAML initial LR, eval mode; not a continued training or generalization test')
    (dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('UPDATE_PROBE_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
