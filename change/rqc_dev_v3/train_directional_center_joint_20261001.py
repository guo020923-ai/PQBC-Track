"""Single-GPU restricted Head+center training with original four-source loaders.

Default is checks only. --train explicitly required. Compact checkpoints contain
trained Head (including buffers) + refiner; immutable E300 supplies the Backbone.
"""
import os,sys,json,time,argparse,random,warnings
from pathlib import Path
import numpy as np
import torch,cv2
# Suppress only known legacy collate/storage deprecations, including in workers.
# Numerical warnings, exceptions and nonfinite checks remain visible.
warnings.filterwarnings('ignore',message='An output with one or more elements was resized.*',category=UserWarning)
warnings.filterwarnings('ignore',message='TypedStorage is deprecated.*',category=UserWarning)
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from directional_center_joint_contract_20261001 import (NAME,register_config,file_hash,install_metadata,
    JointCenterActor,DirectionalCenterRefiner,fixed_snapshot,verify_joint)
from lib.config.uvltrack.config import cfg,update_config_from_file
from lib.train.admin.settings import Settings
from lib.train.base_functions import update_settings,build_dataloaders
from lib.models.uvltrack.uvltrack import build_model

def prepare():
    os.chdir(str(ROOT));torch.set_num_threads(1);cv2.setNumThreads(1)
    assert int(os.environ.get('LOCAL_RANK','-1'))==-1,'Single GPU only, no DDP'
    assert not torch.distributed.is_initialized()
    register_config();path=ROOT/'experiments/uvltrack'/ (NAME+'.yaml');update_config_from_file(str(path))
    import yaml
    prior=yaml.safe_load((ROOT/'experiments/uvltrack/baseline_base_head_rqc_jq_bs16_30ep_20260929.yaml').read_text())
    current=yaml.safe_load(path.read_text());assert current['DATA']==prior['DATA'],'Dataset configuration changed'
    assert cfg.DATA.TRAIN.SAMPLE_PER_EPOCH==30000 and cfg.TRAIN.BATCH_SIZE==16
    assert cfg.TRAIN.VL_RATIO==.44 and cfg.TRAIN.GROUNDING_RATIO==.11
    assert cfg.TRAIN.EPOCH==cfg.TRAIN.SCHEDULER.T_MAX==30
    assert cfg.TRAIN.FREEZE_BACKBONE and not cfg.TRAIN.FREEZE_HEAD
    assert cfg.TRAIN.HEAD_LR==1e-6 and cfg.TRAIN.CENTER_OUTPUT_LR==1e-4
    assert not cfg.MODEL.RQC.ENABLED and not cfg.MODEL.HEAD.ERC.ENABLED
    assert cfg.TRAIN.HEAD_PEAK_PRESERVE_WEIGHT==0 and cfg.TRAIN.HEAD_ANCHOR_WEIGHT==0
    assert not cfg.TRAIN.HEAD_RESPONSE_DISTILL_ENABLE
    assert cfg.MODEL.CENTER_OUTPUT.ENABLED and cfg.MODEL.CENTER_OUTPUT.OUTPUT_ONLY
    assert cfg.MODEL.CENTER_OUTPUT.CENTER_LIMIT==.1
    assert cfg.MODEL.CENTER_OUTPUT.DESIGN=='directional_center_pair'
    assert cfg.TRAIN.NATIVE_DECISION_BOX_WEIGHT==.5 and cfg.TRAIN.NATIVE_DECISION_RANK_WEIGHT==.25
    assert cfg.TEST.MODE=='NLBBOX'
    source=Path(cfg.TRAIN.INIT_CHECKPOINT);assert source.name=='UVLTrack_ep0300.pth.tar' and source.parent.name=='baseline_base'
    source_hash=file_hash(source)
    random.seed(20260930);np.random.seed(20260930);torch.manual_seed(20260930);torch.cuda.manual_seed_all(20260930)
    settings=Settings();settings.script_name='uvltrack';settings.config_name=NAME
    settings.local_rank=-1;settings.base_seed=20260930;settings.use_lmdb=False
    settings.save_dir=str(ROOT);update_settings(settings,cfg)
    loaders=build_dataloaders(cfg,settings)
    for loader in loaders:loader.dataset.processing=install_metadata(loader.dataset.processing)
    net=build_model(cfg)
    state=torch.load(str(source),map_location='cpu')['net'];net.load_state_dict(state,strict=True);del state
    for p in net.parameters():p.requires_grad_(False)
    for p in net.box_head.parameters():p.requires_grad_(True)
    net.output_refiner=DirectionalCenterRefiner(cfg.MODEL.CENTER_OUTPUT.CENTER_LIMIT)
    net.cuda().eval();net.output_refiner.train()
    actor=JointCenterActor(net,cfg);actor.train(True);verify_joint(net)
    assert not actor.decision_teacher.training
    assert all(not p.requires_grad for p in actor.decision_teacher.parameters())
    for key,value in net.box_head.state_dict().items():
        assert torch.equal(value,actor.decision_teacher.state_dict()[key]),'Teacher is not official initialization'
    assert actor._head_teacher is None and not actor._is_response_v7
    optimizer=torch.optim.AdamW([
        {'params':list(net.box_head.parameters()),'lr':cfg.TRAIN.HEAD_LR,'name':'head'},
        {'params':list(net.output_refiner.parameters()),'lr':cfg.TRAIN.CENTER_OUTPUT_LR,'name':'center_output'}],
        weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    expected={id(p) for p in net.box_head.parameters()}|{id(p) for p in net.output_refiner.parameters()}
    assert {id(p) for g in optimizer.param_groups for p in g['params']}==expected
    assert len(expected)==sum(len(g['params']) for g in optimizer.param_groups)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30,eta_min=0)
    identity=dict(config=NAME,config_sha256=file_hash(path),source_checkpoint=str(source),
        source_checkpoint_sha256=source_hash,epochs=30,samples_per_epoch=30000,batch=16,
        vl_ratio=.44,grounding_ratio=.11,head_trainable=sum(p.numel() for p in net.box_head.parameters()),
        backbone_trainable=0,refiner_trainable=sum(p.numel() for p in net.output_refiner.parameters()),
        native_decision_box_weight=cfg.TRAIN.NATIVE_DECISION_BOX_WEIGHT,native_decision_rank_weight=cfg.TRAIN.NATIVE_DECISION_RANK_WEIGHT,head_lr=cfg.TRAIN.HEAD_LR,
        lr=cfg.TRAIN.CENTER_OUTPUT_LR,weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        head_bn_running_stats_fixed=True,refiner_to_head_gradient=False,gradient_clip='per_group',
        output_only=True,ddp=False,validation_interval=int(cfg.TRAIN.VAL_EPOCH_INTERVAL),
        source_files={name:file_hash(Path(__file__).parent/name) for name in (
            'center_safe_refiner_20260930.py','relative_local_objective_20260930.py',
            'local_box_refiner_20260930.py','center_formal_contract_20260930.py',
            'head_quality_features_20260930.py','directional_center_joint_contract_20261001.py','directional_center_refiner_20261001.py',
            'head_native_decision_guard_20261001.py',
            'center_joint_metadata_exact_20260930.py',
            'train_directional_center_joint_20261001.py','directional_center_joint_tracker_20261001.py',
            '../../lib/train/actors/uvltrack.py','../../lib/models/uvltrack/uvltrack.py',
            '../../lib/config/uvltrack/config.py','../../lib/test/tracker/uvltrack.py')})
    return net,actor,loaders,optimizer,scheduler,identity

def cuda_data(data):return data.to(torch.device('cuda:0'))

def check(net,actor,loaders,optimizer,identity):
    buffers=fixed_snapshot(net);checks=[];active_batches=0
    original_head={k:v.detach().cpu().clone() for k,v in net.box_head.state_dict().items()}
    actor.check_gradients=True
    # Read actual training batches; no optimizer.step, no scheduler.step.
    for i,data in enumerate(loaders[0]):
        optimizer.zero_grad(set_to_none=True);loss,stats=actor(cuda_data(data))
        assert torch.isfinite(loss)
        assert 'HeadDecision/active' in stats and 'Loss/head_decision' in stats
        if stats['Center/active_samples']>0:
            head_loss,center_loss=actor.last_losses
            assert stats['HeadDecision/active']>0
            cross_head=torch.autograd.grad(center_loss,list(net.box_head.parameters()),allow_unused=True,retain_graph=True)
            cross_center=torch.autograd.grad(head_loss,list(net.output_refiner.parameters()),allow_unused=True,retain_graph=True)
            assert all(g is None for g in cross_head+cross_center),'Loss gradient crossed module boundary'
            loss.backward();verify_joint(net,buffers)
            norms={}
            for group in optimizer.param_groups:
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in group['params'])
                norm=float(torch.nn.utils.clip_grad_norm_(group['params'],1.))
                assert norm>0 and np.isfinite(norm);norms[group['name']]=norm
            active_batches+=1
            checks.append(dict(loader='train',gradients=norms,cross_gradient_none=True,**stats))
        if active_batches>=2:break
        if i>=19:raise RuntimeError('No two VL-active batches in first 20 train batches')
    # Actual image batch, deliberately force only tracking/BBOX for isolation.
    from lib.utils import TensorDict
    inactive=TensorDict(dict(data));inactive['flag']=torch.zeros_like(data['flag'])
    optimizer.zero_grad(set_to_none=True);inactive_loss,inactive_stats=actor(cuda_data(inactive))
    assert inactive_stats['Center/active_samples']==0 and torch.isfinite(inactive_loss)
    inactive_loss.backward()
    assert all(p.grad is None for p in net.output_refiner.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.box_head.parameters())
    checks.append(dict(loader='forced_bbox_gradient_check',refiner_gradient_none=True,**inactive_stats))
    actor.check_gradients=False;actor.last_losses=None;actor.train(False)
    for loader in loaders[1:]:
        data=next(iter(loader))
        with torch.no_grad():loss,stats=actor(cuda_data(data))
        assert torch.isfinite(loss);checks.append(dict(loader=loader.name,**stats))
        if loader.name.startswith('vl_'):assert stats['Center/active_samples']>0
        else:assert stats['Center/active_samples']==0
    verify_joint(net,buffers);optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in actor.decision_teacher.parameters())
    assert not actor.decision_teacher.training
    for key,value in actor.decision_teacher.state_dict().items():
        assert torch.equal(value.detach().cpu(),original_head[key]),'Frozen teacher changed'
    assert optimizer.state=={} and scheduler_initial(optimizer)
    for tensor in net.output_refiner.net[-1].parameters():assert torch.count_nonzero(tensor)==0
    for k,v in net.box_head.state_dict().items():assert torch.equal(v.detach().cpu(),original_head[k])
    report=dict(identity=identity,checks=checks,optimizer_steps=0,weights_saved=False,
        frozen_backbone_and_bn_unchanged=True,head_weights_unchanged=True,train_validation_loaders_checked=True)
    print('JOINT_CENTER_CHECK_PASS',json.dumps(report),flush=True)
    return report

def scheduler_initial(optimizer):
    return {g['name']:g['lr'] for g in optimizer.param_groups}=={'head':cfg.TRAIN.HEAD_LR,'center_output':cfg.TRAIN.CENTER_OUTPUT_LR}

def restore_resume(net,optimizer,scheduler,identity,dest,checkpoint,allow_synthetic=False):
    checkpoint=Path(checkpoint).resolve()
    if checkpoint.parent!=dest.resolve():raise ValueError('Resume checkpoint must belong to the output directory')
    saved=torch.load(str(checkpoint),map_location='cpu')
    if saved.get('synthetic_check') and not allow_synthetic:raise ValueError('Synthetic check weights cannot resume formal training')
    if saved.get('variant')!='directional_center_guard_joint' or saved.get('identity')!=identity:
        raise ValueError('Resume protocol/source/config/code identity mismatch')
    epoch=int(saved['epoch'])
    if not 1<=epoch<30:raise ValueError('Resume requires a completed Epoch 1..29')
    if checkpoint.name!=f'DirectionalCenterJoint_ep{epoch:04d}.pth.tar':raise ValueError('Checkpoint filename/epoch mismatch')
    if json.loads((dest/'protocol.json').read_text())!=identity:raise ValueError('Output protocol mismatch')
    records=[json.loads(line) for line in (dest/'stats.jsonl').read_text().splitlines() if line.strip()]
    if [r['epoch'] for r in records]!=list(range(1,epoch+1)):
        raise ValueError('Stats/checkpoint not aligned; refusing to truncate or overwrite logs')
    if any((dest/f'DirectionalCenterJoint_ep{e:04d}.pth.tar').exists() for e in range(epoch+1,31)):
        raise FileExistsError('Newer checkpoints exist; refusing to overwrite')
    if list(dest.glob('*.partial')):raise FileExistsError('Incomplete checkpoint found; preserve and inspect it before resuming')
    net.output_refiner.load_state_dict(saved['model'],strict=True)
    net.box_head.load_state_dict(saved['head'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    if scheduler.last_epoch!=epoch:raise ValueError('Scheduler epoch mismatch')
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    verify_joint(net)
    return epoch+1


def train(net,actor,loaders,optimizer,scheduler,identity,dest,resume=None):
    if resume:
        first_epoch=restore_resume(net,optimizer,scheduler,identity,dest,resume)
    else:
        dest.mkdir(parents=True,exist_ok=False)
        (dest/'protocol.json').write_text(json.dumps(identity,indent=2))
        first_epoch=1
    buffers=fixed_snapshot(net);start=time.time()
    with (dest/'stats.jsonl').open('a' if resume else 'x') as logfile:
        for epoch in range(first_epoch,31):
            epoch_start=time.time();summary={}
            for loader in loaders:
                if epoch%loader.epoch_interval!=0:continue
                actor.train(loader.training);sums={};counts={};active_frames=0
                for step,data in enumerate(loader):
                    data['epoch']=epoch
                    optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(loader.training):loss,stats=actor(cuda_data(data))
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss '+loader.name)
                    if loader.training:
                        loss.backward()
                        for group in optimizer.param_groups:
                            norm=torch.nn.utils.clip_grad_norm_(group['params'],cfg.TRAIN.GRAD_CLIP_NORM)
                            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient '+group['name'])
                            stats['Gradient/'+group['name']+'_norm']=float(norm)
                        if stats['Center/active_samples']==0:
                            assert all(p.grad is None for p in net.output_refiner.parameters())
                        optimizer.step()
                    active=stats['Center/active_samples'];active_frames+=active
                    for key,value in stats.items():
                        # Active metrics weighted by active search frames, not inactive BBOX/NL.
                        if key in ('Center/active_samples','Head/samples'):weight=1
                        elif key.startswith(('Center/','HeadDecision/')) or key in ('Loss/center_total','Gradient/center_output_norm'):weight=active
                        else:weight=stats['Head/samples']
                        sums[key]=sums.get(key,0.)+float(value)*weight;counts[key]=counts.get(key,0.)+weight
                    if step%cfg.TRAIN.PRINT_INTERVAL==0:
                        print('BATCH',json.dumps(dict(epoch=epoch,loader=loader.name,step=step,
                            lr={g['name']:g['lr'] for g in optimizer.param_groups},**stats)),flush=True)
                summary[loader.name]=dict(stats={k:sums[k]/max(counts[k],1) for k in sums},active_frames=active_frames)
                verify_joint(net,buffers)
            scheduler.step()
            record=dict(epoch=epoch,seconds=time.time()-epoch_start,next_lr={g['name']:g['lr'] for g in optimizer.param_groups},loaders=summary)
            logfile.write(json.dumps(record)+'\n');logfile.flush();print('EPOCH',json.dumps(record),flush=True)
            payload=dict(variant='directional_center_guard_joint',epoch=epoch,identity=identity,
                head={k:v.cpu() for k,v in net.box_head.state_dict().items()},
                model={k:v.cpu() for k,v in net.output_refiner.state_dict().items()},optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate())
            final=dest/f'DirectionalCenterJoint_ep{epoch:04d}.pth.tar';tmp=final.with_suffix('.partial')
            if final.exists() or tmp.exists():raise FileExistsError('Refusing to overwrite checkpoint '+str(final))
            torch.save(payload,str(tmp));tmp.replace(final)
    print('TRAIN_COMPLETE',json.dumps(dict(epochs=30,seconds=time.time()-start,output=str(dest))),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--check-only',action='store_true')
    p.add_argument('--output',default=str(ROOT/'change/directional_center_joint_training_20261001'/NAME))
    p.add_argument('--resume',help='Completed compact checkpoint, epoch-boundary continuation only')
    p.add_argument('--report',help='Exclusive JSON path for checks; never overwrites an existing report')
    args=p.parse_args()
    assert not(args.train and args.check_only)
    if args.resume and not args.train:p.error('--resume requires --train')
    if args.train and args.report:p.error('--report is only for checks')
    dest=Path(args.output)
    if args.train and not args.resume and dest.exists():raise FileExistsError('Refusing to overwrite '+str(dest))
    if args.resume and not Path(args.resume).is_file():raise FileNotFoundError(args.resume)
    if args.report and Path(args.report).exists():raise FileExistsError(args.report)
    prepared=prepare()
    if args.train:train(*prepared,dest,resume=args.resume)
    else:
        report=check(prepared[0],prepared[1],prepared[2],prepared[3],prepared[5])
        if args.report:
            with Path(args.report).open('x') as out:json.dump(report,out,indent=2)

if __name__=='__main__':main()
