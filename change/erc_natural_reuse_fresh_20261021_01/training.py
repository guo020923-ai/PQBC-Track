"""Exactly one fresh natural-only full60. No old ERC/optimizer/RNG loading."""
import hashlib,json,os,random,time
from pathlib import Path
import torch
from .checkpoint import identity,KIND,SOURCE_SHA,TRAIN_SHA
from .model import FreshNaturalERC
from .objective import objective
from .device_stats import DeviceStatistics

POLICY=dict(epochs=60,source_epoch=30,first_epoch=1,initial_optimizer_steps=0,
    training_views=1,natural_RQC_only=True,fresh_ERC=True,old_ERC_loaded=False,
    tickets_per_epoch=30000,batch_tickets=16,VL_ratio=.44,valid_searches_per_VL_ticket=2,
    seed=20261016,warmup_frames=16,clip_frames=192,LR=4e-5,weight_decay=1e-4,
    grad_clip=1.,scheduler='fresh_cosine60',scheduler_horizon=60,
    original_TRAIN_no_new_holdout=True,automatic_test=False,automatic_retrain=False,
    automatic_retry=False,gain_based_early_stop=False,wallclock_limit_seconds=None,
    checkpoint_selection='fixed_E60_no_best')

def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())

def fresh_state(device):
    import numpy as np
    seed=POLICY['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    model=FreshNaturalERC().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['LR'],weight_decay=POLICY['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60)
    assert not optimizer.state and scheduler.last_epoch==0 and optimizer.param_groups[0]['lr']==4e-5
    return model,optimizer,scheduler

def validate_admission(admission):
    expected=identity()
    if not isinstance(admission,dict) or admission.get('implementation')!=expected or admission.get('policy')!=POLICY:
        raise ValueError('Exact fresh implementation/policy admission required')
    auth=admission.get('authorization',{})
    if auth.get('scope')!='one_fresh_natural_only_ERC60' or not auth.get('explicit_user_training_authorization'):
        raise PermissionError('Explicit fresh60 training authorization required')
    if auth.get('target_epoch')!=60 or auth.get('maximum_launches')!=1 or not auth.get('human_authorization_reference'):
        raise ValueError('Fixed full60 endpoint and one launch required')
    if any(auth.get(k,True) for k in ('automatic_test','automatic_retrain','automatic_retry')):
        raise PermissionError('No automatic expansion')
    cpu,eng=admission.get('CPU',{}),admission.get('real_engineering',{})
    if not cpu.get('passed') or cpu.get('implementation')!=expected or cpu.get('GPU_operations')!=0:
        raise ValueError('Fresh CPU evidence required')
    if not eng.get('passed') or eng.get('implementation')!=expected or eng.get('optimizer_steps')!=0:
        raise ValueError('Fresh real-engineering evidence required')
    required=('native_context_exact','source_history_isolated','natural_training_deploy_equal',
              'same_TRAIN_pixels','zero_equals_RQC','source_GT_input_gradients_isolated','all600_actual_actions')
    if not all(eng.get(k) is True for k in required) or eng.get('observations')!=24:
        raise ValueError('Missing real24 engineering evidence')
    deps=admission.get('dependencies',{})
    if not deps:raise ValueError('No immutable dependencies')
    for p,sha in deps.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha:raise ValueError('Dependency changed: '+p)
    if SOURCE_SHA!=admission.get('source_sha256') or TRAIN_SHA!=admission.get('TRAIN_sha256'):
        raise ValueError('Source/TRAIN mismatch')
    return expected

def train60(admission,output_dir):
    expected=validate_admission(admission)
    from .bridge import original_training_components,source_snapshot,assert_unchanged
    from .sampling import GraphSampler
    legacy,Stream=original_training_components();legacy.idle()
    directory=Path(output_dir)
    if directory.exists():raise FileExistsError('Fresh output only; never reset old output/lock')
    directory.mkdir(parents=False)
    write_new(directory/'launch_once.json',dict(admission=admission,process=legacy.process(os.getpid())))
    spec,cfg,plan,params=legacy.settings();torch.set_num_threads(1)
    assert spec['NATURAL_INPUT']['WARMUP_FRAMES']==16 and spec['NATURAL_INPUT']['CLIP_FRAMES']==192
    model,optimizer,scheduler=fresh_state('cuda')
    initial_sha=hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    write_new(directory/'fresh_initial_state_01.json',dict(fresh_ERC=True,old_ERC_loaded=False,
        old_optimizer_loaded=False,old_scheduler_loaded=False,old_RNG_loaded=False,
        optimizer_steps_total=0,optimizer_state_parameters=0,scheduler_last_epoch=scheduler.last_epoch,
        initial_LR=optimizer.param_groups[0]['lr'],seed=POLICY['seed'],
        initial_state_tensor_sha256=initial_sha,model_parameters=model.parameter_count,implementation=expected))
    pools=legacy.FitPools(cfg,plan);streams={};snapshots={};sampler=None
    total_steps=0;seed=POLICY['seed'];start=time.monotonic();epochs=[]
    write_new(directory/'protocol.json',dict(implementation=expected,policy=POLICY,
        source_sha256=SOURCE_SHA,TRAIN_sha256=TRAIN_SHA,fresh_ERC=True,old_ERC_loaded=False,
        optimizer_initial_steps=0,old_optimizer_loaded=False,scheduler_initial_epoch=0,
        natural_only_from_epoch1=True,automatic_test=False))
    try:
        for i in pools.vl:
            streams[i]=Stream(pools,i,params,random.Random(seed+i),spec)
            snapshots[i]=source_snapshot(streams[i].collector)
        import numpy as np
        sampler=GraphSampler(streams)
        for epoch in range(1,61):
            started=time.monotonic();model.train();stats=DeviceStatistics(pools.vl)
            histogram=torch.zeros(25,device='cuda',dtype=torch.long)
            tickets=pools.epoch_plan(30000,random.Random(seed+epoch));steps=vl_tickets=0
            for i,s in streams.items():s.rng=random.Random(seed+epoch*100+i);s.clip=None
            for offset in range(0,30000,16):
                batch=tickets[offset:offset+16];rows=sampler.collect(batch)
                vl_tickets+=sum(flag==2 for flag,_ in batch)
                if not rows:continue
                inputs,fixed,truth,hw=legacy.collate(rows)
                optimizer.zero_grad(set_to_none=True);pred=model(**inputs)
                loss,samples,_=objective(model,pred,inputs,fixed,truth,hw,return_device_samples=True)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();steps+=1;total_steps+=1
                if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                    raise FloatingPointError('Nonfinite parameter')
                ids=torch.tensor([r['meta']['source'] for r in rows],device=fixed.device)
                stats.add(samples,ids);histogram+=torch.bincount(pred['index'],minlength=25)
                if total_steps==1:
                    write_new(directory/'actual_optimizer_step_01.json',dict(optimizer_steps=1,
                        optimizer_state_parameters=len(optimizer.state),model_parameters=model.parameter_count,
                        loss=float(loss.detach()),gradient_norm=float(norm),fresh_ERC=True,
                        old_ERC_loaded=False,initial_state_tensor_sha256=initial_sha,
                        process=legacy.process(os.getpid()),implementation=expected))
                if offset%1600==0:
                    print('FRESH_BATCH',json.dumps(dict(epoch=epoch,tickets=offset+len(batch),
                        steps_total=total_steps,**stats.result()['train'])),flush=True)
            result=stats.result()
            assert result['train']['valid_samples']==2*vl_tickets and steps>0
            for i,s in streams.items():assert_unchanged(s.collector,snapshots[i])
            validate_admission(admission);scheduler.step()
            row=dict(epoch=epoch,train=result['train'],domains={pools.datasets[i].get_name():v for i,v in result['domains'].items()},
                seconds=time.monotonic()-started,tickets=30000,VL_tickets=vl_tickets,
                natural_observations=2*vl_tickets,training_anchor_observations=2*vl_tickets,
                optimizer_steps=steps,optimizer_steps_total=total_steps,action_histogram=histogram.cpu().tolist(),
                next_LR=optimizer.param_groups[0]['lr'],source_unchanged=True,
                training_stage='fresh_natural_only',official_test_evaluations=0,monitor_evaluations=0,
                collection_backend='batch_one_graph_exact_prefetch_native_context_reuse')
            payload=dict(kind=KIND,identity=expected,epoch=epoch,model=model.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),optimizer_steps_total=total_steps,
                fresh_ERC=True,old_ERC_loaded=False,admission=admission,rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
            with (directory/('FreshNaturalERC_ep%04d.pth.tar'%epoch)).open('xb') as f:
                torch.save(payload,f);f.flush();os.fsync(f.fileno())
            write_new(directory/('stats_ep%04d.json'%epoch),row);epochs.append(row)
            print('FRESH_EPOCH',json.dumps(row),flush=True)
        assert [r['epoch'] for r in epochs]==list(range(1,61)) and scheduler.get_last_lr()==[0.]
        last=directory/'FreshNaturalERC_ep0060.pth.tar'
        write_new(directory/'completed_formal60.json',dict(full_60_epochs=True,fresh_ERC=True,
            optimizer_steps_total=total_steps,final_LR=0.,
            checkpoint_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),
            seconds=time.monotonic()-start,implementation=expected,automatic_test=False))
    except Exception as e:
        write_new(directory/'failed.json',dict(error=repr(e),automatic_retry=False));raise
    finally:
        if sampler is not None:sampler.close()
        for s in streams.values():s.close()
