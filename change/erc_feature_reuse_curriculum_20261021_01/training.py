"""Full60 engine, gated on NEW explicit authorization AND real engineering evidence.

No launcher, GPU diagnostic, test dispatch or auto-resume. A call with missing
admission fails before importing a tracker or allocating any CUDA tensors.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from .model import FeatureReuseCurriculumERC
from .objective import objective, summarize
from .checkpoint import KIND, identity, SOURCE_SHA, TRAIN_SHA
from .inputs import INPUT_KEYS
from .curriculum import training_loss

POLICY = dict(epochs=60,tickets_per_epoch=30000,batch_tickets=16,seed=20261016,
    VL_ratio=.44,warmup_frames=16,clip_frames=192,valid_searches_per_VL_ticket=2,
    LR=4e-5,weight_decay=1e-4,grad_clip=1.,scheduler='cosine60',
    original_TRAIN_no_new_holdout=True,source_epoch=30,fresh_ERC=True,
    automatic_test=False,automatic_retrain=False,gain_based_early_stop=False,
    perturbation_epochs=[1,40],natural_only_epochs=[41,60],
    anchor_offsets_logwh=[[0.,0.],[-.05,0.],[.05,0.],[0.,-.05],[0.,.05]],
    stage1_loss='equal_mean_over_five_anchors',stage2_loss='natural_anchor_only',
    gain_statistics='natural_RQC_anchor_only',extra_Backbone_for_perturbations=False)
COUNTS = {'valid_samples','changed_saved_boxes','integer_improved_frames',
          'integer_harmed_frames','oracle_positive_frames','changed_integer_gain_sum',
          'integer_gain_positive_sum','integer_gain_negative_sum'}


class Statistics:
    def __init__(self):
        self.values,self.weights={},{}

    def add(self,stats):
        n=stats['valid_samples']
        for key,value in stats.items():
            if not math.isfinite(float(value)):
                raise FloatingPointError('Nonfinite statistic: '+key)
            # Every retained mean is over the same valid gain observations.
            w=n
            self.values[key]=self.values.get(key,0)+float(value)*(1 if key in COUNTS else w)
            if key not in COUNTS:
                self.weights[key]=self.weights.get(key,0)+w

    def result(self):
        result={k:(v/max(1,self.weights[k]) if k in self.weights else v) for k,v in self.values.items()}
        if 'gain_MSE_pp2' in result:result['loss']=result['gain_MSE_pp2']
        return result


def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.flush();os.fsync(f.fileno())


def validate_admission(admission):
    """Typed immutable run binding; old engineering reports cannot satisfy this."""
    expected=identity()
    if not isinstance(admission,dict) or admission.get('implementation') != expected or admission.get('policy') != POLICY:
        raise ValueError('NEW implementation/policy-bound admission required')
    auth,eng,cpu=(admission.get(k,{}) for k in ('authorization','real_engineering','CPU'))
    if auth.get('scope')!='one_fresh_feature_reuse_curriculum_ERC60' or auth.get('maximum_launches')!=1 or not auth.get('explicit_user_training_authorization'):
        raise PermissionError('Implementation request is not authorization to train')
    if not auth.get('human_authorization_reference') or auth.get('automatic_test',True) or auth.get('automatic_retry',True):
        raise PermissionError('Missing training-only authorization evidence')
    if cpu.get('implementation') != expected or not cpu.get('passed') or cpu.get('GPU_operations')!=0:
        raise ValueError('Exact CPU admission required')
    if eng.get('implementation')!=expected or not eng.get('passed') or eng.get('optimizer_steps')!=0 or eng.get('source_sha256')!=SOURCE_SHA or eng.get('TRAIN_sha256')!=TRAIN_SHA:
        raise ValueError('NEW real工程验收尚未通过')
    required=('curriculum_geometry_verified','curriculum_gradients_verified','natural_statistics_isolated','captured_feature_coordinates','zero_equals_RQC','nonzero_history_isolated','cross_prompt_update',
              'source_GT_input_gradients_isolated','train_runtime_same_tensor_equal',
              'all600_actual_actions','original_TRAIN_membership','source_unchanged','serial_graph_same_inputs_and_history','prefetch_same_pixels',
              'initialization_has_no_extra_Backbone','no_old_ERC_evidence_or_MLP')
    if not all(eng.get(k) is True for k in required) or eng.get('observations')!=24:
        raise ValueError('Real24工程证据不完整')
    dependencies=admission.get('dependencies',{})
    if not dependencies or any(not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha
                               for p,sha in dependencies.items()):
        raise ValueError('Frozen dependency binding missing/changed')
    return expected


def train60(admission, output_dir):
    """No side effects until all prior authorization/engineering checks pass."""
    expected=validate_admission(admission)
    from .bridge import original_training_components,source_snapshot,assert_unchanged
    from .sampling import GraphSampler
    legacy,Stream=original_training_components()
    legacy.idle()  # No takeover of any existing task/GPU workload.
    directory=Path(output_dir).resolve()
    if directory.exists():
        raise FileExistsError('Fresh output only; never resume/reset an old lock')
    directory.mkdir(parents=False)
    write_new(directory/'launch_once.json',dict(admission=admission,process=legacy.process(os.getpid())))
    spec,cfg,plan,params=legacy.settings()
    assert spec['NATURAL_INPUT']['WARMUP_FRAMES']==16 and spec['NATURAL_INPUT']['CLIP_FRAMES']==192
    torch.set_num_threads(1)
    seed=POLICY['seed'];random.seed(seed)
    import numpy as np
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=FeatureReuseCurriculumERC().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['LR'],weight_decay=POLICY['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60)
    pools=legacy.FitPools(cfg,plan)
    streams,snapshots={},{}
    sampler=None
    total_steps=0;start=time.monotonic();epochs=[]
    write_new(directory/'protocol.json',dict(implementation=expected,policy=POLICY,
        source_sha256=SOURCE_SHA,TRAIN_sha256=TRAIN_SHA,parameters=model.parameter_count,
        old_ERC_loaded=False,old_optimizer_loaded=False,optimizer_initial_steps=0))
    try:
        for source in pools.vl:
            streams[source]=Stream(pools,source,params,random.Random(seed+source),spec)
            snapshots[source]=source_snapshot(streams[source].collector)
        # Original batch-one native forward graph and exact CPU RGB prefetch.
        # Mandatory NEW real equivalence admission above; never reuse old report.
        sampler=GraphSampler(streams)
        for epoch in range(1,61):
            model.train();started=time.monotonic();stats=Statistics()
            domains={i:Statistics() for i in pools.vl}
            tickets=pools.epoch_plan(30000,random.Random(seed+epoch))
            for i,stream in streams.items():
                stream.rng=random.Random(seed+epoch*100+i);stream.clip=None
            steps=0;vl_tickets=0;hist=[0]*25;aug_sums={};aug_weight=0
            timing=dict(collection=0.,ERC_step=0.,metrics=0.)
            for offset in range(0,30000,16):
                phase=time.monotonic()
                batch=tickets[offset:offset+16]
                rows=sampler.collect(batch)
                vl_tickets+=sum(flag==2 for flag,_source in batch)
                # One synchronization per phase gives honest disjoint timings.
                torch.cuda.synchronize()
                timing['collection']+=time.monotonic()-phase
                if not rows:
                    continue
                phase=time.monotonic()
                inputs,fixed,truth,hw=legacy.collate(rows)
                if set(inputs)!=INPUT_KEYS:
                    raise ValueError('Training integration lost reused feature inputs')
                optimizer.zero_grad(set_to_none=True)
                pred=model(**inputs)
                loss,detail,_,sample_stats=objective(model,pred,inputs,fixed,truth,hw,return_samples=True)
                loss,aug_detail=training_loss(model,pred,inputs,fixed,truth,hw,rows,epoch,loss)
                loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                if not math.isfinite(norm):
                    raise FloatingPointError('Nonfinite parameter gradient')
                optimizer.step();steps+=1;total_steps+=1
                if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                    raise FloatingPointError('Nonfinite ERC parameter')
                torch.cuda.synchronize()
                timing['ERC_step']+=time.monotonic()-phase
                phase=time.monotonic()
                stats.add(detail)
                aug_weight+=len(rows)
                for key,value in aug_detail.items():aug_sums[key]=aug_sums.get(key,0.)+float(value)*len(rows)
                hist=[a+b for a,b in zip(hist,torch.bincount(pred['index'].detach().cpu(),minlength=25).tolist())]
                with torch.no_grad():
                    for i in pools.vl:
                        selected=torch.tensor([r['meta']['source']==i for r in rows],device=fixed.device)
                        if selected.any():
                            domains[i].add(summarize(sample_stats,selected))
                torch.cuda.synchronize()
                timing['metrics']+=time.monotonic()-phase
                if total_steps==1:
                    write_new(directory/'actual_optimizer_step_01.json',dict(optimizer_steps=1,
                        optimizer_state_parameters=len(optimizer.state),process=legacy.process(os.getpid()),
                        model_parameters=model.parameter_count,loss=float(loss),gradient_norm=norm,
                        implementation=expected))
                if offset%1600==0:
                    print('BQ_BATCH',json.dumps(dict(epoch=epoch,tickets=offset+16,steps=total_steps,**detail)),flush=True)
            assert stats.values['valid_samples']==2*vl_tickets and steps>0
            for i,stream in streams.items():
                assert_unchanged(stream.collector,snapshots[i])
            if expected!=identity():
                raise RuntimeError('Implementation changed while training')
            validate_admission(admission)
            scheduler.step()
            row=dict(epoch=epoch,curriculum_stage='paired_anchor_learning' if epoch<=40 else 'natural_calibration',
                curriculum={k:v/max(1,aug_weight) for k,v in aug_sums.items()},
                natural_observations=int(stats.values['valid_samples']),
                training_anchor_observations=int(stats.values['valid_samples'])*(5 if epoch<=40 else 1),
                train=stats.result(),domains={pools.datasets[i].get_name():v.result() for i,v in domains.items()},
                optimizer_steps=steps,optimizer_steps_total=total_steps,tickets=30000,VL_tickets=vl_tickets,
                action_histogram=hist,next_LR=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-started,
                timing_seconds=timing,collection_backend='batch_one_graph_exact_prefetch',
                source_unchanged=True,monitor_evaluations=0,official_test_evaluations=0)
            payload=dict(kind=KIND,epoch=epoch,identity=expected,model=model.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),optimizer_steps_total=total_steps,
                admission=admission,rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
            with (directory/('FeatureReuseCurriculumERC_ep%04d.pth.tar'%epoch)).open('xb') as f:
                torch.save(payload,f);f.flush();os.fsync(f.fileno())
            write_new(directory/('stats_ep%04d.json'%epoch),row);epochs.append(row)
            print('BQ_EPOCH',json.dumps(row),flush=True)
        assert [r['epoch'] for r in epochs]==list(range(1,61)) and scheduler.get_last_lr()==[0.]
        last=directory/'FeatureReuseCurriculumERC_ep0060.pth.tar'
        write_new(directory/'completed_formal60.json',dict(full_60_epochs=True,optimizer_steps_total=total_steps,
            final_LR=0.,checkpoint_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),
            seconds=time.monotonic()-start,implementation=expected,automatic_test=False))
    except Exception as error:
        write_new(directory/'failed.json',dict(error=repr(error),automatic_retry=False))
        raise
    finally:
        if sampler is not None:
            sampler.close()
        for stream in streams.values():
            stream.close()

