"""Future real24 admission only, never run by import or synthetic CPU checks.

Requires NEW explicit fixed-list GPU engineering authorization. Three independent
source trajectories: graph-light, serial-light and bare unchanged RQC control.
No optimizer, test loader, training launch or automatic repair is present.
"""
import copy
import hashlib
import random
import time
from pathlib import Path
import torch
from .checkpoint import identity,SOURCE_SHA,TRAIN_SHA
from .bridge import (original_training_components,collector_class,base_class,provider,
                     source_snapshot,assert_unchanged)
from .model import FeatureReuseCurriculumERC
from .objective import objective
from .curriculum import training_loss,anchor_view
from .geometry import candidates,apply_size,supported_actions
from .runtime import OutputOnlyERC
from .sampling import GraphSampler
from change.erc_rqc30_dense_curve_20261020_01.runtime import history,equal


def verify_real24(plan,pools,source_index,clip,params):
    expected=identity()
    if plan.get('implementation')!=expected or not plan.get('explicit_user_GPU_engineering_authorization'):
        raise PermissionError('NEW light-model real24 authorization required')
    if plan.get('observations')!=24 or plan.get('gradient_observations')!=8 or plan.get('optimizer_steps')!=0:
        raise ValueError('Fixed24/8grad/0step required')
    if plan.get('source_trajectories')!=3 or plan.get('native_graph_extra_calls')!=4 or plan.get('ERC_image_forwards')!=128:
        raise ValueError('Declare three causal trajectories, graph warmup/capture and all ERC forwards')
    if not plan.get('all_frames_fixed_before_forward') or plan.get('automatic_retry',True):
        raise ValueError('One predeclared execution required')
    if plan.get('TRAIN_sha256')!=TRAIN_SHA or plan.get('source_sha256')!=SOURCE_SHA:
        raise ValueError('Wrong source/TRAIN binding')
    budget=plan.get('wallclock_seconds')
    if not isinstance(budget,(int,float)) or not 0<budget<=300:raise ValueError('Explicit <=300s budget required')
    if any(plan.get(k) for k in ('official_test','monitor','confirmation','optimizer_training')):
        raise PermissionError('Engineering scope violation')
    legacy,Stream=original_training_components()
    if source_index not in pools.vl or clip['source']!=source_index or clip['name'] not in pools.datasets[source_index].sequence_list:
        raise ValueError('Original TRAIN member required')
    if plan.get('clip')!=dict(domain=clip['domain'],sequence=clip['name'],begin=clip['begin'],end=clip['end']):
        raise ValueError('Wrong fixed clip')
    claim=Path(plan.get('claim_file','')).resolve()
    approved=(legacy.ROOT/'change/erc_feature_reuse_curriculum_engineering_20261021_01/engineering_once.lock').resolve()
    if claim!=approved or not claim.parent.is_dir():raise ValueError('New dedicated engineering directory required')
    legacy.idle()
    with claim.open('x') as f:f.write('one approved fixed24 execution; never reset\n')
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>budget:raise TimeoutError('Budget exhausted; do not retry')
    spec,_cfg=legacy.infrastructure.settings()
    stream=Stream(pools,source_index,params,random.Random(20261021),spec,forced_clips=[clip])
    control=bare=sampler=None
    try:
        sampler=GraphSampler({source_index:stream})
        decoded={};frames=sampler.fast_frames[source_index];original_image=frames.image
        def record_image(frame):
            image=original_image(frame);decoded[frame]=hashlib.sha256(image.tobytes()).hexdigest();return image
        frames.image=record_image
        control=collector_class()(params,clip['domain']);bare=base_class()(params,clip['domain'])
        model=FeatureReuseCurriculumERC().cuda()
        class Forced(torch.nn.Module):
            def __init__(self,source):super().__init__();self.inner=copy.deepcopy(source)
            def forward(self,**inputs):
                pred=self.inner(**inputs);self.last_zero=pred
                changed=inputs['candidate_boxes'].long().ne(inputs['candidate_boxes'][:,:1].long()).any(-1)
                usable=pred['allowed']&changed;usable=usable.clone();usable[:,0]=False
                idx=torch.where(usable.any(1),usable.float().argmax(1),torch.zeros(len(usable),device=usable.device,dtype=torch.long))
                return dict(pred,index=idx,delta=self.inner.actions[idx])
        forced=Forced(model);runtime=OutputOnlyERC(control,forced,provider)
        init=dict(init_bbox=clip['initial_box'],language=clip['language'])
        runtime.initialize(clip['initial_image'],init);bare.initialize(clip['initial_image'],init)
        assert control.initialization_reused_Backbone_calls==1
        snapshots=(source_snapshot(stream.collector),source_snapshot(control),source_snapshot(bare))
        import cv2
        cursor=clip['begin']+1;seen=[];updates=0
        for j in range(24):
            deadline();row=stream.next_sample()
            if row is None:raise ValueError('Fixed clip did not supply24 observations; no replacement')
            assert plan['frames'][j]==row['meta']['frame'];seen.append(row['meta']['frame'])
            before_prompt=control.prompt.clone()
            while cursor<=row['meta']['frame']:
                frame=cursor;cursor+=1
                image=cv2.imread(str(clip['paths'][frame]))
                if image is None:raise ValueError('Missing fixed frame')
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                assert decoded.pop(frame)==hashlib.sha256(image.tobytes()).hexdigest()
                returned=runtime.track(image,{'truth':'ignored','previous_output':'ignored'})
                original=bare.track(image,None)
                assert runtime.last_rqc_box==original['target_bbox'],'Collector changed bare RQC output'
            updates+=int(not torch.equal(before_prompt,control.prompt))
            assert equal(history(control),history(bare)) and equal(history(control),history(stream.collector))
            assert row['fixed'][0].tolist()==runtime.last_rqc_box
            di,df,dh=provider(control,image,dict(target_bbox=runtime.last_rqc_box))
            assert all(torch.equal(di[k],row['inputs'][k]) for k in di)
            model.train();pred=model(**di);model.eval();deployed=model(**di)
            assert all(torch.equal(pred[k],deployed[k]) and torch.equal(pred[k],forced.last_zero[k]) for k in pred)
            assert pred['index'].eq(0).all() and pred['utility'].eq(0).all()
            assert torch.equal(di['candidate_boxes'],candidates(df,dh,model.actions))
            assert torch.equal(pred['allowed'],supported_actions(di['valid'],model.actions))
            assert returned['target_bbox']!=runtime.last_rqc_box
            if j<8:
                model.zero_grad(set_to_none=True);saved=model.score[-1].weight.detach().clone()
                with torch.no_grad():model.score[-1].weight.fill_(.001)
                pp=model(**di);natural_loss,_,_=objective(model,pp,di,df,row['truth'],dh)
                loss,aug=training_loss(model,pp,di,df,row['truth'],dh,[row],1,natural_loss)
                assert aug['anchor_views']==5 and aug['augmented_views']==4
                for delta in ((-.05,0.),(.05,0.),(0.,-.05),(0.,.05)):
                    vi,vf=anchor_view(di,df,dh,[row],delta)
                    assert torch.equal(vi['candidate_boxes'],candidates(vf,dh,model.actions))
                    torch.testing.assert_close(vf[:,:2]+vf[:,2:]/2,df[:,:2]+df[:,2:]/2,rtol=0,atol=1e-10)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                assert all(v.grad is None for v in di.values())
                with torch.no_grad():model.score[-1].weight.copy_(saved)
                model.zero_grad(set_to_none=True)
            for obj,snapshot in zip((stream.collector,control,bare),snapshots):assert_unchanged(obj,snapshot)
            deadline()
        assert seen==plan['frames'] and updates>=2 and stream.collector.initialization_reused_Backbone_calls==1
        assert not hasattr(control,'erc') and not hasattr(stream.collector,'erc')
        assert expected==identity()
        return dict(passed=True,implementation=expected,observations=24,gradient_observations=8,optimizer_steps=0,
            source_sha256=SOURCE_SHA,TRAIN_sha256=TRAIN_SHA,captured_feature_coordinates=True,
            zero_equals_RQC=True,nonzero_history_isolated=True,cross_prompt_update=True,
            source_GT_input_gradients_isolated=True,train_runtime_same_tensor_equal=True,all600_actual_actions=True,
            curriculum_geometry_verified=True,curriculum_gradients_verified=True,natural_statistics_isolated=True,
            augmented_views=32,ERC_image_forwards=128,
            original_TRAIN_membership=True,source_unchanged=True,serial_graph_same_inputs_and_history=True,
            prefetch_same_pixels=True,initialization_has_no_extra_Backbone=True,no_old_ERC_evidence_or_MLP=True,
            seconds=time.monotonic()-started,empirical_gain_proven=False,automatic_train=False,automatic_retry=False)
    finally:
        if sampler is not None:sampler.close()
        stream.close()
        if control is not None:control.close()
        if bare is not None:bare.close()

