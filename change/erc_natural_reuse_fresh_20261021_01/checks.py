"""CPU-only fresh initialization, gradients and native-context regression. Optimizer0."""
import copy,inspect,json,sys
from pathlib import Path
import torch
from .model import FreshNaturalERC
from .objective import objective
from .checkpoint import identity,validate_payload,KIND
from .training import validate_admission,fresh_state,POLICY
from .device_stats import DeviceStatistics
from .native_tap import NativeContextTap
from . import bridge


def must_fail(call):
    try:call()
    except (ValueError,PermissionError,RuntimeError):return
    raise AssertionError('Expected a fail-closed rejection')


def run():
    torch.set_num_threads(1)
    from change.erc_feature_reuse_curriculum_20261021_01.checks import fixture
    from change.erc_feature_reuse_curriculum_20261021_01.objective import objective as old_objective
    from change.erc_feature_reuse_curriculum_20261021_01.checkpoint import identity as old_identity
    before=old_identity();old,inputs,fixed,truth,hw=fixture();tests={}
    model=FreshNaturalERC();assert model.parameter_count==old.parameter_count==34987
    assert model.state_dict().keys()==old.state_dict().keys()
    zero=model(**inputs)
    assert zero['index'].eq(0).all() and zero['utility'].eq(0).all()
    tests['same_parameter_schema_zero_KEEP_and_25_actions']=True

    model,optimizer,scheduler=fresh_state('cpu')
    first={k:v.clone() for k,v in model.state_dict().items()}
    repeat,other_optimizer,other_scheduler=fresh_state('cpu')
    assert all(torch.equal(v,repeat.state_dict()[k]) for k,v in first.items())
    assert not optimizer.state and not other_optimizer.state and scheduler.last_epoch==other_scheduler.last_epoch==0
    assert optimizer.param_groups[0]['lr']==4e-5 and scheduler.T_max==60
    old.load_state_dict(first,strict=True)
    actual=model(**inputs);expected=old(**inputs)
    assert actual.keys()==expected.keys() and all(torch.equal(actual[k],expected[k]) for k in actual)
    tests['fresh_seed_reproducible_zero_optimizer_scheduler_and_no_checkpoint_load']=True
    with torch.no_grad():
        model.score[-1].weight.normal_(0,.01)
    old.load_state_dict(model.state_dict(),strict=True)
    actual=model(**inputs);expected=old(**inputs)
    assert all(torch.equal(actual[k],expected[k]) for k in actual)
    tests['shared_natural_model_predictions_match_reference']=True

    l,stats,picked=objective(model,actual,inputs,fixed,truth,hw)
    ol,os,op=old_objective(old,expected,inputs,fixed,truth,hw)
    assert torch.equal(l,ol) and torch.equal(picked,op) and stats==os
    tests['unchanged_gain_labels_support_loss_writer_and_statistics']=True

    for k,v in inputs.items():
        if v.is_floating_point():inputs[k]=v.detach().clone().requires_grad_(True)
    gt=truth.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True);old.zero_grad(set_to_none=True)
    packet,samples,chosen=objective(model,model(**inputs),inputs,fixed,gt,hw,return_device_samples=True)
    oldloss,_,_=old_objective(old,old(**inputs),inputs,fixed,gt,hw)
    packet.backward();oldloss.backward()
    for (k,p),(ok,q) in zip(model.named_parameters(),old.named_parameters()):
        assert k==ok and p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert gt.grad is None and all(v.grad is None for v in inputs.values())
    tests['packet_loss_gradients_exact_and_source_GT_detached']=True

    accumulator=DeviceStatistics([1,2,3]);ids=torch.tensor([1,2])
    accumulator.add(samples,ids);accumulator.add(samples,ids)
    summary=accumulator.result()
    for k,v in stats.items():
        expected_value=2*v if k in __import__(__package__+'.objective',fromlist=['COUNTS']).COUNTS else v
        assert abs(summary['train'][k]-expected_value)<1e-10,k
    assert summary['domains'][3]['valid_samples']==0 and summary['domains'][3]['loss']==0
    tests['device_accumulation_matches_old_weighting_and_empty_domains']=True

    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'change/rqc_dev_v3'))
    from directional_center_refiner_20261001 import DirectionalCenterRefiner
    torch.manual_seed(20261021)
    first=DirectionalCenterRefiner(.1).eval()
    for p in first.parameters():p.requires_grad_(False)
    features=dict(context=torch.randn(2,288),pairs=torch.randn(2,2,2,96))
    boxes=torch.tensor([[.5,.5,.2,.3],[.4,.5,.1,.2]])
    original=first(features,boxes)
    tap=NativeContextTap(first);calls=[]
    counter=first.net[-2].register_forward_hook(lambda m,a,r:calls.append(1))
    try:
        tap.begin();out=first(features,boxes);reused=tap.finish();tap.cancel()
        assert len(calls)==2 and all(torch.equal(original[k],out[k]) for k in out)
        c=features['context'][:,None].expand(-1,2,-1);pos,neg=features['pairs'][:,:,0],features['pairs'][:,:,1]
        reference=((first.net[:-1](torch.cat((c,pos,neg),-1))+first.net[:-1](torch.cat((c,neg,pos),-1)))*.5).mean(1)
        assert len(calls)==4 and torch.equal(reused,reference)
        assert not tap.active and not tap.values
        tap.begin();must_fail(tap.finish);tap.cancel()
        tap.begin();first(features,boxes);must_fail(lambda:first(features,boxes));tap.cancel()
    finally:counter.remove();tap.close()
    assert not first.net[-2]._forward_hooks
    tests['native_hidden_reuse_exact_4_to_2_passes_no_output_replacement']=True
    tests['tap_fail_closed_missing_extra_calls_and_cleanup']=True

    assert 'gather_directional' not in inspect.getsource(bridge.captured_fields)
    assert 'frozen_context' not in inspect.getsource(bridge.captured_fields)
    assert 'NativeContextTap' in inspect.getsource(bridge.collector_class)
    tests['hot_path_no_second_RQC_feature_gather_or_hidden_network']=True
    must_fail(lambda:validate_admission({}))
    payload=dict(kind=KIND,identity=identity(),model=copy.deepcopy(model.state_dict()),
        epoch=1,optimizer_steps_total=1,fresh_ERC=True,old_ERC_loaded=False)
    validate_payload(payload,FreshNaturalERC(),identity())
    must_fail(lambda:validate_payload(dict(payload,parent_sha256='forbidden'),FreshNaturalERC(),identity()))
    must_fail(lambda:validate_payload(dict(payload,epoch=0),FreshNaturalERC(),identity()))
    tests['no_launch_without_authorization_and_strict_fresh_checkpoint_provenance']=True
    assert POLICY['training_views']==1 and POLICY['natural_RQC_only']
    assert POLICY['tickets_per_epoch']==30000 and POLICY['valid_searches_per_VL_ticket']==2
    tests['natural_only_original_TRAIN_budget_no_40_epoch_wait_or_5_views']=True
    assert old_identity()==before and not torch.cuda.is_initialized()
    return dict(passed=True,checks=tests,implementation=identity(),parameters=34987,
        fresh_ERC=True,optimizer_initial_steps=0,scheduler_initial_epoch=0,
        optimizer_steps=0,GPU_operations=0,real_observations=0,
        native_CPU_hidden_reuse_exact=True,new_GPU_speedup_proven=False,
        positive_gain_proven=False,training_started=False,test_started=False)
