"""Synthetic CPU engineering only. No real data, optimizer or source weights."""
import copy
import inspect
import json
from types import SimpleNamespace
import torch
from .model import FeatureReuseCurriculumERC,GAIN_UNIT,strip_grid
from .inputs import build_inputs,template_roles,INPUT_KEYS
from .geometry import candidates,apply_size,action_lattice,supported_actions
from .objective import objective,gain_rows,summarize
from .checkpoint import identity,KIND,validate_payload
from .training import validate_admission,Statistics
from .runtime import OutputOnlyERC
from . import bridge,sampling
from .engineering import verify_real24


def fixture():
    torch.manual_seed(20261021)
    model=FeatureReuseCurriculumERC()
    fixed=torch.tensor([[80.,70.,80.,60.],[4.2,5.6,35.9,40.7]],dtype=torch.double)
    hw=torch.tensor([[256.,320.],[256.,320.]],dtype=torch.double)
    origin=torch.tensor([[-16.,-24.],[-80.,-90.]],dtype=torch.double)
    side=torch.tensor([256.,256.],dtype=torch.double)
    search=torch.randn(2,256,768);head=torch.randn(2,32,16,16)
    template=torch.randn(2,64,768);fg=torch.zeros(2,64,dtype=torch.bool);fg[:,18:46]=True
    roles=template_roles(template,fg)
    bbox=torch.zeros(2,256,4);bbox[...,2:]=.2
    inputs=build_inputs(search,head,roles,bbox,fixed,origin,side,hw,torch.randn(2,64))
    truth=fixed.clone();truth[:,0]-=2;truth[:,2]+=4
    return model,inputs,fixed,truth,hw


def fail(call):
    try:call()
    except (ValueError,PermissionError,TypeError):return
    raise AssertionError('Expected rejection')


def run():
    torch.set_num_threads(1);results={}
    model,inputs,fixed,truth,hw=fixture()
    assert 30000<=model.parameter_count<=50000
    assert not any(isinstance(m,torch.nn.Conv2d) for m in model.modules())
    assert not any('rgb' in key for key in INPUT_KEYS)
    assert not any('image' in name or 'rgb' in name for name in inspect.signature(build_inputs).parameters)
    results['parameter_budget_no_RGB_encoder_or_spatial_Conv2d']=True

    pred=model(**inputs)
    assert pred['index'].eq(0).all() and pred['utility'].eq(0).all()
    assert torch.equal(pred['allowed'],supported_actions(inputs['valid'],model.actions))
    assert torch.equal(inputs['candidate_boxes'],candidates(fixed,hw,model.actions))
    results['zero_KEEP_original_25_actions_and_support']=True

    # Image integer execution precedes crop conversion; no crop-space rounding.
    expected=(inputs['candidate_boxes'][1,0].long().double()[:2]-torch.tensor([-80.,-90.]))/256
    torch.testing.assert_close(inputs['saved_crop_boxes'][1,0,:2].double(),expected,rtol=0,atol=1e-7)
    assert strip_grid(inputs['crop_boxes']).shape==(2,200,5,2)
    results['actual_writer_before_crop_and_ordered_1000_strip_points']=True

    with torch.no_grad():model.score[-1].weight.normal_(0,.01)
    duplicate=copy.deepcopy(inputs)
    for key in ('candidate_boxes','crop_boxes','saved_crop_boxes','valid'):
        duplicate[key][:,1]=duplicate[key][:,0]
    dp=model(**duplicate)
    torch.testing.assert_close(dp['score'][:,1],dp['score'][:,0],rtol=0,atol=0)
    permutation=torch.tensor([0]+list(range(24,0,-1)))
    permuted=copy.deepcopy(inputs)
    for key in ('candidate_boxes','crop_boxes','saved_crop_boxes','valid'):
        permuted[key]=permuted[key][:,permutation]
    torch.testing.assert_close(model(**permuted)['score'],model(**inputs)['score'][:,permutation],rtol=1e-5,atol=1e-6)
    one={k:v[:1] for k,v in inputs.items()}
    torch.testing.assert_close(model(**one)['score'],model(**inputs)['score'][:1],rtol=1e-5,atol=1e-6)
    results['duplicate_geometry_permutation_and_batch_independence']=True

    gi={k:v.detach().clone().requires_grad_(True) if v.dtype.is_floating_point else v.clone() for k,v in inputs.items()}
    gt=truth.clone().requires_grad_(True)
    pred=model(**gi);loss,stats,_=objective(model,pred,gi,fixed,gt,hw)
    loss.backward()
    assert gt.grad is None and all(v.grad is None for v in gi.values())
    for prefix in ('search','head','context','point','score'):
        grads=[p.grad for k,p in model.named_parameters() if k.startswith(prefix)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads),prefix
        assert sum(float(g.abs().sum()) for g in grads)>0,prefix
    model.eval();assert all(torch.equal(v,model(**inputs)[k]) for k,v in pred.items())
    model.train()
    results['all_branch_gradients_GT_inputs_detached_train_deploy_equal']=True

    param=model.score[-1].weight;where=(0,7);analytic=float(param.grad[where]);eps=1e-3
    def scalar():return float(objective(model,model(**inputs),inputs,fixed,truth,hw)[0].detach())
    with torch.no_grad():
        before=float(param[where]);param[where]=before+eps;plus=scalar()
        param[where]=before-eps;minus=scalar();param[where]=before
    numerical=(plus-minus)/(2*eps)
    assert abs(numerical-analytic)<5e-4+5e-3*abs(analytic),(analytic,numerical)
    results['exact_gain_loss_central_difference']=dict(analytic=analytic,numerical=numerical)

    scores=torch.linspace(-.3,.4,50,dtype=torch.double).reshape(2,25).requires_grad_(True)
    labels=torch.linspace(-.04,.02,50,dtype=torch.double).reshape(2,25);labels[:,0]=0
    mask=torch.ones(2,25,dtype=torch.bool);mask[:,7]=False
    utility=GAIN_UNIT*(scores-scores[:,:1]);l=gain_rows(utility,labels,mask).mean()
    derivative,=torch.autograd.grad(l,scores)
    expected=2*((utility.detach()-labels)/GAIN_UNIT)*mask/(2*24)
    expected[:,0]=-expected[:,1:].sum(1)
    torch.testing.assert_close(derivative,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(gain_rows(utility,labels,mask),gain_rows(GAIN_UNIT*((scores+10)-(scores+10)[:,:1]),labels,mask),rtol=1e-12,atol=1e-12)
    assert gain_rows(labels,labels,mask).eq(0).all()
    results['proper_gain_gradient_KEEP_coupling_support_and_units']=True

    partial=copy.deepcopy(inputs)
    invalid=~partial['source_mask']
    for key in ('search','head','native_extent'):
        partial[key]=partial[key].masked_fill(invalid,float('nan'))
    pp=model(**partial);assert all(torch.isfinite(v).all() for v in pp.values())
    empty=copy.deepcopy(inputs);empty['source_mask'].zero_()
    for key in ('search','head','native_extent'):empty[key].fill_(float('nan'))
    assert model(**empty)['index'].eq(0).all() and model(**empty)['utility'].eq(0).all()
    only=copy.deepcopy(inputs);only['valid'].zero_();assert model(**only)['index'].eq(0).all()
    fail(lambda:model(**dict(inputs,search_rgb=torch.zeros(2,3,192,192))))
    fail(lambda:model(**dict(inputs,truth=truth)))
    fail(lambda:model(**dict(inputs,source_context=torch.full_like(inputs['source_context'],float('nan')))))
    results['padding_NaN_empty_KEEP_and_no_RGB_GT_input']=True

    pred=model(**inputs);full,detail,_,samples=objective(model,pred,inputs,fixed,truth,hw,return_samples=True)
    assert abs(float(full)-detail['loss'])<1e-10
    for membership in (torch.tensor([True,False]),torch.tensor([False,True])):
        sp={k:v[membership] for k,v in pred.items()};si={k:v[membership] for k,v in inputs.items()}
        _,again,_=objective(model,sp,si,fixed[membership],truth[membership],hw[membership])
        reused=summarize(samples,membership)
        assert all(abs(again[k]-reused[k])<1e-10 for k in again)
    accumulation=Statistics();accumulation.add(detail)
    assert accumulation.result()['loss']==detail['loss']
    results['domain_statistics_reuse_matches_exact_subset']=True

    ident=identity();payload=dict(kind=KIND,identity=ident,epoch=60,optimizer_steps_total=1,model=copy.deepcopy(model.state_dict()))
    target=FeatureReuseCurriculumERC();validate_payload(payload,target,ident)
    assert all(torch.equal(v,target.state_dict()[k]) for k,v in model.state_dict().items())
    for bad in (dict(payload,kind='RQC30_original_TRAIN_boundary_gain_ERC_20261021_02'),dict(payload,diagnostic_only=True),dict(payload,optimizer_steps_total=0)):
        fail(lambda:validate_payload(bad,target,ident))
    bad=copy.deepcopy(payload);bad['model']['score.3.weight'][0,0]=float('nan')
    fail(lambda:validate_payload(bad,target,ident));fail(lambda:validate_admission({}))
    fail(lambda:verify_real24({},None,None,None,None))
    results['strict_new_checkpoint_and_no_training_authorization_gate']=True

    # Exercise the actual adapter against a deterministic CPU native source.
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=0;self.tokens=torch.arange(64*768,dtype=torch.float).reshape(1,64,768)/10000
        def forward(self,*_args):self.calls+=1;return dict(template=self.tokens.clone())
    class Network(torch.nn.Module):
        def __init__(self):super().__init__();self.backbone=Backbone()
        def forward_test(self,*_args):
            return dict(search=inputs['search'][:1].flatten(2).transpose(1,2),head=inputs['head'][:1],
                bbox_map=torch.cat((torch.zeros(1,256,2),torch.full((1,256,2),.2)),2),context=inputs['source_context'][:1])
    class FakeRQC:
        def __init__(self,*_args,**_kwargs):
            self.network=Network();self.params=SimpleNamespace(search_factor=4.)
        def initialize(self,image,info):
            self.state=info['init_bbox'][:];self.frame_id=0;self.prompt=torch.zeros(1)
            self.template_mask=torch.zeros(1,64,dtype=torch.bool);self.template_mask[:,18:46]=True
            self.network.backbone(None) # Exactly the existing native prompt-initialization call.
        def track(self,image,info=None):
            assert info is None;self.frame_id+=1
            self.state[0]+=.1;self.last_native_box=self.state[:]
            if self.frame_id%2==0:self.prompt=self.prompt+1
            self.network.forward_test(None)
            rqc=self.state[:];rqc[0]+=.125
            return dict(target_bbox=rqc)
    original_base,original_fields=bridge.base_class,bridge.captured_fields
    bridge.base_class=lambda:FakeRQC
    bridge.captured_fields=lambda src,out:(out['search'],out['head'],out['bbox_map'],out['context'])
    try:
        Collector=bridge.collector_class();source=Collector(None,'CPU');control=FakeRQC()
        image=torch.zeros(256,320,3,dtype=torch.uint8)
        info=dict(init_bbox=[80.,70.,80.,60.])
        source.initialize(image,info);control.initialize(image,info)
        assert source.network.backbone.calls==control.network.backbone.calls==1
        assert not source.network.backbone._forward_hooks and not hasattr(source,'erc')
        for _ in range(4):
            a=source.track(image);b=control.track(image)
            assert a==b and source.state==control.state and torch.equal(source.prompt,control.prompt)
            built,fb,ih=bridge.provider(source,image,a)
            assert set(built)==INPUT_KEYS and fb[0].tolist()==a['target_bbox']
        assert source.network.backbone.calls==1 # No hidden per-frame second Backbone call.
        fail(lambda:bridge.provider(source,image,dict(target_bbox=[0.,0.,10.,10.])))

        # Actual new CPU prefetch adapter: ordered frames, no legacy ERC member,
        # current search GT is read only after that frame has been inferred.
        source.initialize(image,info)
        class Annotations:
            def __getitem__(self,frame):
                assert source.frame_id>=frame,'GT was read before source inference'
                return torch.tensor([80.,70.,80.,60.])
        stream=SimpleNamespace(collector=source,clip=dict(begin=0,end=5,paths=list(range(5)),
            visible=[True]*5,annotations=Annotations(),domain='CPU',name='fixed'),cursor=1,
            spec={'WARMUP_FRAMES':2},frames=0,seconds=0.,source=1,initial_prompt=source.prompt.clone(),
            next_sample=lambda:None)
        original_decode=sampling.read_rgb
        sampling.read_rgb=lambda path:torch.zeros(256,320,3,dtype=torch.uint8)+int(path)
        frames=sampling.PrefetchFrames(stream)
        try:
            frames.enable();a=stream.next_sample();b=stream.next_sample()
            assert [a['meta']['frame'],b['meta']['frame']]==[3,4]
            assert stream.frames==4 and source.frame_id==4 and not hasattr(source,'erc')
        finally:frames.close();sampling.read_rgb=original_decode
        results['CPU_prefetch_causal_order_warmup_and_GT_after_forward']=True
    finally:bridge.base_class,bridge.captured_fields=original_base,original_fields
    results['CPU_adapter_reuses_init_Backbone_no_old_ERC_and_stale_guard']=True

    class Source:
        def initialize(self,image,info):self.state=info['init_bbox'][:];self.prompt=torch.zeros(1);self.frame_id=0
        def track(self,image,info=None):
            assert info is None;self.frame_id+=1;self.state[0]+=.1;self.last_native_box=self.state[:]
            if self.frame_id%2==0:self.prompt+=1
            return dict(target_bbox=self.state[:])
    class Forced(torch.nn.Module):
        def __init__(self,nonzero):super().__init__();self.nonzero=nonzero
        def forward(self,**_kw):
            index=torch.tensor([4 if self.nonzero else 0]);return dict(index=index,delta=action_lattice()[index])
    def get(src,image,result):return {},torch.tensor([result['target_bbox']],dtype=torch.double),hw[:1]
    for mode in (False,True):
        src,ctrl=Source(),Source();runtime=OutputOnlyERC(src,Forced(mode),get)
        init={'init_bbox':[80.,70.,80.,60.]};runtime.initialize(None,init);ctrl.initialize(None,init);changed=False
        for _ in range(4):
            a=runtime.track(None,{'truth':'ignored'});b=ctrl.track(None)
            assert src.state==ctrl.state and torch.equal(src.prompt,ctrl.prompt)
            changed|=a['target_bbox']!=b['target_bbox']
        assert changed==mode
    results['CPU_zero_nonzero_history_isolation_across_prompt_updates']=True

    # Wire the original replay adapter by interface without running CUDA capture.
    class Replay:
        def __init__(self,tracker):self.native=tracker._original_forward
    dummy=SimpleNamespace(collector=SimpleNamespace(_original_forward=lambda:None),
        next_sample=lambda:None,next_pair=lambda:['first','second'])
    original_import=sampling.importlib.import_module
    def replay_module(name):
        if name=='graph_sampler':return SimpleNamespace(GraphForward=Replay)
        return original_import(name)
    sampling.importlib.import_module=replay_module
    sampler=None;native=dummy.collector._original_forward
    try:
        sampler=sampling.GraphSampler({1:dummy})
        assert dummy.collector._original_forward is sampler.adapters[1]
        assert sampler.collect([(0,1),(2,1),(1,1),(2,1)])==['first','second','first','second']
    finally:
        if sampler is not None:sampler.close()
        sampling.importlib.import_module=original_import
    assert dummy.collector._original_forward is native
    results['CPU_graph_adapter_ticket_order_and_restore_without_CUDA']=True

    from .curriculum_checks import run_curriculum_checks
    results.update(run_curriculum_checks())
    return dict(passed=True,CPU_suite_pass=True,checks=results,parameters=model.parameter_count,
        implementation=identity(),optimizer_steps=0,GPU_operations=0,real_observations=0,
        real_source_equivalence_verified=False,speedup_measured=False,empirical_gain_proven=False,
        training_started=False,test_started=False,torch_version=torch.__version__)


if __name__=='__main__':print('FEATURE_REUSE_CURRICULUM_CPU_REPORT='+json.dumps(run(),ensure_ascii=False,allow_nan=False))

