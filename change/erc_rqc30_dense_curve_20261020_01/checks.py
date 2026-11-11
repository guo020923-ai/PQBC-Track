"""CPU synthetic admission only. No optimizer, real images, weights or dataset."""
import ast
import copy
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import DenseSpatialCurveERC,MODEL_PARAMETERS,decode_coefficients,supported_actions,exact_gelu
from .inputs import build_inputs,map_mask,sample_map
from .evidence import template_association_map,crop_geometry
from . import geometry as g
from .gain_risk_objective import loss_from_targets
from .adapter import SecondOrderObjectiveAdapter
from .runtime import OutputOnlyERC,equal
from .checkpoint import SOURCE_SHA,SPLIT_SHA,KIND,checkpoint_identity,validate_payload,semantics
from .provider import RQC30Provider

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'change/erc_rqc30_spatial_association_20261020_01'

def fake_inputs(b=2):
    mask=torch.ones(b,1,16,16,dtype=torch.bool)
    roi=torch.randn(b,951,16,16);roi[:,950:951]=mask.float()
    return dict(roi_evidence=roi.requires_grad_(),roi_valid=mask,
        valid=torch.ones(b,25,2,2,2,5,dtype=torch.bool),
        source_context=torch.randn(b,64,requires_grad=True),
        executed_delta=g.action_lattice()[None].expand(b,-1,-1).clone().requires_grad_(True))

def source_args(b=2):
    fg=torch.zeros(b,64,dtype=torch.bool);fg[:,:32]=True
    return dict(search=torch.randn(b,256,768,requires_grad=True),
        template=torch.randn(b,64,768,requires_grad=True),foreground=fg,
        bbox_features=torch.randn(b,32,16,16,requires_grad=True),
        fixed_xywh=torch.tensor([[30.,30.,60.,50.],[0.,0.,20.,30.]],dtype=torch.double)[:b],
        origin=torch.tensor([[0.,0.],[-16.,-16.]],dtype=torch.double)[:b],
        side=torch.tensor([192.,96.],dtype=torch.double)[:b],
        image_hw=torch.tensor([[300.,400.],[200.,200.]],dtype=torch.double)[:b],
        source_context=torch.randn(b,64,requires_grad=True),
        search_rgb=torch.randn(b,3,256,256,requires_grad=True),
        template_rgb=torch.randn(b,3,128,128,requires_grad=True),
        bbox_map=torch.cat((torch.rand(b,256,2),torch.rand(b,256,2)+.1),-1),
        actions=g.action_lattice())

def reject(fn):
    try:fn()
    except (ValueError,FloatingPointError):return
    raise AssertionError('Invalid fixture was accepted')

def check_parity():
    import importlib
    old=importlib.import_module('change.erc_rqc30_spatial_association_20261020_01.inputs')
    old_model=importlib.import_module('change.erc_rqc30_spatial_association_20261020_01.model')
    for f in ('geometry.py','objective.py','gain_risk_objective.py','runtime.py'):
        ours=ast.dump(ast.parse((Path(__file__).parent/f).read_text()))
        theirs=ast.dump(ast.parse((OLD/f).read_text()))
        assert ours==theirs,f
    assert ast.dump(ast.parse(inspect.getsource(supported_actions)))==ast.dump(ast.parse(inspect.getsource(old_model.supported_actions)))
    assert ast.dump(ast.parse(inspect.getsource(decode_coefficients)))==ast.dump(ast.parse(inspect.getsource(old_model.decode_coefficients)))
    args=source_args();built=build_inputs(**args)
    legacy=old.build_inputs(**args,initial_refs=torch.randn(2,5,768),current_refs=torch.randn(2,5,768))
    assert torch.equal(built['valid'],legacy['valid'])
    assert torch.equal(built['executed_delta'],legacy['executed_delta'])
    assert not any(v.requires_grad for v in built.values())
    assert 'truth' not in inspect.signature(build_inputs).parameters
    gt=args['fixed_xywh'].clone();gt[:,2:]*=1.03
    labels=g.action_targets(args['fixed_xywh'],gt,args['image_hw'],g.action_lattice())
    assert torch.equal(labels['utility'],torch.minimum(labels['float_gain'],labels['integer_gain'])/.01)
    assert torch.equal(labels['boxes'][:,0],args['fixed_xywh'])
    center=labels['boxes'][...,:2]+labels['boxes'][...,2:]/2
    torch.testing.assert_close(center,(gt[:,:2]+args['fixed_xywh'][:,2:]/2)[:,None].expand(-1,25,-1),rtol=0,atol=1e-12)
    assert not torch.equal(built['executed_delta'][1],g.action_lattice())
    return dict(core_AST_equal=True,action_support_bitwise_equal=True,actual_geometry_equal=True,
        source_inputs_detached=True,GT_not_input=True,per_sample_min_integer_writer=True,duplicate_nominal_actions_retained=True)

def check_roi():
    args=source_args();first=build_inputs(**args)
    assert first['roi_evidence'].shape==(2,951,16,16)
    # Linear coordinate ramps recover analytic ROI points away from interpolation padding.
    coords=(torch.arange(16)+.5)/16
    yy,xx=torch.meshgrid(coords,coords,indexing='ij')
    ramp=torch.stack((xx,yy),0)[None]
    box,bounds=crop_geometry(args['fixed_xywh'][:1],args['origin'][:1],args['side'][:1],args['image_hw'][:1])
    rel=first['roi_evidence'][:1,948:950]
    expected=box[:,None,None,:2]+box[:,None,None,2:]/2+rel.permute(0,2,3,1)*box[:,None,None,2:]
    sampled=sample_map(ramp,expected*2-1,bounds)
    torch.testing.assert_close(sampled,expected.permute(0,3,1,2),rtol=1e-6,atol=1e-7)
    # Padding NaNs removed BEFORE token normalization, RGB neighborhoods and interpolation.
    dirty={k:v.detach().clone() for k,v in args.items()}
    clean={k:v.detach().clone() for k,v in args.items()}
    _,bounds=crop_geometry(args['fixed_xywh'],args['origin'],args['side'],args['image_hw'])
    for key in ('search','bbox_map','bbox_features','search_rgb'):
        if key in ('search','bbox_map'):
            m=map_mask(bounds,16,16).flatten(2).transpose(1,2)
        else:m=map_mask(bounds,*args[key].shape[-2:])
        dirty[key]=torch.where(m,dirty[key],torch.full_like(dirty[key],float('nan')))
        clean[key]=torch.where(m,clean[key],torch.ones_like(clean[key]) if key=='bbox_map' else torch.zeros_like(clean[key]))
    a,b=build_inputs(**dirty),build_inputs(**clean)
    for k in a:assert torch.equal(a[k],b[k]),k
    # Native log extent uses the same interpolation support on subtraction.
    # At crop x=0, constant log native/RQC ratio must be half-weighted.
    bounds=torch.tensor([[0.,0.,1.,1.]])
    grid=torch.tensor([[[[-1.,0.]]]])
    native=torch.full((1,2,16,16),.25)
    support=sample_map(torch.ones(1,1,16,16),grid,bounds)
    ratio=sample_map(native.log(),grid,bounds)-support*math.log(.2)
    expected_ratio=(support*math.log(.25/.2)).expand_as(ratio)
    torch.testing.assert_close(ratio,expected_ratio,rtol=1e-6,atol=1e-7)
    search=args['search'];template=args['template'];fg=args['foreground']
    bank=template_association_map(search,template,fg)
    perm=torch.arange(64);perm[0]=1;perm[1]=0
    other=template_association_map(search,template[:,perm],fg)
    assert not torch.equal(bank,other)
    torch.testing.assert_close(other[:,:64],bank[:,perm],rtol=0,atol=0)
    assert bank[:,32:96].eq(0).all()
    bad=dict(args,inherited_valid=~first['valid'])
    reject(lambda:build_inputs(**bad))
    return dict(channels=951,grid=16,analytic_xy_ramp=True,NaN_padding_before_normalization=True,
        native_log_ratio_padding_support=True,FG_BG_order_preserved=True,collector_support_mismatch_rejected=True)

def check_model():
    model=DenseSpatialCurveERC()
    assert sum(p.numel() for p in model.parameters())==MODEL_PARAMETERS==120257
    inputs=fake_inputs();initial=model(**inputs)
    assert initial['index'].eq(0).all() and initial['utility'][:,0].eq(0).all()
    assert torch.linalg.eigvalsh(initial['curvature']).gt(0).all()
    with torch.no_grad():model.curve[-1].weight.normal_(std=.003)
    all_grad={}
    model(**inputs)['utility'].square().mean().backward()
    for name in ('search','association','head','pixels','fuse','spatial','reduce','curve'):
        params=list(getattr(model,name).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),name
        assert all(p.grad.abs().sum()>0 for p in params),name
        all_grad[name]=sum(float(p.grad.abs().sum()) for p in params)
    assert all(v.grad is None for v in inputs.values() if v.is_floating_point())
    baseline=model.encode(**inputs)[0].detach()
    interior={k:v.detach().clone() for k,v in inputs.items()}
    interior['roi_evidence'][:,0,7:9,7:9]+=2
    assert not torch.equal(baseline,model.encode(**interior)[0])
    for mode in ('empty','stripe','single','partial'):
        case={k:v.detach().clone() for k,v in inputs.items()}
        mask=torch.zeros_like(case['roi_valid'])
        if mode=='stripe':mask[:,:,:,7]=True
        if mode=='single':mask[:,:,7,7]=True
        if mode=='partial':mask[:,:,:8,:8]=True
        case['roi_valid']=mask
        case['roi_evidence'][:,950:951]=mask.float()
        clean={k:v.clone() for k,v in case.items()}
        case['roi_evidence']=torch.where(mask,case['roi_evidence'],torch.full_like(case['roi_evidence'],float('nan')))
        clean['roi_evidence']=torch.where(mask,clean['roi_evidence'],torch.zeros_like(clean['roi_evidence']))
        pred=model(**case);ref=model(**clean)
        for key in pred:assert torch.equal(pred[key],ref[key]),(mode,key)
        assert torch.equal(pred['allowed'],initial['allowed'])
        if mode=='empty':assert pred['index'].eq(0).all() and pred['direction'].eq(0).all()
    model.train();train=model(**inputs);model.eval();evaluation=model(**inputs)
    for key in train:assert torch.equal(train[key],evaluation[key])
    assert not any(isinstance(m,(nn.Dropout,nn.modules.batchnorm._BatchNorm)) for m in model.modules())
    return dict(parameters=MODEL_PARAMETERS,summary_dimension=1152,zero_KEEP=True,
        PSD=True,nonzero_fixture_branch_gradient_L1=all_grad,input_gradients_none=True,
        mask_cases=4,empty_raw_zero_support_unchanged=True,interior_encoding_sensitive=True,
        synthetic_legacy_observation_held_constant_not_raw_image_collision=True,train_eval_bitwise_equal=True)

def check_derivatives():
    # Independent analytic GELU derivative, including noncontiguous upstream gradient.
    x=torch.linspace(-4,4,2*16*16*24).reshape(2,16,16,24).requires_grad_()
    upstream=torch.randn(2,24,16,16).permute(0,2,3,1)
    y=exact_gelu(x)
    y.backward(upstream)
    analytic=.5*(1+torch.erf(x.detach()/math.sqrt(2)))+x.detach()*torch.exp(-x.detach().square()/2)/math.sqrt(2*math.pi)
    torch.testing.assert_close(x.grad,upstream*analytic,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(y.detach(),F.gelu(x.detach()),rtol=1e-5,atol=1e-6)
    model=DenseSpatialCurveERC();inp=fake_inputs()
    with torch.no_grad():model.curve[-1].weight.normal_(std=.003)
    model(**inp)['utility'].square().mean().backward()
    differences=[]
    for name in ('search.1.weight','association.1.weight','head.1.weight','pixels.1.weight',
                 'fuse.weight','spatial.0.weight','reduce.weight','curve.3.weight'):
        p=dict(model.named_parameters())[name]
        # Predefined representative entry, not selection by gradient agreement.
        idx=tuple(0 for _ in p.shape);value=float(p[idx]);step=1e-3
        automatic=float(p.grad[idx])
        with torch.no_grad():
            p[idx]=value+step;a=float(model(**inp)['utility'].square().mean())
            p[idx]=value-step;b=float(model(**inp)['utility'].square().mean())
            p[idx]=value
        numerical=(a-b)/(2*step)
        assert abs(automatic-numerical)<=1e-7+abs(numerical)*.02,(name,automatic,numerical)
        differences.append(dict(parameter=name,automatic=automatic,central_difference=numerical))
    return dict(exact_GELU_analytic_derivative=True,strided_upstream=True,
        forward_equivalent_to_GELU=True,parameter_finite_differences=differences,
        CPU_kernel_workaround_not_old_training_root_cause_proof=True)

def check_loss():
    score=torch.randn(3,25,dtype=torch.double,requires_grad=True)
    truth=torch.randn(3,25,dtype=torch.double,requires_grad=True)
    with torch.no_grad():score[:,0]=0;truth[:,0]=0
    allowed=torch.ones(3,25,dtype=torch.bool);allowed[0,2:7]=False
    delta=torch.zeros(3,25,2,requires_grad=True)
    loss,_=loss_from_targets(dict(utility=score,allowed=allowed),delta,truth);loss.backward()
    active=allowed.clone();active[:,0]=False
    analytic=torch.where(active,score.detach()-truth.detach(),torch.zeros_like(score))/(3*24)
    torch.testing.assert_close(score.grad,analytic,rtol=0,atol=1e-14)
    assert truth.grad is None and delta.grad is None
    inputs=fake_inputs();model=DenseSpatialCurveERC()
    fixed=torch.tensor([[40.,50.,100.,80.],[30.,20.,80.,60.]],dtype=torch.double,requires_grad=True)
    hw=torch.tensor([[300.,400.],[300.,400.]],dtype=torch.double)
    gt=(fixed.detach()+torch.tensor([[0.,0.,3.,-2.],[0.,0.,-3.,2.]])).requires_grad_(True)
    inputs['executed_delta']=(g.candidates(fixed,hw,g.action_lattice())[:,:,2:]/fixed.detach()[:,None,2:]).log().float().requires_grad_()
    adapter=SecondOrderObjectiveAdapter(model)
    loss,stats,pred=adapter(inputs,fixed,gt,hw);loss.backward()
    assert model.curve[-1].weight.grad.abs().sum()>0
    assert gt.grad is None and fixed.grad is None
    assert all(v.grad is None for v in inputs.values() if v.is_floating_point())
    model.eval();other,stats2,_=adapter.monitor(inputs,fixed,gt,hw)
    assert torch.equal(other,loss.detach()) and stats==stats2
    return dict(analytic_gradient_max_error=float((score.grad-analytic).abs().max()),
        fixed24=True,GT_geometry_source_detached=True,fit_monitor_same_objective=True,optimizer_steps=0)

class FakeSource:
    def initialize(self,image,info):
        self.state=[40.,50.,100.,80.];self.last_native_box=list(self.state)
        self.prompt=torch.zeros(1,5,8);self.frame_id=0
        self.out_dict={'shared':[torch.ones(2)]}
    def track(self,image,info=None):
        self.frame_id+=1;self.state[0]+=.25;self.last_native_box=list(self.state)
        if self.frame_id%5==0:self.prompt=self.prompt+1
        fixed=list(self.state);fixed[0]+=2
        return dict(target_bbox=fixed,metadata=self.out_dict)

class ForcedActionFixture(nn.Module):
    def forward(self,**inputs):
        return dict(index=torch.ones(1,dtype=torch.long),delta=g.action_lattice()[1:2])

def check_runtime():
    fi=fake_inputs(1)
    def provider(source,image,result,previous):
        fixed=torch.tensor([result['target_bbox']],dtype=torch.double);hw=torch.tensor([[400.,500.]],dtype=torch.double)
        inp={k:v.detach().clone() for k,v in fi.items()}
        inp['executed_delta']=(g.candidates(fixed,hw,g.action_lattice())[:,:,2:]/fixed[:,None,2:]).log().float()
        return inp,fixed,hw
    a,b=FakeSource(),FakeSource()
    zero=OutputOnlyERC(a,DenseSpatialCurveERC(),provider)
    nonzero=OutputOnlyERC(b,ForcedActionFixture(),provider)
    zero.initialize(None,{});nonzero.initialize(None,{})
    for i in range(24):
        x,y=zero.track(None),nonzero.track(None)
        assert equal(a.state,b.state) and torch.equal(a.prompt,b.prompt) and equal(a.out_dict,b.out_dict)
        assert x['target_bbox']==zero.last_rqc_box and x['target_bbox']!=a.state
        assert x['target_bbox']!=y['target_bbox']
        xx,yy=torch.tensor(x['target_bbox']),torch.tensor(y['target_bbox'])
        torch.testing.assert_close(xx[:2]+xx[2:]/2,yy[:2]+yy[2:]/2,rtol=0,atol=1e-5)
        x['metadata']['shared'].append('caller')
        assert len(a.out_dict['shared'])==1
    assert a.prompt.eq(4).all()
    return dict(synthetic_frames=24,prompt_boundaries=4,RQC_not_native=True,
        width_height_only=True,history_isolated=True,output_deepcopy=True,forced_fixture_not_gain_proof=True)

def check_checkpoint():
    model=DenseSpatialCurveERC();before=copy.deepcopy(model.state_dict());identity=checkpoint_identity()
    good=dict(kind=KIND,identity=identity,epoch=1,model=copy.deepcopy(before))
    payloads=[None,{},dict(good,kind='old_ERC'),dict(good,diagnostic_only=True),dict(good,epoch=0),
              dict(good,epoch=1.5),dict(good,model={}),dict(good,model={'actions':None})]
    for key in ('source_sha256','split_sha256','model_class'):
        p=copy.deepcopy(good);p['identity']['semantics'][key]='wrong';payloads.append(p)
    for name,mutation in (
        ('actions',lambda t:t.flip(0)),
        ('curve.1.weight',lambda t:t.double()),
        ('curve.3.bias',lambda t:torch.full_like(t,float('nan'))),
        ('curve.1.weight',lambda t:t[:1])):
        p=copy.deepcopy(good);p['model'][name]=mutation(p['model'][name]);payloads.append(p)
    for p in payloads:
        reject(lambda:validate_payload(p,model,identity))
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    validate_payload(good,model,identity)
    return dict(rejections=len(payloads),no_partial_mutation=True,in_memory_fixture_only=True,checkpoint_files_loaded=0)

def check_provider():
    args=source_args(1);att=dict(source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,epoch=30,variant='directional_center_guard_joint')
    provider=RQC30Provider(att)
    previous=[40.,50.,100.,80.];fixed=[42.,50.,100.,80.]
    side=math.ceil(math.sqrt(100*80)*4)
    origin=torch.round(torch.tensor(previous[:2])+torch.tensor(previous[2:])/2-side/2)[None].double()
    special=dict(args,fixed_xywh=torch.tensor([fixed],dtype=torch.double),
        origin=origin,side=torch.tensor([side],dtype=torch.double),image_hw=torch.tensor([[400.,500.]],dtype=torch.double))
    old=build_inputs(**special)
    source=SimpleNamespace(flag=torch.tensor([2]),params=SimpleNamespace(search_factor=4.),
        observed=dict(search=args['search'],bbox_map=args['bbox_map']),
        initial_template=args['template'],initial_foreground=args['foreground'],
        _capture=SimpleNamespace(maps={'conv_bbox':args['bbox_features']}),
        observed_rgb=args['search_rgb'],template=args['template_rgb'],
        last_inputs={k:old[k] for k in ('valid','executed_delta','source_context')})
    result,_,_=provider(source,np.zeros((400,500,3),dtype=np.uint8),dict(target_bbox=fixed),previous)
    for key in result:assert torch.equal(result[key],old[key]),key
    source.flag=torch.tensor([1])
    reject(lambda:provider(source,np.zeros((400,500,3)),dict(target_bbox=fixed),previous))
    reject(lambda:RQC30Provider(dict(att,source_sha256='wrong')))
    return dict(captured_tensors_only=True,original_support_and_context_bound=True,VL_flag2_only=True,real_source_loaded=False)

def main():
    torch.set_num_threads(1);torch.manual_seed(2026102091)
    assert not torch.cuda.is_initialized()
    before=list(sys.path)
    checks={}
    for name,fn in (('parity',check_parity),('ROI',check_roi),('derivatives',check_derivatives),('model',check_model),('loss',check_loss),
                    ('runtime',check_runtime),('checkpoint',check_checkpoint),('provider',check_provider)):
        checks[name]=fn()
        print('CPU_CHECK_PASS='+name,flush=True)
    assert before==sys.path and not torch.cuda.is_initialized()
    report=dict(CPU_suite_pass=True,semantics=semantics(),checks=checks,
        optimizer_steps=0,GPU_operations=0,real_image_forwards=0,dataset_reads=0,
        trained_checkpoint_files_loaded=0,real_engineering_verified=False,empirical_gain_proven=False,
        training_started=False,test_started=False)
    print('DENSE_CURVE_CPU_REPORT='+json.dumps(report,allow_nan=False),flush=True)

if __name__=='__main__':main()
