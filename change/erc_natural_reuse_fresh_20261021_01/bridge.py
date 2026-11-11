"""Direct frozen RQC adapter; bypass old ERC evidence and its zero MLP entirely."""
import math
import random
import importlib
import torch
from .inputs import build_inputs,template_roles,INPUT_KEYS
from .checkpoint import SOURCE_SHA,TRAIN_SHA
from .runtime import OutputOnlyERC
from .native_tap import NativeContextTap


def load_legacy():
    return importlib.import_module('change.erc_rqc30_originaltrain_formal60_20261020_01.rqc30_train_contract')


def base_class():
    legacy=load_legacy()
    return legacy.infrastructure.nc.previous.base_runtime.JointCenterTracker


@torch.no_grad()
def captured_fields(source,out):
    return out['search'],source._capture.maps['conv_bbox'],out['bbox_map'],source._reused_context



def collector_class():
    Base=base_class()
    class LightCollector(Base):
        def __init__(self,params,domain,enabled=True):
            super().__init__(params,domain)
            self.enabled=bool(enabled)
            self._native_context_tap=NativeContextTap(self.refiner)
            original=self.network.forward_test
            def observe(*args,**kwargs):
                out=original(*args,**kwargs)
                self.observed=out
                return out
            self.network.forward_test=observe

        @torch.no_grad()
        def initialize(self,image,info):
            self.max_score=0
            for key in ('pred_box_net','out_dict','observed','last_inputs','last_decision',
                        'last_native_box','_light_stamp','_initial_roles'):
                self.__dict__.pop(key,None)
            self._center_prediction=None
            captured=[]
            def reuse_initial(_module,_args,result):
                # Observe the EXISTING forward_prompt_init Backbone call.
                # Clone only; never mutate its dict, tensor, prompt or graph.
                captured.append(result['template'].detach().clone())
            handle=self.network.backbone.register_forward_hook(reuse_initial)
            try:super().initialize(image,info)
            finally:handle.remove()
            if len(captured)!=1:
                raise ValueError('Expected exactly one existing initialization Backbone call')
            self._initial_roles=template_roles(captured[0],self.template_mask.detach().bool())
            self.initialization_reused_Backbone_calls=len(captured)

        @torch.no_grad()
        def track(self,image,info=None):
            previous=list(self.state)
            self._native_context_tap.begin()
            try:
                returned=super().track(image,None)
                self._reused_context=self._native_context_tap.finish()
            finally:self._native_context_tap.cancel()
            if not self.enabled:return returned
            search,head,bbox,context=captured_fields(self,self.observed)
            device=search.device
            fixed=torch.tensor([returned['target_bbox']],device=device,dtype=torch.double)
            hw=torch.tensor([image.shape[:2]],device=device,dtype=torch.double)
            p=torch.tensor(previous,device=device,dtype=torch.double)
            side=math.ceil(math.sqrt(previous[2]*previous[3])*self.params.search_factor)
            origin=torch.round(p[:2]+p[2:]/2-side/2)[None]
            self.last_inputs=build_inputs(search,head,self._initial_roles,bbox,fixed,origin,
                torch.tensor([side],device=device,dtype=torch.double),hw,context)
            # Stream collects FIXED RQC trajectories; no ERC model is run here.
            self.last_decision=dict(index=torch.zeros(1,device=device,dtype=torch.long))
            self._light_stamp=(self.frame_id,tuple(returned['target_bbox']),tuple(image.shape[:2]))
            return returned
        def close(self):
            self._native_context_tap.close()
            super().close()
    return LightCollector


def stream_class():
    legacy=load_legacy();Collector=collector_class()
    class LightStream(legacy.Stream):
        def __init__(self,pools,source,params,rng,spec,forced_clips=None):
            self.pools,self.source,self.rng=pools,source,rng
            self.spec=spec['NATURAL_INPUT']
            self.forced_clips=iter(forced_clips) if forced_clips is not None else None
            self.collector=Collector(params,pools.datasets[source].get_name())
            self.clip=self.cursor=None
            self.frames=self.clips=0;self.seconds=0.
    return LightStream


@torch.no_grad()
def provider(source,image,fixed_output):
    if source._light_stamp!=(source.frame_id,tuple(fixed_output['target_bbox']),tuple(image.shape[:2])):
        raise ValueError('Stale/RQC-mismatched feature evidence')
    if set(source.last_inputs)!=INPUT_KEYS:raise ValueError('Wrong lightweight input keys')
    device=source.last_inputs['search'].device
    return source.last_inputs,torch.tensor([fixed_output['target_bbox']],device=device,dtype=torch.double),torch.tensor([image.shape[:2]],device=device,dtype=torch.double)


def create_runtime(params,domain,strictly_loaded_model):
    return OutputOnlyERC(collector_class()(params,domain),strictly_loaded_model,provider)


def original_training_components():
    legacy=load_legacy()
    if legacy.digest(legacy.SOURCE)!=SOURCE_SHA or legacy.digest(legacy.SPLIT)!=TRAIN_SHA:
        raise ValueError('Frozen source/original TRAIN changed')
    legacy.verify_split();legacy.validate_source()
    return legacy,stream_class()


def source_snapshot(source):
    return ({k:v.detach().cpu().clone() for k,v in source.network.box_head.state_dict().items()},
            {k:v.detach().cpu().clone() for k,v in source.refiner.state_dict().items()},
            {k:v.detach().cpu().clone() for k,v in source.network.named_buffers()})


def assert_unchanged(source,snapshot):
    head,refiner,buffers=snapshot
    for key,value in source.network.box_head.state_dict().items():assert torch.equal(value.cpu(),head[key]),key
    for key,value in source.refiner.state_dict().items():assert torch.equal(value.cpu(),refiner[key]),key
    for key,value in source.network.named_buffers():assert torch.equal(value.cpu(),buffers[key]),key
    for model in (source.network,source.refiner):
        assert all(not m.training for m in model.modules())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
