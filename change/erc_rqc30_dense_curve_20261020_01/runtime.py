"""Output-only execution. Zero ERC == RQC output, never necessarily native box."""
import copy
import numpy as np
import torch
from .geometry import apply_size


def clone(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
    if isinstance(value,list):return [clone(v) for v in value]
    if isinstance(value,tuple):return tuple(clone(v) for v in value)
    return copy.deepcopy(value)


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def history(source):
    return {k:clone(getattr(source,k)) for k in ('state','last_native_box','prompt','frame_id',
        'max_score','pred_box_net','out_dict') if hasattr(source,k)}


class OutputOnlyERC:
    def __init__(self,source,inference_model,provider):
        self.source,self.model,self.provider=source,inference_model.eval(),provider
        for p in self.model.parameters():p.requires_grad_(False)
        self.last_native_box=self.last_rqc_box=self.last_output_box=None

    @torch.no_grad()
    def initialize(self,image,info):
        self.source.initialize(image,info)
        self.last_native_box=self.last_rqc_box=self.last_output_box=None

    @torch.no_grad()
    def track(self,image,info=None):
        previous=clone(self.source.state)
        fixed_output=clone(self.source.track(image,info))
        before=history(self.source)
        inputs,fixed,hw=self.provider(self.source,image,fixed_output,previous)
        pred=self.model(**inputs)
        result=clone(fixed_output)
        if pred['index'].ne(0).any():result['target_bbox']=apply_size(fixed,pred['delta'],hw)[0].cpu().tolist()
        if pred['index'].eq(0).all():assert equal(result,fixed_output)
        assert equal(before,history(self.source)),'ERC/evidence provider wrote native history'
        self.last_native_box=clone(self.source.state)
        self.last_rqc_box=clone(fixed_output['target_bbox'])
        self.last_output_box=clone(result['target_bbox'])
        self.last_inputs,self.last_prediction=inputs,pred
        return result
