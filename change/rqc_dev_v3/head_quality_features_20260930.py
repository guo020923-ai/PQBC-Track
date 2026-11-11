"""Passive diagnostic capture of localization-head features, not a new scorer.

No model parameters, no altered forward values, and no deployment registration.
Handles are removed by context exit. Features are detached deliberately.
"""
import torch
from torch.nn import functional as F

class HeadQualityCapture:
    def __init__(self,head):
        self.head=head;self.maps={};self.handles=[]
    def __enter__(self):
        for name in ('conv_bbox','conv_offset','conv_cls'):
            def hook(module,inputs,key=name):
                self.maps[key]=inputs[0].detach()
            self.handles.append(getattr(self.head,name)[-1].register_forward_pre_hook(hook))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.maps.clear()
    def gather(self,indices,active=None):
        # Center/local average/local std preserve localization branch identity.
        features=[]
        for name in ('conv_bbox','conv_offset','conv_cls'):
            x=self.maps[name]
            if active is not None:x=x[active]
            assert x.shape[0]==indices.shape[0] and x.shape[1]==32
            mean=F.avg_pool2d(x,3,stride=1,padding=1,count_include_pad=False)
            variance=(F.avg_pool2d(x.square(),3,stride=1,padding=1,count_include_pad=False)-mean.square()).clamp_min(0)
            for value in (x,mean,variance.sqrt()):
                flat=value.flatten(2).transpose(1,2)
                features.append(flat.gather(1,indices[...,None].expand(-1,-1,32)))
        answer=torch.cat(features,-1)
        assert answer.shape[-1]==288 and torch.isfinite(answer).all()
        return answer
