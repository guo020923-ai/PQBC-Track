"""Small shared boundary-strip gain scorer, no RGB encoder/spatial fusion CNN."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .inputs import INPUT_KEYS
from .geometry import action_lattice,supported_actions

MODEL_CLASS='FeatureReuseCurriculumERC'
GAIN_UNIT=.01
FEATURES=36
DESCRIPTOR=384  # 40 ordered points x 8 + 40 coverage + 8 geometry + 16 context


def strip_grid(boxes):
    """Same four directed edges, inner/outer support and five tangential points."""
    offsets=boxes.new_zeros(2,2,2,5,2)
    tangent=boxes.new_tensor([-.4,-.2,0.,.2,.4])
    for axis in range(2):
        for edge,sign in enumerate((-1.,1.)):
            for region,radius in enumerate((.5*math.exp(-.05),.5*math.exp(.05))):
                offsets[axis,edge,region,:,axis]=sign*radius
                offsets[axis,edge,region,:,1-axis]=tangent
    points=(boxes[...,:2]+boxes[...,2:]/2)[:,:,None,None,None,None]+offsets[None,None]*boxes[:,:,None,None,None,None,2:]
    return (points*2-1).reshape(len(boxes),25*8,5,2)


class FeatureReuseCurriculumERC(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('actions',action_lattice())
        self.search=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,24),nn.SiLU())
        self.head=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,8),nn.SiLU())
        self.context=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,16),nn.SiLU())
        self.point=nn.Sequential(nn.LayerNorm(37),nn.Linear(37,8),nn.SiLU())
        self.score=nn.Sequential(nn.LayerNorm(DESCRIPTOR),nn.Linear(DESCRIPTOR,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())

    def forward(self,**inputs):
        if set(inputs)!=INPUT_KEYS:
            raise ValueError('Strict reused-feature schema; no RGB/GT/previous ERC output input')
        t={k:v.detach() for k,v in inputs.items()};b=len(t['search'])
        expected=dict(search=(b,768,16,16),head=(b,32,16,16),template_roles=(b,2,768),
            native_extent=(b,2,16,16),source_mask=(b,1,16,16),source_context=(b,64),
            candidate_boxes=(b,25,4),crop_boxes=(b,25,4),saved_crop_boxes=(b,25,4),valid=(b,25,2,2,2,5))
        if any(t[k].shape!=shape for k,shape in expected.items()):
            raise ValueError('Wrong lightweight input shape')
        if t['source_mask'].dtype!=torch.bool or t['valid'].dtype!=torch.bool:
            raise ValueError('Boolean source/action support required')
        mask=t['source_mask']
        for key in ('search','head','native_extent'):
            t[key]=torch.where(mask,t[key].float(),torch.zeros_like(t[key],dtype=torch.float))
        if not all(torch.isfinite(v).all() for v in t.values()):
            raise ValueError('Nonfinite visible input')
        if not t['candidate_boxes'][...,2:].gt(0).all() or not t['crop_boxes'][...,2:].gt(0).all():
            raise ValueError('Positive executed boxes required')
        q=self.search(t['search'].permute(0,2,3,1)).permute(0,3,1,2)*mask
        reference=self.search(t['template_roles'].float())
        sim=torch.bmm(F.normalize(reference,dim=-1),F.normalize(q.flatten(2),dim=1)).reshape(b,2,16,16)*mask
        h=self.head(t['head'].permute(0,2,3,1)).permute(0,3,1,2)*mask
        field=torch.cat((q,h,sim,t['native_extent']),1)*mask
        grid=strip_grid(t['crop_boxes'].float())
        sampled=F.grid_sample(field,grid,align_corners=False,padding_mode='zeros')
        coverage=F.grid_sample(mask.float(),grid,align_corners=False,padding_mode='zeros')
        # Retain every ordered edge/region/tangent point, not a directionless mean.
        points=torch.cat((sampled,coverage),1).permute(0,2,3,1).reshape(b,25,40,37)
        embedded=self.point(points)*coverage.reshape(b,1,25,40).squeeze(1)[:,:,:,None]
        observed=mask.flatten(1).any(1)
        cb=t['crop_boxes'].float();sb=t['saved_crop_boxes'].float()
        geometry=torch.cat((cb[...,:2],cb[...,:2]+cb[...,2:],sb[...,:2],sb[...,:2]+sb[...,2:]),-1)
        context=self.context(t['source_context'].float())[:,None].expand(-1,25,-1)
        descriptor=torch.cat((embedded.flatten(2),coverage.reshape(b,25,40),geometry,context),-1)
        if descriptor.shape!=(b,25,DESCRIPTOR) or not torch.isfinite(descriptor).all():
            raise ValueError('Invalid shared lightweight descriptor')
        score=self.score(descriptor).squeeze(-1)
        score=torch.where(observed[:,None],score,torch.zeros_like(score))
        utility=GAIN_UNIT*(score-score[:,:1])
        allowed=supported_actions(t['valid'],self.actions)
        index=utility.masked_fill(~allowed,-torch.inf).argmax(1)
        return dict(index=index,delta=self.actions[index],utility=utility,allowed=allowed,score=score,observed=observed)

