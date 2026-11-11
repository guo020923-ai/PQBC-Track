"""Ordered dense ROI -> direction and unchanged balanced PSD quadratic utility."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import action_lattice,ACTION_UNIT

MODEL_CLASS='DenseSpatialCurveERC'
MODEL_PARAMETERS=120257
ROI_CHANNELS=951
SUMMARY_DIM=1152

def exact_gelu(x):
    """Exact GELU algebra; avoids observed torch2 CPU strided-gradient kernel error."""
    return .5*x*(1+torch.erf(x/math.sqrt(2)))

class ExactGELU(nn.Module):
    def forward(self,x):return exact_gelu(x)

# Constants derived only from the unchanged initial L diagonal (.1), not from
# TRAIN/monitor/test statistics or a hyperparameter search.
READOUT_REFERENCE_LOGIT=float(torch.tensor(math.log(math.expm1(.1)),dtype=torch.float32))
READOUT_REFERENCE_L=math.log1p(math.exp(READOUT_REFERENCE_LOGIT))
READOUT_SIGMOID=1/(1+math.exp(-READOUT_REFERENCE_LOGIT))
READOUT_DIAGONAL_SCALE=1/(2*READOUT_REFERENCE_L*READOUT_SIGMOID)
READOUT_CROSS_SCALE=1/(math.sqrt(2)*READOUT_REFERENCE_L)


def decode_coefficients(raw):
    """Same PSD curve family, unit local coefficient Jacobian at zero raw."""
    if raw.ndim!=2 or raw.shape[1]!=5 or not torch.isfinite(raw).all():
        raise ValueError('Expected finite Bx5 readout')
    direction=raw[:,:2].double()
    l11=F.softplus(READOUT_REFERENCE_LOGIT+READOUT_DIAGONAL_SCALE*raw[:,2].double())
    l21=READOUT_CROSS_SCALE*raw[:,3].double()
    l22=F.softplus(READOUT_REFERENCE_LOGIT+READOUT_DIAGONAL_SCALE*raw[:,4].double())
    zeros=torch.zeros_like(l11)
    lower=torch.stack((torch.stack((l11,zeros),1),torch.stack((l21,l22),1)),1)
    curvature=lower.bmm(lower.transpose(1,2))
    if not torch.isfinite(curvature).all():raise FloatingPointError('Nonfinite balanced PSD curvature')
    return direction,curvature



def supported_actions(valid,actions):
    count=valid.detach().bool().sum(-1)
    axis_supported=count.ge(3).all(-1).any(-1)
    allowed=(actions[None].eq(0)|axis_supported).all(-1)
    allowed=allowed.clone();allowed[:,0]=True
    return allowed



class DenseSpatialCurveERC(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('actions',action_lattice())
        def projection(n,out):
            return nn.Sequential(nn.LayerNorm(n),nn.Linear(n,out),ExactGELU())
        self.search=projection(768,24)
        self.association=projection(128,16)
        self.head=projection(32,8)
        self.pixels=projection(18,8)
        self.fuse=nn.Conv2d(61,32,1)
        self.spatial=nn.ModuleList([nn.Conv2d(32,32,3,padding=1),nn.Conv2d(32,32,3,padding=1)])
        self.reduce=nn.Conv2d(32,16,1)
        self.curve=nn.Sequential(nn.LayerNorm(SUMMARY_DIM),nn.Linear(SUMMARY_DIM,64),ExactGELU(),nn.Linear(64,5))
        nn.init.zeros_(self.curve[-1].weight)
        nn.init.zeros_(self.curve[-1].bias)
        assert sum(p.numel() for p in self.parameters())==MODEL_PARAMETERS

    def encode(self,roi_evidence,roi_valid,valid,source_context,executed_delta):
        b=len(roi_evidence)
        if roi_evidence.shape!=(b,951,16,16) or roi_valid.shape!=(b,1,16,16) or roi_valid.dtype!=torch.bool:
            raise ValueError('Expected ROI Bx951x16x16 and boolean Bx1x16x16')
        if valid.shape!=(b,25,2,2,2,5) or valid.dtype!=torch.bool:
            raise ValueError('Original action support schema required')
        if source_context.shape!=(b,64) or executed_delta.shape!=(b,25,2):
            raise ValueError('Wrong context/actual geometry schema')
        mask=roi_valid.detach()
        raw=torch.where(mask,roi_evidence.detach().float(),torch.zeros_like(roi_evidence,dtype=torch.float))
        context=source_context.detach().float()
        x=executed_delta.detach().double()/ACTION_UNIT
        if not all(torch.isfinite(v).all() for v in (raw,context,x)):
            raise ValueError('Nonfinite valid evidence/context/geometry')
        if not x[:,0].eq(0).all() or (x.numel() and x.abs().max()>1+1e-5):
            raise ValueError('Expected actual logwh in .05 units with zero KEEP')
        if not torch.equal(raw[:,950:951],mask.float()):
            raise ValueError('Evidence validity channel differs from ROI mask')
        groups=[]
        start=0
        for n,project in ((768,self.search),(128,self.association),(32,self.head),(18,self.pixels)):
            group=raw[:,start:start+n].permute(0,2,3,1)
            # Mask both before LN (above) and after biased projection.
            groups.append(project(group).permute(0,3,1,2)*mask)
            start+=n
        z=torch.cat((*groups,raw[:,946:951]),1)
        z=exact_gelu(self.fuse(z))*mask
        for layer in self.spatial:z=exact_gelu(layer(z))*mask
        z=self.reduce(z)*mask
        # Sum/count, not avg/denominator-clamped-to-one: partial cells retain scale.
        count=F.avg_pool2d(mask.float(),2)*4
        total=F.avg_pool2d(z,2)*4
        pooled=total/count.clamp_min(1)
        coverage=count/4
        summary=torch.cat((pooled.flatten(1),coverage.flatten(1),context),1)
        if summary.shape!=(b,SUMMARY_DIM) or not torch.isfinite(summary).all():
            raise ValueError('Invalid ordered spatial summary')
        return summary,supported_actions(valid,self.actions),x,mask.flatten(1).any(1)

    def decision(self,**inputs):
        summary,allowed,x,observed=self.encode(**inputs)
        raw=self.curve(summary)
        raw=torch.where(observed[:,None],raw,torch.zeros_like(raw))
        direction,curvature=decode_coefficients(raw)
        utility=(x*direction[:,None]).sum(-1)-torch.einsum('bki,bij,bkj->bk',x,curvature,x)
        if not torch.isfinite(utility).all():raise FloatingPointError('Nonfinite quadratic utility')
        assert utility[:,0].eq(0).all()
        index=utility.masked_fill(~allowed,-torch.inf).argmax(1)
        return dict(utility=utility,allowed=allowed,index=index,delta=self.actions[index],
            direction=direction,curvature=curvature)

    def forward(self,**inputs):return self.decision(**inputs)
