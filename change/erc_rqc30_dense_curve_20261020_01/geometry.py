"""Draft Eq.(5),(9): actual size execution and unchanged integer serialization."""
import torch

ACTION_UNIT = .05
GAIN_UNIT = .01


def action_lattice():
    levels=torch.tensor([0.,-.025,.025,-.05,.05])
    actions=levels[torch.cartesian_prod(torch.arange(5),torch.arange(5))]
    return actions[actions.square().sum(1).argsort(stable=True)]


def apply_size(fixed_xywh, delta_logwh, image_hw):
    fixed,hw,delta=fixed_xywh.detach().double(),image_hw.detach().double(),delta_logwh.double()
    if fixed.shape!=(len(fixed),4) or hw.shape!=(len(fixed),2) or delta.shape!=(len(fixed),2):
        raise ValueError('Wrong execution schema')
    if not all(torch.isfinite(v).all() for v in (fixed,hw,delta)) or not fixed[:,2:].gt(0).all() or not hw.gt(0).all():
        raise ValueError('Invalid executed geometry')
    if delta.numel() and delta.detach().abs().max()>ACTION_UNIT+1e-6:
        raise ValueError('Unsupported action magnitude')
    extent=hw.flip(-1);center=fixed[:,:2]+fixed[:,2:]/2
    numerical=4*torch.finfo(torch.float32).eps*extent.clamp_min(1)
    if (fixed[:,:2]<-numerical).any() or (fixed[:,:2]+fixed[:,2:]>extent+numerical).any():
        raise ValueError('Fixed output outside source numerical envelope')
    available=2*torch.minimum(center,extent-center)
    minimum=torch.minimum(fixed[:,2:],torch.full_like(available,10))
    requested=fixed[:,2:]*delta.exp()
    resized=torch.minimum(torch.maximum(requested,minimum),torch.maximum(available,fixed[:,2:]))
    difference=resized-fixed[:,2:]
    return torch.cat((fixed[:,:2]-.5*difference,fixed[:,2:]+difference),-1)


def overlap(boxes,target):
    size=(torch.minimum(boxes[:,:2]+boxes[:,2:],target[:,:2]+target[:,2:])-
          torch.maximum(boxes[:,:2],target[:,:2])).clamp_min(0)
    inter=size.prod(1)
    return inter/(boxes[:,2:].prod(1)+target[:,2:].prod(1)-inter).clamp_min(1e-10)


@torch.no_grad()
def candidates(fixed,image_hw,actions):
    b,k=len(fixed),len(actions)
    return apply_size(fixed[:,None].expand(-1,k,-1).reshape(-1,4),
        actions[None].expand(b,-1,-1).reshape(-1,2),
        image_hw[:,None].expand(-1,k,-1).reshape(-1,2)).reshape(b,k,4)


@torch.no_grad()
def action_targets(fixed,truth,image_hw,actions):
    b,g=fixed.detach().double(),truth.detach().double()
    if not torch.isfinite(g).all() or not g[:,2:].gt(0).all():
        raise ValueError('Targets require finite positive GT')
    boxes=candidates(b,image_hw,actions);n,k=boxes.shape[:2]
    target=g[:,None].expand(-1,k,-1).reshape(-1,4)
    gf=overlap(boxes.reshape(-1,4),target).reshape(n,k)-overlap(b,g)[:,None]
    gz=overlap(boxes.long().double().reshape(-1,4),target).reshape(n,k)-overlap(b.long().double(),g)[:,None]
    gain=torch.minimum(gf,gz)
    assert gain[:,0].eq(0).all()
    return dict(boxes=boxes,float_gain=gf,integer_gain=gz,gain=gain,utility=gain/GAIN_UNIT,
        executed_delta=(boxes[:,:,2:]/b[:,None,2:]).log())
