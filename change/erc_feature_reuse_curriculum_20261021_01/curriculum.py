"""TRAIN-only anchor augmentation. No source forward/state/GT in input building.

The original natural RQC view is NEVER reconstructed or relabelled as augmented.
Each changed anchor uses exact existing image/crop execution and source maps.
All labels (positive and negative) retain the original fixed24 proper objective.
"""
import torch
from .inputs import build_inputs
from .geometry import apply_size
from .objective import objective

OFFSETS=((-0.05,0.),(0.05,0.),(0.,-0.05),(0.,0.05))


@torch.no_grad()
def anchor_view(inputs,fixed,hw,rows,offset):
    if tuple(offset) not in OFFSETS:
        raise ValueError('Only the four predeclared axis offsets are authorized')
    if len(rows)!=len(fixed) or not rows:
        raise ValueError('Every natural observation needs its original crop metadata')
    device=fixed.device
    origin=torch.tensor([r['meta']['crop_origin'] for r in rows],device=device,dtype=torch.double)
    side=torch.tensor([r['meta']['side'] for r in rows],device=device,dtype=torch.double)
    native=torch.cat([r['native_bbox_map'].detach() for r in rows])
    delta=fixed.new_tensor(offset).expand(len(fixed),-1)
    changed=apply_size(fixed,delta,hw)
    torch.testing.assert_close(changed[:,:2]+changed[:,2:]/2,
                               fixed[:,:2]+fixed[:,2:]/2,rtol=0,atol=1e-9)
    # Keep the original frozen search/Head/reference/context. Recompute extents,
    # candidates, actual integer writer geometry and support for the NEW anchor.
    result=build_inputs(inputs['search'].flatten(2).transpose(1,2),inputs['head'],
        inputs['template_roles'],native,changed,origin,side,hw,inputs['source_context'])
    for key in ('search','head','template_roles','source_mask','source_context'):
        if not torch.equal(result[key],inputs[key]):
            raise ValueError('Augmentation changed frozen evidence: '+key)
    return result,changed


def training_loss(model,natural_pred,inputs,fixed,truth,hw,rows,epoch,natural_loss):
    if type(epoch) is not int or not 1<=epoch<=60:
        raise ValueError('Fixed 60-epoch curriculum only')
    if epoch>40:
        return natural_loss,dict(anchor_views=1,augmented_views=0,
            total_loss_pp2=float(natural_loss.detach()),augmented_gain_MSE_pp2=0.)
    losses=[]
    for offset in OFFSETS:
        view,anchor=anchor_view(inputs,fixed,hw,rows,offset)
        pred=model(**view)
        loss,_,_=objective(model,pred,view,anchor,truth,hw)
        losses.append(loss)
    loss=(natural_loss+torch.stack(losses).sum())/5
    return loss,dict(anchor_views=5,augmented_views=4,total_loss_pp2=float(loss.detach()),
        augmented_gain_MSE_pp2=float(torch.stack(losses).mean().detach()))
