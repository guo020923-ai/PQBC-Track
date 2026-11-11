"""Direct conditional net-gain regression; labels enter only AFTER prediction.

No best-action classification or positive-only/oracle reweighting. KEEP is exact
zero; all 24 actual candidate-minus-KEEP conservative labels are supervised.
"""
import torch
from .geometry import candidates, overlap, supported_actions
from .model import GAIN_UNIT

COUNTS = {'valid_samples','changed_saved_boxes','integer_improved_frames',
          'integer_harmed_frames','oracle_positive_frames','changed_integer_gain_sum',
          'integer_gain_positive_sum','integer_gain_negative_sum'}



def gain_rows(utility, target_gain, allowed):
    """Fixed 24-action proper regression; KEEP has no artificial target margin."""
    difference=(utility.double()-target_gain.detach().double())/GAIN_UNIT
    return (difference[:,1:].square()*allowed[:,1:].detach()).sum(1)/24


@torch.no_grad()
def summarize(samples, selected=None):
    """Reuse per-row sufficient statistics; no model/loss/geometry recomputation.

    A single packed device-to-host copy replaces many scalar synchronizations.
    Every domain reuses the same per-observation gain statistics.
    """
    first=next(iter(samples.values()))
    mask=torch.ones_like(first,dtype=torch.bool) if selected is None else selected.bool()
    n=mask.double().sum()
    keys=list(samples)
    values=[]
    for key in keys:
        numerator=(samples[key].double()*mask).sum()
        denom=n
        values.append(numerator if key in COUNTS else numerator/denom.clamp_min(1))
    packed=torch.stack(values).detach().cpu().tolist()
    result=dict(zip(keys,packed))
    result['loss']=result['gain_MSE_pp2']
    return result


def objective(model, pred, inputs, fixed, truth, hw, return_samples=False):
    truth=truth.detach().double()
    if truth.shape!=fixed.shape or not torch.isfinite(truth).all() or not truth[:,2:].gt(0).all():
        raise ValueError('Training expects finite positive GT; invalid labels are not replaced')
    boxes=candidates(fixed,hw,model.actions)
    torch.testing.assert_close(boxes,inputs['candidate_boxes'].double(),rtol=0,atol=0)
    allowed=supported_actions(inputs['valid'],model.actions)
    if not torch.equal(allowed,pred['allowed']):
        raise ValueError('Loss and decision support differ')
    b,k=boxes.shape[:2]
    target=truth[:,None].expand(-1,k,-1).reshape(-1,4)
    with torch.no_grad():
        fi=overlap(boxes.reshape(-1,4),target).reshape(b,k)
        zi=overlap(boxes.long().double().reshape(-1,4),target).reshape(b,k)
        gf,gz=fi-fi[:,:1],zi-zi[:,:1]
        gain=torch.minimum(gf,gz)
    assert k==25 and gain[:,0].eq(0).all()
    # Target and prediction are in percentage points for numerical conditioning.
    # Expected squared loss minimizes E[conservative gain | available inputs];
    # neither the existence of an oracle nor TRAIN loss proves learnability.
    row_gain_loss=gain_rows(pred['utility'],gain,allowed)
    gain_loss=row_gain_loss.mean()
    # The removed RGB boundary decoder has no auxiliary GT box head here.
    loss=gain_loss
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite lightweight gain objective')
    with torch.no_grad():
        row,index=torch.arange(b,device=boxes.device),pred['index']
        picked=boxes[row,index]
        cf=gain[row,index];ig=gz[row,index];fg=gf[row,index]
        changed=picked.long().ne(boxes[:,0].long()).any(-1)
        oracle=gain.masked_fill(~allowed,-torch.inf).max(1).values
        samples=dict(valid_samples=torch.ones_like(cf),gain_MSE_pp2=row_gain_loss.detach(),
            KEEP_rate=index.eq(0).double(),
            float_IoU_gain=fg,integer_IoU_gain=ig,conservative_gain=cf,
            changed_saved_boxes=changed.double(),integer_improved_frames=ig.gt(0).double(),
            integer_harmed_frames=ig.lt(0).double(),oracle_positive_frames=oracle.gt(0).double(),
            oracle_conservative_gain=oracle,decision_regret=oracle-cf,
            selected_predicted_gain=pred['utility'][row,index],
            changed_integer_gain_sum=ig*changed,integer_gain_positive_sum=ig.clamp_min(0),
            integer_gain_negative_sum=ig.clamp_max(0))
        stats=summarize(samples)
    result=(loss,stats,picked)
    return result+(samples,) if return_samples else result

