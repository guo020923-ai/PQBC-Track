"""Shared mean-gain risk for fit and monitor; unchanged actual execution."""
import torch
from torch.nn import functional as F
from .geometry import action_targets,apply_size
from .model import supported_actions


from .gain_risk_objective import loss_from_targets


def objective(model,prediction,inputs,fixed_rqc_box,truth,image_hw):
    """GT enters here only, after causal source and ERC forward."""
    expected=supported_actions(inputs['valid'],model.actions)
    if not torch.equal(expected,prediction['allowed']):raise ValueError('Support/statistics mismatch')
    boxes=apply_size(fixed_rqc_box,prediction['delta'],image_hw)
    valid=torch.isfinite(truth).all(1)&truth[:,2:].gt(0).all(1)
    if not valid.any():
        return prediction['utility'].sum()*0,dict(valid_samples=0,loss=0.),boxes
    target=action_targets(fixed_rqc_box[valid],truth[valid],image_hw[valid],model.actions)
    if not torch.allclose(inputs['executed_delta'][valid].detach().double(),target['executed_delta'],rtol=1e-6,atol=1e-7):
        raise ValueError('Nominal action used where actual executed delta is required')
    subset={k:v[valid] for k,v in prediction.items()}
    loss,stats=loss_from_targets(subset,inputs['executed_delta'][valid],target['utility'])
    row=torch.arange(int(valid.sum()),device=boxes.device);index=subset['index']
    assert torch.equal(boxes[valid],target['boxes'][row,index])
    gf,gz,gain=(target[key][row,index] for key in ('float_gain','integer_gain','gain'))
    predicted_gain=subset['utility'][row,index].detach()*.01
    # Observability only. No extra model forward, no influence on action/loss.
    with torch.no_grad():
        d=subset['direction'];q=subset['curvature']
        supported=subset['utility'][subset['allowed']]
        stats.update(direction_norm=float(d.norm(dim=1).mean()),
            curvature_trace=float(q.diagonal(dim1=1,dim2=2).sum(1).mean()),
            curvature_smallest_eigenvalue_mean=float(torch.linalg.eigvalsh(q)[:,0].mean()),
            supported_positive_utility_rate=float(supported.gt(0).double().mean()))
    stats.update(valid_samples=int(valid.sum()),float_IoU_gain=float(gf.mean()),integer_IoU_gain=float(gz.mean()),
        selected_predicted_gain=float(predicted_gain.mean()),
        selected_gain_calibration_MSE=float((predicted_gain-gain).square().mean()),
        conservative_gain=float(gain.mean()),KEEP_rate=float(index.eq(0).double().mean()),
        changed_saved_boxes=int(boxes[valid].long().ne(fixed_rqc_box[valid].long()).any(1).sum()),
        integer_improved_frames=int(gz.gt(0).sum()),integer_harmed_frames=int(gz.lt(0).sum()),
        oracle_positive_frames=int(target['gain'].masked_fill(~subset['allowed'],-torch.inf).max(1).values.gt(0).sum()))
    return loss,stats,boxes
