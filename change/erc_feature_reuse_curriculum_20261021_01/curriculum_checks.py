"""Additional synthetic checks, zero optimizer and zero real data."""
import copy
import torch
from .inputs import build_inputs
from .geometry import apply_size,candidates,overlap
from .objective import objective
from .curriculum import OFFSETS,anchor_view,training_loss


def run_curriculum_checks():
    from .checks import fixture,fail
    model,inputs,fixed,truth,hw=fixture()
    bbox=torch.zeros(2,256,4);bbox[...,2:]=.2
    origins=[[-16.,-24.],[-80.,-90.]]
    rows=[dict(native_bbox_map=bbox[i:i+1],
        meta=dict(crop_origin=origins[i],side=256)) for i in range(2)]
    snapshot={k:v.clone() for k,v in inputs.items()}
    for offset in OFFSETS:
        view,anchor=anchor_view(inputs,fixed,hw,rows,offset)
        exact=build_inputs(inputs['search'].flatten(2).transpose(1,2),inputs['head'],
            inputs['template_roles'],bbox,apply_size(fixed,fixed.new_tensor(offset).expand(2,-1),hw),
            fixed.new_tensor(origins),fixed.new_tensor([256.,256.]),hw,inputs['source_context'])
        assert all(torch.equal(view[k],exact[k]) for k in view)
        assert torch.equal(view['candidate_boxes'],candidates(anchor,hw,model.actions))
        pred=model(**view)
        assert pred['index'].eq(0).all() and pred['utility'].eq(0).all()
        _,_,picked=objective(model,pred,view,anchor,truth,hw)
        assert torch.equal(picked,anchor)
    assert all(torch.equal(inputs[k],snapshot[k]) for k in inputs)
    fail(lambda:anchor_view(inputs,fixed,hw,rows,(.1,0.)))
    with torch.no_grad():model.score[-1].weight.normal_(0,.01)
    pred=model(**inputs);natural,stats,_=objective(model,pred,inputs,fixed,truth,hw)
    combined,detail=training_loss(model,pred,inputs,fixed,truth,hw,rows,1,natural)
    independent=[natural]
    for offset in OFFSETS:
        vi,vf=anchor_view(inputs,fixed,hw,rows,offset)
        independent.append(objective(model,model(**vi),vi,vf,truth,hw)[0])
    torch.testing.assert_close(combined,torch.stack(independent).mean(),rtol=1e-12,atol=1e-12)
    assert detail['anchor_views']==5 and detail['augmented_views']==4
    combined.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(v.grad is None for v in inputs.values())
    calibrated,second=training_loss(model,pred,inputs,fixed,truth,hw,rows,41,natural)
    assert calibrated is natural and second['augmented_views']==0
    assert training_loss(model,pred,inputs,fixed,truth,hw,rows,60,natural)[0] is natural
    _,again,_=objective(model,model(**inputs),inputs,fixed,truth,hw)
    assert again==stats,'Augmented statistics must never replace natural statistics'
    # Sign/direction exercise with an exact rectangular target: symmetric anchor
    # errors have opposite beneficial corrections, with real integer execution.
    target=torch.tensor([[80.,80.,80.,80.]],dtype=torch.double)
    image=torch.tensor([[256.,256.]],dtype=torch.double)
    for offset in ((-.05,0.),(.05,0.),(0.,-.05),(0.,.05)):
        anchor=apply_size(target,target.new_tensor([offset]),image)
        corrected=apply_size(anchor,target.new_tensor([[-offset[0],-offset[1]]]),image)
        assert overlap(corrected,target).item()>overlap(anchor,target).item()
        assert overlap(corrected.long().double(),target).item()>overlap(anchor.long().double(),target).item()
    return dict(curriculum_actual_rebuilt_geometry=True,curriculum_original_inputs_unchanged=True,
        curriculum_equal_five_view_loss_and_finite_gradients=True,
        curriculum_epochs41_60_exact_natural_loss=True,natural_statistics_isolated=True,
        synthetic_bidirectional_correction_only_not_efficacy=True)
