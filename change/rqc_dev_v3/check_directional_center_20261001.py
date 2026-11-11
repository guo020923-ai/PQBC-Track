"""CPU structural checks only; no optimizer step, no checkpoint generation."""
import json,torch
from directional_center_refiner_20261001 import DirectionalCenterRefiner,DirectionalCenterCapture
from center_safe_refiner_20260930 import center_safe_loss
from local_box_refiner_20260930 import decode_image_boxes

def main():
    torch.set_num_threads(1);torch.manual_seed(20261001)
    model=DirectionalCenterRefiner();context=torch.randn(4,288,requires_grad=True)
    pairs=torch.randn(4,2,2,96,requires_grad=True);boxes=torch.tensor([[.5,.5,.3,.2]]*4,requires_grad=True)
    x=dict(context=context,pairs=pairs);zero=model(x,boxes)
    assert torch.equal(zero['boxes'],boxes) and torch.count_nonzero(zero['delta'])==0
    # A single synthetic backward checks learnability, without updating weights.
    gt=torch.tensor([[.37,.38,.3,.2]]*4)*256
    side=torch.full((4,),256.);origin=torch.zeros(4,2);hw=torch.full((4,2),256.)
    native=decode_image_boxes(boxes.detach(),origin,side,hw)
    refined=decode_image_boxes(zero['boxes'],origin,side,hw)
    loss,_=center_safe_loss(refined,native,gt,hw,torch.ones(4,dtype=torch.bool));loss.backward()
    assert torch.isfinite(loss) and model.net[-1].weight.grad.abs().sum()>0
    assert context.grad is pairs.grad is boxes.grad is None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Synthetic nonzero weights, NOT saved or usable for training initialization.
    with torch.no_grad():model.net[-1].weight.normal_(std=.2)
    a=model(x,boxes);swapped=dict(context=context,pairs=pairs.flip(-2))
    b=model(swapped,boxes);torch.testing.assert_close(a['delta'],-b['delta'],rtol=0,atol=0)
    equal=dict(context=context,pairs=pairs[...,0:1,:].expand_as(pairs))
    assert torch.count_nonzero(model(equal,boxes)['delta'])==0
    assert torch.equal(a['boxes'][...,2:],boxes[...,2:]) and a['delta'].abs().max()<=.1
    # Actual spatial sampler on a synthetic x/y coordinate ramp, not assumed ordering.
    capture=DirectionalCenterCapture(None);grid=(torch.arange(16)+.5)/16
    ramp=torch.stack(torch.meshgrid(grid,grid,indexing='ij'),0)
    capture.maps={name:ramp.repeat(16,1,1)[None] for name in ('conv_bbox','conv_offset','conv_cls')}
    sampled=capture.gather_directional(torch.tensor([[136]]),boxes[:1,None].detach())['pairs']
    assert sampled[0,0,0,0,1]>sampled[0,0,0,1,1]  # +x versus -x, x channel
    assert sampled[0,0,1,0,0]>sampled[0,0,1,1,0]  # +y versus -y, y channel
    print('DIRECTIONAL_CENTER_CHECK_PASS',json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
        zero_native_exact=True,opposite_side_swap_negates_delta=True,equal_sides_zero=True,
        spatial_order_verified=True,backbone_head_gradient_detached=True,optimizer_steps=0,weights_saved=False)))
if __name__=='__main__':main()
