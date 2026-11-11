"""Same objective for fit and monitor. Does not construct an optimizer."""
import torch
from .objective import objective

class SecondOrderObjectiveAdapter:
    def __init__(self,model):self.model=model
    def __call__(self,inputs,fixed_rqc_box,truth,image_hw):
        pred=self.model(**inputs)
        loss,stats,boxes=objective(self.model,pred,inputs,fixed_rqc_box,truth,image_hw)
        return loss,stats,dict(pred,boxes=boxes)
    @torch.no_grad()
    def monitor(self,*args,**kwargs):
        if self.model.training:raise ValueError("Monitor requires eval mode")
        return self(*args,**kwargs)
