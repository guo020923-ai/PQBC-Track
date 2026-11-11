"""Observe the two EXISTING frozen RQC hidden passes, never run the net again."""
import torch


class NativeContextTap:
    def __init__(self,refiner):
        if not hasattr(refiner,'net') or len(refiner.net)!=4:
            raise ValueError('Exact four-layer DirectionalCenterRefiner required')
        self.active=False;self.values=[]
        self.handle=refiner.net[-2].register_forward_hook(self._capture)

    def _capture(self,module,args,result):
        if self.active:
            if len(self.values)>=2:
                raise ValueError('Unexpected extra native hidden pass; fail closed')
            self.values.append(result.detach())
        # Returning None is essential: native output is never replaced.

    def begin(self):
        if self.active:raise RuntimeError('Nested capture is forbidden')
        self.values=[];self.active=True

    def finish(self):
        if not self.active or len(self.values)!=2:
            raise ValueError('Native plus/minus hidden observations incomplete')
        plus,minus=self.values
        if plus.shape!=minus.shape or plus.ndim!=3 or plus.shape[1:]!=(2,64):
            raise ValueError('Wrong native hidden schema')
        # Exact old frozen_context arithmetic/order, only its duplicate net calls
        # have been removed. There is no detach/normalization change to RQC.
        answer=((plus+minus)*.5).mean(1).detach()
        if not torch.isfinite(answer).all():raise ValueError('Nonfinite native context')
        return answer

    def cancel(self):self.active=False;self.values=[]
    def close(self):self.cancel();self.handle.remove()
