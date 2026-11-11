"""Accumulate detached sufficient statistics on device; copy only for reports."""
import torch
from .objective import COUNTS


class DeviceStatistics:
    def __init__(self,domain_ids):
        self.domain_ids=tuple(domain_ids);self.keys=None;self.sums=None

    @torch.no_grad()
    def add(self,samples,source_ids):
        keys=tuple(samples)
        if self.keys is None:self.keys=keys
        if keys!=self.keys:raise ValueError('Statistic schema changed')
        values=torch.stack([samples[k].detach().double() for k in keys])
        if source_ids.shape!=(values.shape[1],):raise ValueError('Wrong source metadata')
        weights=torch.stack([torch.ones_like(source_ids,dtype=torch.double)]+
            [source_ids.eq(i).double() for i in self.domain_ids])
        sums=weights@values.t()
        self.sums=sums if self.sums is None else self.sums+sums

    def result(self):
        if self.sums is None:raise ValueError('No observations')
        raw=self.sums.detach().cpu().tolist()  # one device->host transfer per report
        result=[]
        for row in raw:
            values=dict(zip(self.keys,row));n=values['valid_samples']
            item={k:(v if k in COUNTS else v/max(1,n)) for k,v in values.items()}
            item['loss']=item['gain_MSE_pp2'];result.append(item)
        return dict(train=result[0],domains=dict(zip(self.domain_ids,result[1:])))
