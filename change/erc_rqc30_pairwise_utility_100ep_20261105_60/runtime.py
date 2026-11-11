from copy import deepcopy
import json
from pathlib import Path
import torch
from .model import PairwiseUtilityERC
def _history(s):
 out={}
 for k in ('state','prompt','max_score','pred_box_net','out_dict'):
  if hasattr(s,k):
   v=getattr(s,k);out[k]=v.detach().clone() if torch.is_tensor(v) else deepcopy(v)
 return out
def _equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
 return a==b
class OutputOnlyPairwiseRuntime:
 def __init__(self,source,model,provider,calibration_path):
  if type(model) is not PairwiseUtilityERC:raise TypeError('exact pairwise model required')
  c=json.loads(Path(calibration_path).read_text())
  if c.get('status')!='TRAIN_PAIRWISE_CALIBRATION_VERIFIED' or c.get('test_data_read') is not False:raise ValueError('verified TRAIN calibration required')
  self.threshold=float(c['threshold']);self.source,self.model,self.provider=source,model,provider;self.last=None
 @torch.no_grad()
 def initialize(self,image,info):self.last=None;return self.source.initialize(image,info)
 @torch.no_grad()
 def track(self,image,explore=False):
  if explore:raise ValueError('official inference cannot explore')
  previous=list(self.source.state);rqc=self.source.track(image,None);after=_history(self.source);batch=self.provider(self.source,image,rqc,previous);p=self.model(batch['evidence'],batch['executable'],batch['anchor_index']);index=self.model.choose(p['utility_score'],batch['executable'],batch['anchor_index'],self.threshold);rows=torch.arange(len(batch['boxes']),device=batch['boxes'].device);chosen=batch['boxes'][rows,index]
  if len(chosen)!=1 or not _equal(after,_history(self.source)):raise RuntimeError('invalid output-only state')
  action=int(index[0]);box=chosen[0].detach().double().cpu().tolist();self.last={'index':action,'anchor_index':int(batch['anchor_index'][0]),'rqc_box':list(rqc['target_bbox']),'output_box':box,'utility_score':float(p['utility_score'][0,action]),'threshold':self.threshold,'source_history_feedback':False};return dict(rqc,target_bbox=box)
