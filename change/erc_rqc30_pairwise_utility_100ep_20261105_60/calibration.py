import json,os
from pathlib import Path
import numpy as np,torch
from change.erc_rqc30_harm_aware_joint_utility_100ep_20261105_49.training import load_cache
from .checkpoint import load_numeric,digest
from .config import PROTOCOL,calibration_mask
def calibrate(numeric,cache):
 numeric=Path(numeric);_,a=load_cache(cache);m,z=load_numeric(numeric);m.eval();idx=np.arange(30000)[calibration_mask(np.arange(30000))];scores=[]
 with torch.inference_mode():
  for start in range(0,len(idx),256):
   q=idx[start:start+256];scores.append(m(torch.from_numpy(a['evidence'][q]).float(),torch.from_numpy(a['executable'][q]).bool(),torch.zeros(len(q),dtype=torch.long))['utility_score'])
 score=torch.cat(scores).numpy();x=a['executable'][idx].astype(bool);x[:,0]=False;u=a['utility'][idx].astype(float);src=a['source'][idx];r=np.arange(len(idx));best=np.where(x,score,-np.inf).max(1);curve=[]
 for target_coverage in PROTOCOL['calibration_target_coverage']:
  threshold=float(np.quantile(best,1-target_coverage,interpolation='higher'));allowed=x&(score>=threshold);active=allowed.any(1);candidate=np.where(allowed,score,-np.inf).argmax(1);cu=u[r,candidate];chosen=np.where(active,cu,0.);worst=0.;per={}
  for value in sorted(np.unique(src)):
   q=(src==value)&active;harm=float((cu[q]<=0).mean()) if q.any() else 0.;worst=max(worst,harm);per[str(int(value))]={'selected':int(q.sum()),'coverage':float(q.sum()/(src==value).sum()),'harm_rate':harm,'mean_utility_all':float(chosen[src==value].mean())}
  curve.append({'target_coverage':target_coverage,'threshold':threshold,'coverage':float(active.mean()),'selected':int(active.sum()),'mean_utility_all':float(chosen.mean()),'mean_utility_selected':float(cu[active].mean()),'harm_rate':float((cu[active]<=0).mean()),'worst_source_harm_rate':worst,'per_source':per})
 diagnostic={'status':'TRAIN_PAIRWISE_CALIBRATION_DIAGNOSTIC','curve':curve,'calibration_rows':len(idx),'checkpoint_sha256':z['source_checkpoint_sha256'],'numeric_npz_sha256':z['numeric_npz_sha256'],'test_data_read':False};dp=numeric/'pairwise_calibration_diagnostic.json'
 with dp.open('x') as f:json.dump(diagnostic,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 feasible=[q for q in curve if q['coverage']>=PROTOCOL['minimum_calibration_coverage'] and q['mean_utility_all']>0 and q['worst_source_harm_rate']<=PROTOCOL['maximum_worst_source_harm_rate']]
 if not feasible:raise RuntimeError('no pairwise TRAIN threshold meets sealed risk constraints')
 selected=max(feasible,key=lambda q:(q['mean_utility_all'],q['threshold']));result={'status':'TRAIN_PAIRWISE_CALIBRATION_VERIFIED','threshold':selected['threshold'],'selected':selected,'curve':curve,'calibration_rows':len(idx),'checkpoint_sha256':z['source_checkpoint_sha256'],'numeric_npz_sha256':z['numeric_npz_sha256'],'test_data_read':False};p=numeric/'pairwise_calibration.json'
 with p.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 result['calibration_sha256']=digest(p);return result
