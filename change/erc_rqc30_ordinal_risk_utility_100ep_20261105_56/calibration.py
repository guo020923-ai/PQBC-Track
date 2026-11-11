import json,os
from pathlib import Path
import numpy as np,torch
from change.erc_rqc30_harm_aware_joint_utility_100ep_20261105_49.training import load_cache
from .checkpoint import load_numeric,digest
from .config import PROTOCOL,calibration_mask
def calibrate(numeric,cache):
 numeric=Path(numeric);_,a=load_cache(cache);m,z=load_numeric(numeric);m.eval();rows=np.arange(30000);idx=rows[calibration_mask(rows)];scores=[]
 with torch.inference_mode():
  for b in range(0,len(idx),256):
   q=idx[b:b+256];e=torch.from_numpy(a['evidence'][q]).float();x=torch.from_numpy(a['executable'][q]).bool();anchor=torch.zeros(len(q),dtype=torch.long);scores.append(m(e,x,anchor)['risk_score'])
 score=torch.cat(scores).numpy();x=a['executable'][idx].astype(bool);u=a['utility'][idx].astype(float);src=a['source'][idx];r=np.arange(len(idx));curve=[]
 for t in PROTOCOL['risk_threshold_grid']:
  allow=x&(score>=t);allow[:,0]=False;active=allow.any(1);cand=np.where(allow,score,-np.inf).argmax(1);cu=u[r,cand];chosen=np.where(active,cu,0.);worst=0.;per={}
  for s in sorted(np.unique(src)):
   q=(src==s)&active;harm=float((cu[q]<=0).mean()) if q.any() else 0.;worst=max(worst,harm);per[str(s.item() if hasattr(s,'item') else s)]={'selected':int(q.sum()),'coverage':float(q.sum()/(src==s).sum()),'harm_rate':harm,'mean_utility_all':float(chosen[src==s].mean())}
  curve.append({'threshold':t,'coverage':float(active.mean()),'selected':int(active.sum()),'mean_utility_all':float(chosen.mean()),'mean_utility_selected':float(cu[active].mean()) if active.any() else 0.,'harm_rate':float((cu[active]<=0).mean()) if active.any() else 0.,'worst_source_harm_rate':worst,'per_source':per})
 feasible=[q for q in curve if q['coverage']>=PROTOCOL['minimum_calibration_coverage'] and q['mean_utility_all']>0 and q['worst_source_harm_rate']<=PROTOCOL['maximum_worst_source_harm_rate']]
 diagnostic={'status':'TRAIN_ORDINAL_CALIBRATION_DIAGNOSTIC','curve':curve,'calibration_rows':len(idx),'checkpoint_sha256':z['source_checkpoint_sha256'],'numeric_npz_sha256':z['numeric_npz_sha256'],'test_data_read':False};d=numeric/'ordinal_calibration_diagnostic.json'
 with d.open('x') as f:json.dump(diagnostic,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 if not feasible:raise RuntimeError('no ordinal TRAIN threshold meets sealed risk constraints')
 s=max(feasible,key=lambda q:(q['mean_utility_all'],q['threshold']));result={'status':'TRAIN_ORDINAL_CALIBRATION_VERIFIED','threshold':s['threshold'],'selected':s,'curve':curve,'calibration_rows':len(idx),'checkpoint_sha256':z['source_checkpoint_sha256'],'numeric_npz_sha256':z['numeric_npz_sha256'],'test_data_read':False};p=numeric/'ordinal_calibration.json'
 with p.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 result['calibration_sha256']=digest(p);return result
