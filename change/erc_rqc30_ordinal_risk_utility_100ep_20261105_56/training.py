"""Fixed 100-epoch ordinal-risk training."""
import hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np,torch
from change.erc_rqc30_harm_aware_joint_utility_100ep_20261105_49.training import load_cache
from .config import PROTOCOL,calibration_mask,validate_protocol
from .model import OrdinalRiskUtilityERC,VERSION
from .objective import ordinal_risk_loss
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);q=p.with_name(p.name+'.partial')
 with q.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def lr(e):
 if e<=5:return PROTOCOL['peak_learning_rate']*e/5
 x=(e-5)/95;return PROTOCOL['minimum_learning_rate']+.5*(PROTOCOL['peak_learning_rate']-PROTOCOL['minimum_learning_rate'])*(1+math.cos(math.pi*x))
def train_epochs(out,cache):
 validate_protocol();out,cache=Path(out),Path(cache);_,a=load_cache(cache);rows=np.arange(30000);held=calibration_mask(rows);train=rows[~held];seed=PROTOCOL['training_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=OrdinalRiskUtilityERC().cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=PROTOCOL['peak_learning_rate'],weight_decay=PROTOCOL['weight_decay']);cks=out/'checkpoints';cks.mkdir();gen=torch.Generator().manual_seed(seed);done=0;started=time.monotonic();recent=[];keys=('loss','harmful_accuracy','marginal_accuracy','actionable_accuracy','predicted_keep_rate','harmful_deploy_rate','mean_deployed_utility','mean_risk_score')
 with (out/'stats.jsonl').open('x') as sf:
  for epoch in range(1,101):
   rate=lr(epoch)
   for g in opt.param_groups:g['lr']=rate
   pieces=[];remain=30000
   while remain:
    o=torch.randperm(len(train),generator=gen).numpy();take=min(remain,len(o));pieces.append(train[o[:take]]);remain-=take
   order=np.concatenate(pieces);sums={k:0. for k in keys};es=time.monotonic()
   for step in range(1875):
    ss=time.monotonic();idx=order[step*16:(step+1)*16];e=torch.from_numpy(a['evidence'][idx]).cuda().float();x=torch.from_numpy(a['executable'][idx]).cuda().bool();u=torch.from_numpy(a['utility'][idx]).cuda().float();anchor=torch.zeros(len(idx),dtype=torch.long,device='cuda');pred=m(e,x,anchor);loss,st=ordinal_risk_loss(pred,x,anchor,u);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),PROTOCOL['gradient_clip']);
    if not torch.isfinite(norm):raise FloatingPointError('gradient')
    opt.step();done+=1
    for k in keys:sums[k]+=float(st[k])
    recent.append(time.monotonic()-ss);recent=recent[-200:]
    if step==0 or (step+1)%25==0:
     sps=sum(recent)/len(recent);atomic(out/'progress.json',{'phase':'EPOCH_TRAINING','epoch':epoch,'total_epochs':100,'step_in_epoch':step+1,'steps_per_epoch':1875,'completed_steps':done,'total_steps':187500,'learning_rate':rate,'recent_seconds_per_step':sps,'eta_seconds':(187500-done)*sps,'updated_unix':time.time()})
   rec={'epoch':epoch,'seconds':time.monotonic()-es,'learning_rate':rate,'stats':{k:v/1875 for k,v in sums.items()}};sf.write(json.dumps(rec,allow_nan=False)+'\n');sf.flush();os.fsync(sf.fileno());p=cks/f'ERC_epoch_{epoch:04d}.pth.tar';q=p.with_suffix(p.suffix+'.partial');torch.save({'kind':VERSION,'model_version':VERSION,'protocol':PROTOCOL,'cache_manifest_sha256':digest(cache/'manifest.json'),'epoch':epoch,'optimizer_steps':done,'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'optimizer':opt.state_dict()},q);os.replace(q,p)
 final=cks/'ERC_epoch_0100.pth.tar';result={'status':'TRAIN_COMPLETE_PENDING_READONLY_AUDIT','fixed_epoch':100,'optimizer_steps':187500,'checkpoint':str(final),'checkpoint_sha256':digest(final),'training_seconds':time.monotonic()-started,'train_rows':int((~held).sum()),'calibration_rows':int(held.sum()),'cache_manifest_sha256':digest(cache/'manifest.json')}
 with (out/'completed.json').open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
 atomic(out/'progress.json',dict(result,phase='TRAINING_COMPLETE'));return result
