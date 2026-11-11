import hashlib,json,os
from pathlib import Path
import numpy as np,torch
from .config import PROTOCOL
from .model import OrdinalRiskUtilityERC,VERSION
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def export_numeric(ck,d):
 ck,d=Path(ck),Path(d);s=torch.load(str(ck),map_location='cpu')
 if s.get('kind')!=VERSION or s.get('protocol')!=PROTOCOL or s.get('epoch')!=100 or s.get('optimizer_steps')!=187500:raise ValueError('identity')
 m=OrdinalRiskUtilityERC();m.load_state_dict(s['model']);a={k:v.numpy() for k,v in m.state_dict().items()};d.mkdir();n=d/'erc_epoch100_numeric.npz'
 with n.open('xb') as f:np.savez(f,**a);f.flush();os.fsync(f.fileno())
 z={'format':'numpy_npz_numeric_only_no_pickle','kind':VERSION,'model_version':VERSION,'epoch':100,'optimizer_steps':187500,'source_checkpoint':str(ck),'source_checkpoint_sha256':digest(ck),'numeric_npz_sha256':digest(n),'tensor_keys':sorted(a),'protocol':PROTOCOL}
 with (d/'manifest.json').open('x') as f:json.dump(z,f,indent=2);f.flush();os.fsync(f.fileno())
 return z
def load_numeric(d,device='cpu'):
 d=Path(d);z=json.loads((d/'manifest.json').read_text());n=d/'erc_epoch100_numeric.npz'
 if z.get('model_version')!=VERSION or z.get('protocol')!=PROTOCOL or digest(n)!=z.get('numeric_npz_sha256'):raise ValueError('numeric identity')
 m=OrdinalRiskUtilityERC().to(device);ref=m.state_dict();state={}
 with np.load(n,allow_pickle=False) as a:
  if set(a.files)!=set(ref):raise ValueError('keys')
  for k,v in ref.items():state[k]=torch.from_numpy(a[k].copy()).to(device=device,dtype=v.dtype)
 m.load_state_dict(state);return m,z
