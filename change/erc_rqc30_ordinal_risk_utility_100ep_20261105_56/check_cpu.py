import json,numpy as np,torch
from .config import calibration_mask,validate_protocol
from .model import OrdinalRiskUtilityERC
from .objective import ordinal_risk_loss,ordinal_target
def main():
 validate_protocol()
 if torch.cuda.is_initialized():raise RuntimeError('CPU only')
 rows=np.arange(30000);held=calibration_mask(rows);m=OrdinalRiskUtilityERC();e=torch.randn(8,25,48);x=torch.ones(8,25,dtype=torch.bool);a=torch.zeros(8,dtype=torch.long);u=torch.randn(8,25)*.05;u[:,0]=0;p=m(e,x,a);loss,s=ordinal_risk_loss(p,x,a,u);loss.backward();assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters());assert p['risk_score'][:,0].eq(0).all();assert ordinal_target(torch.tensor([-.1,.01,.1])).tolist()==[0,1,2]
 score=torch.zeros(3,25);score[0,1]=.4;score[1,1]=.2;xx=torch.ones(3,25,dtype=torch.bool);xx[2,1]=False;ch=m.choose(score,xx,a[:3],.3);assert ch.tolist()==[1,0,0]
 print(json.dumps({'status':'CPU_ADMISSION_PASS','single_head_selection_and_gate':True,'calibration_rows':int(held.sum()),'training_rows':int((~held).sum()),'fixed_epochs':100,'optimizer_steps':187500,'gpu_operations':0},sort_keys=True))
if __name__=='__main__':main()
