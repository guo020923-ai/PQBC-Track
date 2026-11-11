import json,numpy as np,torch
from .config import calibration_mask,validate_protocol
from .model import PairwiseUtilityERC
from .objective import pairwise_utility_loss
def main():
 validate_protocol();assert not torch.cuda.is_initialized();m=PairwiseUtilityERC();e=torch.randn(8,25,48);x=torch.ones(8,25,dtype=torch.bool);a=torch.zeros(8,dtype=torch.long);u=torch.randn(8,25)*.05;u[:,0]=0;o=m(e,x,a);loss,_=pairwise_utility_loss(o,x,a,u);loss.backward();assert torch.isfinite(loss);s=torch.zeros(3,25);s[0,1]=.2;assert m.choose(s,x[:3],a[:3],.1).tolist()==[1,0,0];held=calibration_mask(np.arange(30000));print(json.dumps({'status':'PAIRWISE_CPU_PASS','training_rows':int((~held).sum()),'calibration_rows':int(held.sum()),'gpu_operations':0},sort_keys=True))
if __name__=='__main__':main()
