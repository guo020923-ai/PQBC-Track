"""Read-only reproduction of crop-side metadata mismatch; no optimizer updates."""
import argparse,json,math
from pathlib import Path
import torch
from train_center_joint_20260930 import prepare
from center_formal_contract_20260930 import MetadataTrackProcessing

original=MetadataTrackProcessing._get_jittered_box

def observed(self,box,mode):
    result=original(self,box,mode)
    if self._center_recording and mode=='search':
        x,y,w,h=result.tolist()
        native=math.ceil(math.sqrt(w*h)*self.search_area_factor[mode])
        recorded=self._center_meta[-1]['side']
        if native!=recorded:
            print('CROP_SIDE_MISMATCH',json.dumps(dict(extract_box=[x,y,w,h],
                fp32_product=float(result[2]*result[3]),python_product=w*h,
                native_side=native,recorded_side=recorded,gt=box.tolist())),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',type=int,default=1300)
    p.add_argument('--report',required=True);args=p.parse_args()
    report=Path(args.report)
    if report.exists():raise FileExistsError(report)
    MetadataTrackProcessing._get_jittered_box=observed
    net,actor,loaders,optimizer,scheduler,identity=prepare()
    completed=0
    try:
        for index,data in enumerate(loaders[0]):
            completed=index+1
            if completed%100==0:print('METADATA_BATCH',completed,flush=True)
            if completed>=args.batches:break
        result=dict(completed_batches=completed,optimizer_steps=0,reproduced=False)
    except AssertionError as error:
        result=dict(completed_batches=completed,optimizer_steps=0,reproduced=True,error=str(error))
    assert not optimizer.state
    with report.open('x') as out:json.dump(result,out,indent=2)
    print('METADATA_DIAGNOSIS',json.dumps(result),flush=True)

if __name__=='__main__':main()
