"""Sparse native/refined comparison in ONE unchanged joint trajectory. No OFF run."""
import argparse,json,sys,os,time,random
from pathlib import Path
import numpy as np
import cv2,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from center_joint_contract_20260930 import NAME,file_hash
from center_joint_lifecycle_20261001 import MemorySafeCenterEvaluationTracker
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_iou_overlap

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--num-sequences',type=int,default=32)
    p.add_argument('--frames',type=int,default=128);args=p.parse_args();dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    os.chdir(ROOT);torch.set_num_threads(1);cv2.setNumThreads(1)
    checkpoint=ROOT/'change/center_joint_training_exact_metadata_20260930'/NAME/'CenterJoint_ep0030.pth.tar'
    wrapper=MemorySafeCenterEvaluationTracker('tnl2k',checkpoint,100130,epoch=30)
    dataset=get_dataset('tnl2k');ids=sorted(random.Random(20261001).sample(range(len(dataset)),args.num_sequences))
    identity=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),sequence_indices=ids,
        frames_per_sequence=args.frames,seed=20261001,selection='uniform sequence indices, independent of GT and results',
        optimizer_steps=0,full_off_test=False,interpretation='Same current-Head trajectory; does not isolate Head causality from official history')
    (dest/'identity.json').write_text(json.dumps(identity,indent=2));rows=[];start=time.time()
    read=lambda path:cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    stored=ROOT/'test/tracking_results/uvltrack'/f'{NAME}_100130'/'tnl2k_NLBBOX_030'
    official=ROOT/'test/tracking_results/uvltrack/baseline_base/tnl2k_NLBBOX_300'
    for done,i in enumerate(ids):
        seq=dataset[i];tracker=wrapper.tracker_class(wrapper.params,'tnl2k')
        tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
        existing=np.loadtxt(stored/(seq.name+'.txt'),delimiter='\t',ndmin=2)
        reference=np.loadtxt(official/(seq.name+'.txt'),delimiter='\t',ndmin=2)
        try:
            tracker.initialize(read(seq.frames[0]),dict(seq.init_info(),language=seq.language))
            for t,path in enumerate(seq.frames[1:args.frames],1):
                result=tracker.track(read(path));full=np.asarray(result['target_bbox']);native=np.asarray(tracker.last_native_box)
                assert np.max(np.abs(np.trunc(full)-existing[t]))<=1,'Replay differs from saved FULL: '+seq.name
                gt=np.asarray(seq.ground_truth_rect[t]);visible=seq.target_visible is None or seq.target_visible[t]
                if not visible or not np.isfinite(gt).all() or min(gt[2:])<=0:continue
                pred=torch.tensor(np.stack((reference[t],native,full,np.trunc(native),np.trunc(full))),dtype=torch.float64)
                truth=torch.tensor(gt,dtype=torch.float64)[None].expand(5,4)
                iou=calc_iou_overlap(pred,truth).tolist()
                rows.append(dict(sequence=seq.name,frame=t,iou_official_saved=iou[0],iou_native_float=iou[1],
                    iou_full_float=iou[2],iou_native_saved=iou[3],iou_full_saved=iou[4],
                    refiner_gain=iou[2]-iou[1],refiner_gain_saved=iou[4]-iou[3],
                    native_vs_official=iou[3]-iou[0],delta=tracker._center_prediction['delta'][0].cpu().tolist(),
                    center_shift_px=(full[:2]+full[2:]/2-native[:2]-native[2:]/2).tolist()))
        finally:tracker.close();del tracker
        print('SAME_PATH_SEQUENCE',done+1,len(ids),seq.name,'samples',len(rows),'seconds',round(time.time()-start,1),flush=True)
    def aggregate(part):
        gains=np.array([r['refiner_gain_saved'] for r in part]);float_gains=np.array([r['refiner_gain'] for r in part])
        shifts=np.array([r['center_shift_px'] for r in part]);deltas=np.array([r['delta'] for r in part])
        return dict(samples=len(part),mean_refiner_iou_gain=float(gains.mean()),mean_float_iou_gain=float(float_gains.mean()),
            positive_gt001=float((gains>.01).mean()),negative_lt_minus001=float((gains<-.01).mean()),
            mean_native_vs_official=float(np.mean([r['native_vs_official'] for r in part])),
            delta_mean=deltas.mean(0).tolist(),delta_abs=float(np.abs(deltas).mean()),
            saturation_rate=float((np.abs(deltas)>.095).mean()),center_shift_mean_px=shifts.mean(0).tolist())
    summary=aggregate(rows);buckets={}
    for name,low,high in [('native_high',.75,1.01),('native_mid',.5,.75),('native_low',0,.5)]:
        part=[r for r in rows if low<=r['iou_native_saved']<high]
        if part:buckets[name]=aggregate(part)
    report=dict(identity=identity,summary=summary,buckets=buckets,seconds=time.time()-start,rows=rows,
        limitation='Only early prefixes of 32 sequences; not full-sequence AUC, not a training/tuning set')
    with (dest/'report.json').open('x') as f:json.dump(report,f,indent=2)
    print('SAME_PATH_AUDIT_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if __name__=='__main__':
    with torch.no_grad():main()
