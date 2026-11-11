"""Read-only paired diagnosis. GT oracles are diagnostic, never inference rules."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust,calc_iou_overlap
from center_joint_contract_20260930 import NAME,file_hash

def metrics(box,gt,seq):
    visible=None if seq.target_visible is None else torch.tensor(seq.target_visible)
    iou,err,norm,valid=calc_seq_err_robust(box,gt,seq.dataset,visible)
    thresholds=torch.arange(0,1.0001,.05,dtype=torch.float64)
    return dict(AUC=float((iou[:,None]>thresholds).double().mean()*100),
        OP50=float((iou>.5).double().mean()*100),OP75=float((iou>.75).double().mean()*100),
        Precision=float((err<=20).double().mean()*100),Norm_Precision=float((norm<=.2).double().mean()*100)),iou,valid

def center_oracle(box,gt,limit=.1):
    # Exact nearest-GT-center solution for a fixed-size rectangle, ignoring image
    # clipping. This is an optimistic static bound, NOT a closed-loop AUC bound.
    delta=((gt[:,:2]+gt[:,2:]/2)-(box[:,:2]+box[:,2:]/2))
    return torch.cat((box[:,:2]+delta.clamp(-limit*box[:,2:],limit*box[:,2:]),box[:,2:]),-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();dest=Path(args.output);assert not dest.exists()
    result=ROOT/'test/tracking_results/uvltrack'/f'{NAME}_100130'
    manifest=json.loads((result/'center_joint_run_manifest.json').read_text())
    assert (manifest['config'],manifest['dataset'],manifest['epoch'],manifest['runid'])==(NAME,'tnl2k',30,100130)
    assert file_hash(manifest['checkpoint'])==manifest['checkpoint_sha256']
    baseline=Path(args.baseline_dir);assert baseline.name=='tnl2k_NLBBOX_300'
    totals={};records=[];buckets={};weighted={};frames=0
    for seq in get_dataset('tnl2k'):
        gt=torch.tensor(seq.ground_truth_rect,dtype=torch.float64)
        paths=[baseline/(seq.name+'.txt'),result/'tnl2k_NLBBOX_030'/(seq.name+'.txt')]
        boxes=[torch.tensor(np.loadtxt(path,delimiter='\t',ndmin=2),dtype=torch.float64) for path in paths]
        assert all(box.shape==gt.shape and torch.isfinite(box).all() and (box[:,2:]>0).all() for box in boxes)
        measures=[];ious=[]
        for label,box in zip(('official','joint_full'),boxes):
            m,i,v=metrics(box,gt,seq);measures.append(m);ious.append(i)
            for key,value in m.items():totals[label+'/'+key]=totals.get(label+'/'+key,0)+value
        valid=v.clone();valid[0]=False;frames+=int(valid.sum())
        gain=ious[1]-ious[0]
        for label,box in zip(('official','joint_full'),boxes):
            safe_gt=torch.where(valid[:,None],gt,box)
            io=calc_iou_overlap(box,safe_gt)
            co=calc_iou_overlap(center_oracle(box,safe_gt),safe_gt)
            centered=torch.cat((safe_gt[:,:2]+safe_gt[:,2:]/2-box[:,2:]/2,box[:,2:]),-1)
            aligned=calc_iou_overlap(centered,safe_gt)
            size_only=torch.cat((box[:,:2]+box[:,2:]/2-safe_gt[:,2:]/2,safe_gt[:,2:]),-1)
            so=calc_iou_overlap(size_only,safe_gt)
            for key,value in {'bounded_center_oracle_gain':co-io,'unbounded_center_oracle_gain':aligned-io,
                              'gt_size_same_center_gain':so-io}.items():
                weighted[label+'/'+key]=weighted.get(label+'/'+key,0)+float(value[valid].sum())
        error=((boxes[0][:,:2]+boxes[0][:,2:]/2-gt[:,:2]-gt[:,2:]/2)/gt[:,2:].clamp_min(1)).norm(dim=-1)
        conditions={'base_iou_ge_075':ious[0]>=.75,'base_iou_050_075':(ious[0]>=.5)&(ious[0]<.75),
            'base_iou_010_050':(ious[0]>=.1)&(ious[0]<.5),'base_iou_lt_010':ious[0]<.1,
            'center_error_gt_diagonal_units_ge_1':error>=1}
        for label,mask in conditions.items():
            mask=mask&valid;entry=buckets.setdefault(label,dict(count=0,gain_sum=0,positive=0,negative=0))
            entry['count']+=int(mask.sum());entry['gain_sum']+=float(gain[mask].sum())
            entry['positive']+=int(((gain>.01)&mask).sum());entry['negative']+=int(((gain<-.01)&mask).sum())
        records.append(dict(sequence=seq.name,frames=len(gt),official=measures[0],joint_full=measures[1],
            delta={k:measures[1][k]-measures[0][k] for k in measures[0]}))
    assert len(records)==700
    mean={k:v/700 for k,v in totals.items()};auc=np.array([r['delta']['AUC'] for r in records])
    for entry in buckets.values():
        n=max(entry['count'],1);entry.update(mean_iou_delta=entry['gain_sum']/n,positive_rate=entry['positive']/n,negative_rate=entry['negative']/n)
    report=dict(input_manifest=manifest,baseline_dir=str(baseline),sequences=700,valid_tracking_frames=frames,
        macro_metrics=mean,macro_delta={k:mean['joint_full/'+k]-mean['official/'+k] for k in records[0]['official']},
        sequence_delta=dict(positive=int((auc>0).sum()),negative=int((auc<0).sum()),median=float(np.median(auc))),
        static_geometry_diagnostics={k:v/frames for k,v in weighted.items()},failure_buckets=buckets,sequences_detail=records,
        limitations=['FULL minus official combines Head and refiner; does not attribute Head alone.',
            'GT geometry bounds ignore image clipping and trajectory changes; not deployable or performance predictions.',
            'TNL2K has already informed development; this is diagnostic, not an untouched generalization evaluation.'],
        optimizer_steps=0,model_changed=False)
    with dest.open('x') as f:json.dump(report,f,indent=2)
    print('RESULT_AUDIT_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('sequences_detail','input_manifest')}),flush=True)
if __name__=='__main__':torch.set_num_threads(1);main()
