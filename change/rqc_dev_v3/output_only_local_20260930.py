"""Same trained refiner, output-only correction; native internal state is untouched."""
import os,sys,json,time,math,argparse
from pathlib import Path
import torch,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prepare_locked_refiner_protocol_20260930 import load_role
from local_box_refiner_20260930 import LocalBoxRefiner,decode_image_boxes
from head_quality_features_20260930 import HeadQualityCapture
from audit_jq_natural_20260930 import NAME
from audit_bq_contract_20260929 import ious
from lib.test.tracker.uvltrack import UVLTrack
from lib.test.parameter.uvltrack import parameters


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--count',type=int,default=8)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    os.chdir(str(ROOT));torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(42)
    base=ROOT/'change/rqc_dev_v3';manifest=load_role(base/'locked_refiner_protocol_20260930','dev')[args.shard::args.count]
    if args.smoke:manifest=manifest[:1]
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    params=parameters(NAME,epoch=30);params.debug=0;tracker=UVLTrack(params,'tnl2k')
    model=LocalBoxRefiner().cuda().eval()
    checkpoint=torch.load(str(base/'locked_refiner_parallel_20260930/pilot/point_ep20.pt'),map_location='cpu')
    assert checkpoint['variant']=='point';model.load_state_dict(checkpoint['model'],strict=True)
    cap=HeadQualityCapture(tracker.network.box_head);cap.__enter__();capture={}
    def hook(m,inputs,out):
        batch=inputs[0];feature=cap.gather(batch['indices'][:,:1])[:,0]
        capture['refined']=model(feature,batch['boxes'][:,0])['boxes'].cpu()
        answer=dict(out);answer['decision_score']=torch.zeros_like(out['decision_score']);answer['decision_score'][:,0]=1
        return answer
    handle=tracker.network.rqc.register_forward_hook(hook);start=time.time();rows=[]
    read=lambda f:cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB)
    try:
        for j,m in enumerate(manifest):
            path=Path(m['path']);tnl=m['domain']=='tnl2k_train'
            frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.png','.jpeg'))
            gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2);assert len(gt)==len(frames)
            reference=np.load(str(base/f'closed_loop_local_20260930/shard_{args.shard}/sequence_{j+1:02d}.npz'))
            np.testing.assert_array_equal(reference['gt'],gt)
            visible=reference['valid'];end=min(25,len(frames)) if args.smoke else len(frames)
            language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
            tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
            tracker.initialize(read(frames[0]),dict(init_bbox=gt[0].tolist(),language=language))
            baseline=[gt[0].tolist()];refined=[gt[0].tolist()]
            for t in range(1,end):
                previous=np.asarray(tracker.state);side=math.ceil(math.sqrt(previous[2]*previous[3])*params.search_factor)
                origin=previous[:2]+previous[2:]/2-side/2;image=read(frames[t])
                prediction=tracker.track(image)['target_bbox']
                np.testing.assert_array_equal(prediction,reference['baseline'][t])
                state=list(tracker.state)
                output=decode_image_boxes(capture['refined'],torch.tensor(origin,dtype=torch.float32)[None],
                    torch.tensor([side]),torch.tensor([image.shape[:2]]))[0].tolist()
                assert tracker.state==state
                baseline.append(prediction);refined.append(output)
            a=np.array([ious([box],g)[0] for box,g in zip(baseline,gt[:end])])[visible[:end]]
            b=np.array([ious([box],g)[0] for box,g in zip(refined,gt[:end])])[visible[:end]]
            gain=b-a
            row=dict(sequence=m['name'],domain=m['domain'],frames=end-1,valid_frames=len(gain),
                baseline_mean_iou=float(a.mean()),refined_mean_iou=float(b.mean()),gain=float(gain.mean()),
                beneficial=float((gain>.01).mean()),harmful=float((gain<-.01).mean()),severe=float((gain<-.2).mean()),
                native_trajectory_exact=True)
            rows.append(row);np.savez_compressed(dest/f'sequence_{j+1:02d}.npz',baseline=baseline,refined=refined,gt=gt[:end],valid=visible[:end])
            print('OUTPUT_SEQUENCE',json.dumps(row),flush=True)
    finally:handle.remove();cap.__exit__()
    (dest/'report.json').write_text(json.dumps(dict(results=rows,elapsed=time.time()-start,smoke=args.smoke,
        optimizer_steps=0,note='Output-only corrections never update search/online state. Development visible-frame IoU, not official AUC.'),indent=2))
    print('OUTPUT_COMPLETE',flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
