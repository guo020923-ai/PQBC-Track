"""Development closed-loop comparison only; no production tracker edits."""
import os,sys,json,inspect,time,argparse,types
from pathlib import Path
import torch,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prepare_locked_refiner_protocol_20260930 import load_role
from local_box_refiner_20260930 import LocalBoxRefiner
from head_quality_features_20260930 import HeadQualityCapture
from audit_jq_natural_20260930 import NAME
from audit_bq_contract_20260929 import ious
import lib.test.tracker.uvltrack as module
from lib.test.parameter.uvltrack import parameters


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--count',type=int,default=8)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    os.chdir(str(ROOT));torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(42)
    base=ROOT/'change/rqc_dev_v3';manifest=load_role(base/'locked_refiner_protocol_20260930','dev')[args.shard::args.count]
    if args.smoke:manifest=manifest[:1]
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    source=inspect.getsource(module);needle='        pred_box_net = pred_boxes[selected_index]\n'
    assert source.count(needle)==1
    injected=needle+'        if self.local_enabled:\n            pred_box_net = self.local_refine(pred_box_net, selected_index)\n'
    namespace=dict(module.__dict__);exec(compile(source.replace(needle,injected),'<isolated_local_refiner>', 'exec'),namespace)
    params=parameters(NAME,epoch=30);params.debug=0;tracker=namespace['UVLTrack'](params,'tnl2k')
    tracker.local_enabled=False
    model=LocalBoxRefiner().cuda().eval()
    if not args.smoke:
        weights=torch.load(str(base/'locked_refiner_parallel_20260930/pilot/point_ep20.pt'),map_location='cpu')
        assert weights['variant']=='point';model.load_state_dict(weights['model'],strict=True)
    cap=HeadQualityCapture(tracker.network.box_head);cap.__enter__()
    def refine(self,box,index):
        device=cap.maps['conv_bbox'].device
        feature=cap.gather(torch.tensor([[int(index)]],device=device))[:,0]
        return model(feature,box.to(device)[None])['boxes'][0].cpu()
    tracker.local_refine=types.MethodType(refine,tracker)
    def keep_original(m,inputs,out):
        changed=dict(out);changed['decision_score']=torch.zeros_like(out['decision_score']);changed['decision_score'][:,0]=1
        return changed
    handle=tracker.network.rqc.register_forward_hook(keep_original)
    read=lambda f:cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB)
    results=[];start=time.time()
    try:
        for m in manifest:
            path=Path(m['path']);tnl=m['domain']=='tnl2k_train'
            frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.png','.jpeg'))
            gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2);assert len(gt)==len(frames)
            visible=np.ones(len(gt),bool)
            if not tnl:
                for n in ('full_occlusion.txt','out_of_view.txt'):visible &= np.loadtxt(path/n,delimiter=',').reshape(-1)==0
            visible &= np.isfinite(gt).all(-1)&(gt[:,2:]>0).all(-1);visible[0]=False
            language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
            end=min(len(frames),25) if args.smoke else len(frames);outputs={}
            for enabled in (False,True):
                tracker.local_enabled=enabled;tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
                tracker.initialize(read(frames[0]),dict(init_bbox=gt[0].tolist(),language=language))
                boxes=[gt[0].tolist()]
                for t in range(1,end):boxes.append(tracker.track(read(frames[t]))['target_bbox'])
                outputs[str(enabled)]=np.asarray(boxes)
            if args.smoke:np.testing.assert_array_equal(outputs['False'],outputs['True'])
            valid=visible[:end];a=np.array([ious([b],g)[0] for b,g in zip(outputs['False'],gt[:end])])[valid]
            b=np.array([ious([box],g)[0] for box,g in zip(outputs['True'],gt[:end])])[valid]
            gain=b-a
            row=dict(sequence=m['name'],domain=m['domain'],frames=end-1,valid_frames=len(gain),
                baseline_mean_iou=float(a.mean()),refined_mean_iou=float(b.mean()),gain=float(gain.mean()),
                beneficial=float((gain>.01).mean()),harmful=float((gain<-.01).mean()),severe=float((gain<-.2).mean()))
            results.append(row)
            np.savez_compressed(dest/f'sequence_{len(results):02d}.npz',baseline=outputs['False'],refined=outputs['True'],valid=valid,gt=gt[:end])
            print('CLOSED_SEQUENCE',json.dumps(row),flush=True)
    finally:handle.remove();cap.__exit__()
    report=dict(results=results,elapsed=time.time()-start,smoke=args.smoke,optimizer_steps=0,
        note='Development mean visible-frame IoU, NOT official benchmark AUC. Prompt updates remain original response based.')
    (dest/'report.json').write_text(json.dumps(report,indent=2));print('CLOSED_COMPLETE',flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
