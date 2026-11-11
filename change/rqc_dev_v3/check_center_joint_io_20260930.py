"""Synthetic Head/refiner roundtrip + native state equivalence, zero optim steps."""
import os,sys,time,json,copy,subprocess,random
from pathlib import Path
import numpy as np
import torch,cv2
from train_center_joint_20260930 import ROOT,prepare,restore_resume
from center_joint_tracker_20260930 import JointCenterEvaluationTracker
from center_joint_contract_20260930 import NAME,verify_joint,fixed_snapshot
from prepare_locked_refiner_protocol_20260930 import load_role
from lib.test.parameter.uvltrack import parameters
from lib.test.tracker.uvltrack import UVLTrack
from lib.test.evaluation import Sequence
from lib.test.evaluation.running import run_dataset
from test_center_joint_20260930 import verify_outputs
from evaluate_center_joint_20260930 import run_official_metrics

def main():
    os.chdir(str(ROOT));torch.set_num_threads(1);cv2.setNumThreads(1)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    net,actor,loaders,optimizer,scheduler,identity=prepare()
    dest=ROOT/'change/rqc_dev_v3'/('center_joint_io_check_'+time.strftime('%Y%m%d_%H%M%S'));dest.mkdir(exist_ok=False)
    fixed=fixed_snapshot(net)
    # Deliberately distinguish the saved Head from official E300, without training.
    with torch.no_grad():net.box_head.conv_offset[-1].bias.add_(.01)
    head={k:v.detach().cpu().clone() for k,v in net.box_head.state_dict().items()}
    # Simulated Adam moments exercise device restoration; no optimizer.step.
    for group in optimizer.param_groups:
        p=group['params'][0]
        optimizer.state[p]={'step':torch.tensor(1.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
    sched_state=scheduler.state_dict();sched_state['last_epoch']=1
    payload=dict(variant='center_safe_joint',epoch=1,identity=identity,head=head,synthetic_check=True,
        model={k:v.detach().cpu().clone() for k,v in net.output_refiner.state_dict().items()},
        optimizer=optimizer.state_dict(),scheduler=sched_state,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    checkpoint=dest/'CenterJoint_ep0001.pth.tar';torch.save(payload,str(checkpoint))
    (dest/'protocol.json').write_text(json.dumps(identity));(dest/'stats.jsonl').write_text(json.dumps(dict(epoch=1))+'\n')
    with torch.no_grad():net.box_head.conv_offset[-1].bias.zero_()
    assert restore_resume(net,optimizer,scheduler,identity,dest,checkpoint,allow_synthetic=True)==2
    assert all(torch.equal(head[k],v.detach().cpu()) for k,v in net.box_head.state_dict().items())
    assert all(state['exp_avg'].device==p.device for p,state in optimizer.state.items())
    verify_joint(net,fixed)
    try:restore_resume(net,optimizer,scheduler,identity,dest,checkpoint)
    except ValueError as e:assert 'Synthetic check' in str(e)
    else:raise AssertionError('Synthetic resume not rejected')
    rejected=subprocess.run([sys.executable,str(Path(__file__).parent/'test_center_joint_20260930.py'),
        '--dataset','tnl2k','--checkpoint',str(checkpoint),'--epoch','1','--runid','93198'],capture_output=True,text=True)
    assert rejected.returncode!=0 and 'synthetic check weights are forbidden' in rejected.stderr
    missing=dict(payload);missing.pop('head');missing_file=dest/'missing_head_synthetic.pt';torch.save(missing,str(missing_file))
    # Never serialize an official-training weight from this check.
    manifest=load_role(ROOT/'change/rqc_dev_v3/locked_refiner_protocol_20260930','dev')
    chosen=[next(m for m in manifest if m['domain']==d) for d in ('lasot_train','tnl2k_train')]
    read=lambda p:cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
    results={};smoke_sequences=[]
    for variant in ('zero','synthetic_nonzero'):
        model={k:v.clone() for k,v in payload['model'].items()}
        if variant!='zero':model['net.3.bias'].fill_(.05)
        saved=dict(payload,model=model)
        file=dest/(variant+'.pt');torch.save(saved,str(file))
        native_params=parameters('baseline_base',epoch=300);native_params.debug=0
        native=UVLTrack(native_params,'tnl2k');native.network.box_head.load_state_dict(head,strict=True)
        wrapper=JointCenterEvaluationTracker('tnl2k',file,93198,epoch=1)
        candidate=wrapper.create_tracker(wrapper.params)
        assert all(torch.equal(head[k],v.detach().cpu()) for k,v in candidate.network.box_head.state_dict().items())
        checked=changes=0
        try:
            for row in chosen:
                path=Path(row['path']);tnl=row['domain']=='tnl2k_train'
                frames=sorted(f for f in (path/('imgs' if tnl else 'img')).iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))[:24]
                gt=np.loadtxt(path/'groundtruth.txt',delimiter=',',ndmin=2)[:len(frames)]
                language=(path/('language.txt' if tnl else 'nlp.txt')).read_text().strip().splitlines()[0]
                if variant=='zero':
                    visible=None
                    if not tnl:
                        occ=np.loadtxt(path/'full_occlusion.txt',delimiter=',').reshape(-1)[:len(frames)]
                        oov=np.loadtxt(path/'out_of_view.txt',delimiter=',').reshape(-1)[:len(frames)]
                        visible=(occ==0)&(oov==0)
                    smoke_sequences.append(Sequence(row['name'],[str(p) for p in frames],
                        'tnl2k' if tnl else 'lasot',gt,language=language,target_visible=visible))
                info=dict(init_bbox=gt[0].tolist(),language=language)
                for tracker in (native,candidate):
                    tracker.max_score=0;tracker.out_dict=None;tracker.pred_box_net=None
                    tracker.initialize(read(frames[0]),copy.deepcopy(info))
                for frame in frames[1:]:
                    image=read(frame);a=native.track(image)['target_bbox'];b=candidate.track(image)['target_bbox']
                    np.testing.assert_array_equal(a,candidate.state);assert torch.equal(native.prompt,candidate.prompt)
                    if variant=='zero':np.testing.assert_array_equal(a,b)
                    else:changes+=int(not np.array_equal(a,b))
                    assert np.isfinite(b).all() and min(b[2:])>0;checked+=1
            if variant!='zero':assert changes>0
            results[variant]=dict(frames=checked,output_changes=changes,state_and_prompt_exact=True,trained_head_loaded=True)
        finally:candidate.close();del native,candidate,wrapper;torch.cuda.empty_cache()
    del net,actor,optimizer,loaders;torch.cuda.empty_cache()
    wrapper=JointCenterEvaluationTracker('tnl2k',dest/'zero.pt',93198,epoch=1)
    wrapper.results_dir=str(dest/'runner_results')
    run_dataset(smoke_sequences,[wrapper],debug=0,threads=8,num_gpus=1)
    count=verify_outputs(smoke_sequences,wrapper,1,require_all=True)
    for seq in smoke_sequences:
        report=dest/(seq.dataset+'_synthetic_metric_check')
        run_official_metrics([wrapper],[seq],seq.dataset+'_NLBBOX_001',report)
        assert 'Norm Precision' in (report/'metrics.txt').read_text()
    report=dict(checks=results,resume_roundtrip=True,optimizer_state_device_correct=True,
        synthetic_formal_test_and_resume_rejected=True,optimizer_steps=0,formal_training_started=False,
        eight_worker_sequences=count,synthetic_checkpoint_bytes=checkpoint.stat().st_size,directory=str(dest))
    (dest/'check_report.json').write_text(json.dumps(report,indent=2))
    print('JOINT_CENTER_IO_PASS',json.dumps(report),flush=True)

if __name__=='__main__':main()
