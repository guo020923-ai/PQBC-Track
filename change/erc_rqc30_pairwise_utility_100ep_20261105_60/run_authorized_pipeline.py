import json,os,subprocess,sys,time,traceback
from pathlib import Path
from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.run_authorized_pipeline import digest,write_new,identity,require_idle_gpu
PACKAGE='change.erc_rqc30_pairwise_utility_100ep_20261105_60'
def verify_manifest(project,package):
 m=json.loads((package/'implementation_manifest_60.json').read_text())
 if m.get('status')!='IMPLEMENTATION_SEALED_AND_AUTHORIZED':raise PermissionError('manifest')
 for r in m['files']:
  p=project/'change'/r['package']/r['name']
  if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('implementation '+str(p))
 return m
def audit_training(train,cache):
 import torch
 from .config import PROTOCOL
 from .model import VERSION
 done=json.loads((train/'completed.json').read_text());cks=sorted((train/'checkpoints').glob('ERC_epoch_*.pth.tar'));stats=[json.loads(x) for x in (train/'stats.jsonl').read_text().splitlines()]
 if done.get('optimizer_steps')!=187500 or [p.name for p in cks]!=[f'ERC_epoch_{e:04d}.pth.tar' for e in range(1,101)] or [x['epoch'] for x in stats]!=list(range(1,101)):raise ValueError('continuity')
 final=cks[-1];saved=torch.load(str(final),map_location='cpu')
 if saved.get('kind')!=VERSION or saved.get('protocol')!=PROTOCOL or saved.get('epoch')!=100 or saved.get('optimizer_steps')!=187500 or done.get('checkpoint_sha256')!=digest(final) or digest(cache/'manifest.json')!=PROTOCOL['cache_manifest_sha256']:raise ValueError('identity')
 return {'status':'COMPLETE_VERIFIED','checkpoint_count':100,'stats_count':100,'optimizer_steps':187500,'fixed_epoch':100,'final_checkpoint':str(final),'final_checkpoint_sha256':digest(final),'final_checkpoint_bytes':final.stat().st_size,'cache_manifest_sha256':PROTOCOL['cache_manifest_sha256'],'source_rqc_epoch':30,'source_rqc_checkpoint_sha256':PROTOCOL['source_rqc_checkpoint_sha256'],'checkpoint_selection':'fixed_epoch100_no_best','training_seconds':done['training_seconds'],'train_rows':done['train_rows'],'calibration_rows':done['calibration_rows'],'last_train_stats':stats[-1]['stats'],'audited_unix':time.time()}
def main(argv):
 if len(argv)!=4:raise SystemExit('usage')
 project,cache,stage=map(Path,argv[1:]);package=project/'change'/'erc_rqc30_pairwise_utility_100ep_20261105_60';a=json.loads((package/'authorization_60.json').read_text());ta=json.loads((package/'posttrain_test_authorization_60.json').read_text())
 if a.get('status')!='AUTHORIZED_ONE_PAIRWISE_100EP_TRAINING' or ta.get('status')!='AUTHORIZED_ONE_TEST_AFTER_PAIRWISE_TRAIN_CALIBRATION':raise PermissionError('authorization')
 manifest=verify_manifest(project,package)
 if stage.exists():raise FileExistsError(stage)
 require_idle_gpu();stage.mkdir();os.chdir(project);sys.path.insert(0,str(project));phase='LAUNCH';write_new(stage/'actual_launch.json',{'status':'ACTUAL_PAIRWISE_100EP_PIPELINE_LAUNCH','process':identity(os.getpid()),'started_unix':time.time(),'training_epochs':100,'optimizer_steps':187500,'fixed_checkpoint':'Epoch100','automatic_test_only_after_train_calibration':True,'persistent_workers':8,'automatic_retry':False,'parallel_gpu_job':False,'scientific_variable':'within_frame_pairwise_utility_ranking','implementation_manifest_sha256':digest(package/'implementation_manifest_60.json'),'sealed_file_count':len(manifest['files'])})
 try:
  phase='TRAINING';train=stage/'train';train.mkdir();cmd=[sys.executable,'-B','-m',PACKAGE+'.train_child',str(train),str(cache)];child=subprocess.Popen(cmd,cwd=str(project));write_new(stage/'training_launch.json',{'status':'TRAINING_CHILD_STARTED','process':identity(child.pid),'command':cmd,'started_unix':time.time()});rc=child.wait()
  if rc:raise RuntimeError('training '+str(rc))
  phase='TRAINING_AUDIT';audit=audit_training(train,cache);write_new(stage/'training_audit.json',audit);phase='NUMERIC_EXPORT';from .checkpoint import export_numeric
  numeric=stage/'numeric';nm=export_numeric(audit['final_checkpoint'],numeric);phase='TRAIN_PAIRWISE_CALIBRATION';from .calibration import calibrate
  cal=calibrate(numeric,cache);write_new(stage/'calibration_complete.json',cal);phase='TNL2K_FULL700_TEST';require_idle_gpu();write_new(stage/'test_launch.json',{'status':'AUTHORIZED_TNL2K_FULL700_DEVELOPMENT_TEST_STARTED','fixed_checkpoint_sha256':audit['final_checkpoint_sha256'],'numeric_npz_sha256':nm['numeric_npz_sha256'],'calibration_sha256':cal['calibration_sha256'],'threshold':cal['threshold'],'sequences':700,'frames':516038,'searches':515338,'persistent_workers':8,'same_forward_native_shadow':True,'started_unix':time.time(),'automatic_retry':False});from change.erc_rqc30_pairwise_utility_test_20261105_61.stage import run_stage
  result=run_stage(audit,numeric,stage/'test',processes=8);write_new(stage/'official_complete.json',result);write_new(stage/'terminal_status.json',{'status':'PIPELINE_COMPLETE_PENDING_CONTROLLER_OS0_EXIT','process':identity(os.getpid()),'ended_unix':time.time(),'expected_os_exit':0,'automatic_retry':False});print(json.dumps(result,sort_keys=True),flush=True)
 except BaseException as e:
  write_new(stage/'terminal_failure.json',{'status':'PAIRWISE_PIPELINE_FAILED','phase':phase,'error':repr(e),'traceback':traceback.format_exc(),'ended_unix':time.time(),'automatic_retry':False});raise
if __name__=='__main__':main(sys.argv)
