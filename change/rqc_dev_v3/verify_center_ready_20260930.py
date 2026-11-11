"""Read-only final readiness check; never builds a model or updates weights."""
import json,math,sys,subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from center_formal_contract_20260930 import NAME,file_hash

def main():
    folder=Path(__file__).parent
    report=json.loads((folder/'center_formal_ready_check_20260930_v3.json').read_text())
    identity=report['identity']
    expected='146599daa765c3f14f67d433a5c5b19501f7d0af583fb6e09ebefd123757e6c2'
    assert identity['source_checkpoint_sha256']==expected==file_hash(identity['source_checkpoint'])
    assert identity['config']==NAME and identity['config_sha256']==file_hash(ROOT/'experiments/uvltrack'/(NAME+'.yaml'))
    for name,digest in identity['source_files'].items():assert file_hash(folder/name)==digest,name
    current=yaml.safe_load((ROOT/'experiments/uvltrack'/(NAME+'.yaml')).read_text())
    prior=yaml.safe_load((ROOT/'experiments/uvltrack/baseline_base_head_rqc_jq_bs16_30ep_20260929.yaml').read_text())
    assert current['DATA']==prior['DATA']
    assert report['optimizer_steps']==0 and not report['weights_saved'] and report['frozen_buffers_unchanged']
    assert identity['head_trainable']==identity['backbone_trainable']==0 and identity['refiner_trainable']==19202
    for row in report['checks']:
        assert all(math.isfinite(v) for k,v in row.items() if k!='loader')
    assert len([r for r in report['checks'] if r['loader']=='train' and r['gradient']>0])==2
    output=ROOT/'change/center_safe_training_20260930'/NAME
    assert not output.exists(),'Formal output directory unexpectedly exists'
    assert 'CENTER_RUNNER_CHECK_PASS' in (folder/'center_runner_eval_ready_20260930_v2.log').read_text()
    assert 'EVALUATION_COMPLETE' not in (folder/'center_formal_ready_check_20260930_v3.log').read_text()
    for name in ('train_center_formal_20260930.py','test_center_formal_20260930.py','evaluate_center_formal_20260930.py'):
        result=subprocess.run([sys.executable,str(folder/name),'--help'],capture_output=True,text=True)
        assert result.returncode==0,(name,result.stderr)
    print('CENTER_FORMAL_READY',json.dumps(dict(config=NAME,source_verified=True,data_unchanged=True,
        code_fingerprints_match=True,train_validation_finite=True,trainable=19202,
        single_gpu=True,formal_training_started=False,optimizer_steps=0,
        output_directory=str(output),next='Await user confirmation; no training started')),flush=True)

if __name__=='__main__':main()
