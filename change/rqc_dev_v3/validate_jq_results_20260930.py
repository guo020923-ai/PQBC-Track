"""Read-only validation before resuming / evaluating run 92931."""
import argparse,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');args=p.parse_args()
t=Tracker('uvltrack','baseline_base_head_rqc_jq_bs16_30ep_20260929','tnl2k',run_id=92931)
assert t.params.cfg.TEST.MODE=='NLBBOX' and t.params.cfg.TEST.EPOCH==30
assert t.params.cfg.MODEL.RQC.DESIGN=='candidate_quality_jq'
assert t.params.cfg.MODEL.RQC.ENABLED and not t.params.cfg.MODEL.HEAD.ERC.ENABLED
assert Path(t.params.checkpoint).is_file()
root=Path(t.results_dir)/'tnl2k_NLBBOX_030'
dataset=get_dataset('tnl2k');assert len(dataset)==700
valid=[];missing=[];invalid=[]
for s in dataset:
    path=root/(s.name+'.txt')
    if not path.exists():missing.append(s.name);continue
    try:
        a=np.loadtxt(str(path),delimiter='\t',ndmin=2)
        assert a.shape==(len(s.frames),4) and np.isfinite(a).all()
        valid.append(s.name)
    except Exception as e:invalid.append(dict(sequence=s.name,error=str(e)))
print(json.dumps(dict(valid=len(valid),missing=len(missing),invalid=invalid,results=str(root),checkpoint=t.params.checkpoint)),flush=True)
assert not invalid,'Invalid existing results: refuse silent skipping or overwrite'
if args.complete:assert not missing,'Evaluation requires 700 complete sequences'
