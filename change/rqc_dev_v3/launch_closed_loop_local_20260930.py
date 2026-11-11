import os,sys,json,time,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'change/rqc_dev_v3';dest=base/'closed_loop_local_20260930';dest.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
processes=[];handles=[]
try:
    for i in range(8):
        log=(dest/f'shard_{i}.log').open('x');handles.append(log)
        processes.append(subprocess.Popen([sys.executable,'-u',str(base/'closed_loop_local_20260930.py'),
            '--shard',str(i),'--output',str(dest/f'shard_{i}')],stdout=log,stderr=subprocess.STDOUT,env=env))
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Closed-loop worker failed')
        time.sleep(5)
    assert all(p.returncode==0 for p in processes)
finally:
    for p in processes:
        if p.poll() is None:p.terminate()
    for h in handles:h.close()
rows=[]
for i in range(8):rows+=json.loads((dest/f'shard_{i}/report.json').read_text())['results']
assert len(rows)==len(set(r['sequence'] for r in rows))==20
summary={}
for domain in ('all','lasot_train','tnl2k_train'):
    group=[r for r in rows if domain=='all' or r['domain']==domain];gain=np.array([r['gain'] for r in group]);rng=np.random.RandomState(42)
    boot=gain[rng.randint(len(gain),size=(5000,len(gain)))].mean(1)
    summary[domain]=dict(n=len(group),macro_gain=float(gain.mean()),positive_sequences=int((gain>0).sum()),
        bootstrap95=np.quantile(boot,[.025,.975]).tolist(),severe_macro=float(np.mean([r['severe'] for r in group])))
(dest/'report.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
print('CLOSED_ALL_COMPLETE',json.dumps(summary),flush=True)
