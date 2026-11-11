"""Read-only compact progress and sequence-level pilot report."""
import json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'change/rqc_dev_v3';dest=base/'locked_refiner_parallel_20260930'
manifest=[]
for role in ('fit','dev'):
    manifest+=json.loads((base/'locked_refiner_protocol_20260930'/(role+'_manifest.json')).read_text())
total=sum(r['frames']-1 for r in manifest);processed=0;done=0;errors=[]
for i in range(8):
    path=dest/f'shard_{i:02d}.log';frames={}
    if not path.exists():continue
    content=path.read_text()
    if 'Traceback (most recent call last)' in content:errors.append(str(path))
    assigned=manifest[i::8]
    for line in content.splitlines():
        match=re.match(r'FRAME (\d+) (\d+) ',line)
        if match:frames[int(match[1])]=int(match[2])
        match=re.match(r'SEQUENCE_DONE (\d+) ',line)
        if match:
            idx=int(match[1])-1;frames[idx]=assigned[idx]['frames']-1;done+=1
    processed+=sum(frames.values())
print('PROGRESS',json.dumps(dict(sequences_done=done,total_sequences=len(manifest),frames_at_last_log=processed,total_frames=total,percent=round(100*processed/total,2),errors=errors)))
launch=base/'launch_locked_refiner_parallel_20260930.log'
print('LAUNCH_TAIL',launch.read_text()[-1800:] if launch.exists() else 'missing')
report=dest/'pilot/report.json'
if report.exists():
    results=json.loads(report.read_text());domains={r['name']:r['domain'] for r in manifest if r['role']=='dev'}
    for variant,obj in results.items():
        output={}
        for domain in ('all','lasot_train','tnl2k_train'):
            seq=[v for k,v in obj['sequences'].items() if domain=='all' or domains[k]==domain]
            gains=np.array([s['iou_gain'] for s in seq]);rng=np.random.RandomState(20260930)
            boot=gains[rng.randint(len(gains),size=(5000,len(gains)))].mean(1)
            output[domain]=dict(n=len(seq),macro_gain=float(gains.mean()),positive_sequences=int((gains>0).sum()),
                bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                harmful_macro=float(np.mean([s['harmful_rate'] for s in seq])),
                severe_harmful_macro=float(np.mean([s['severe_harmful_rate'] for s in seq])))
        print('RESULT',variant,json.dumps(output))
