"""Read-only status for the center-safe fixed development experiment."""
import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
manifest=json.loads((BASE/'locked_refiner_protocol_20260930/dev_manifest.json').read_text())
domain={m['name']:m['domain'] for m in manifest}
r=json.loads((BASE/'center_safe_pilot_20260930/report.json').read_text())
for d in ('all','lasot_train','tnl2k_train'):
    group=[v for k,v in r['sequences'].items() if d=='all' or domain[k]==d]
    print('CENTER_CACHE',json.dumps(dict(domain=d,n=len(group),
        gain=float(np.mean([v['iou_gain'] for v in group])),harmful=float(np.mean([v['harmful_rate'] for v in group])))))
dest=BASE/'center_safe_output_only_20260930';count=0;frames=0;errors=[]
if dest.exists():
    for i in range(8):
        log=dest/f'shard_{i}.log'
        if log.exists():
            content=log.read_text()
            if 'Traceback (most recent call last)' in content:errors.append(str(log))
            for line in content.splitlines():
                if line.startswith('OUTPUT_SEQUENCE '):
                    row=json.loads(line.split(' ',1)[1]);count+=1;frames+=row['frames']
    print('CENTER_OUTPUT_STATUS',json.dumps(dict(sequences=count,total=20,frames=frames,errors=errors)))
    result=dest/'report.json'
    if result.exists():print('CENTER_OUTPUT_RESULT',json.dumps(json.loads(result.read_text())['summary']))
