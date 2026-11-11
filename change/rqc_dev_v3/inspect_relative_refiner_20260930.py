"""Read-only sequence-paired comparison; no fitting or test-set access."""
import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
r=json.loads((BASE/'relative_refiner_pilot_20260930/report.json').read_text())
manifest=json.loads((BASE/'locked_refiner_protocol_20260930/dev_manifest.json').read_text())
domain={m['name']:m['domain'] for m in manifest}
for d in ('all','lasot_train','tnl2k_train'):
    names=[n for n in r['point']['sequences'] if d=='all' or domain[n]==d]
    rows={v:np.array([r[v]['sequences'][n]['iou_gain'] for n in names]) for v in ('point','relative')}
    delta=rows['relative']-rows['point'];rng=np.random.RandomState(42)
    boot=delta[rng.randint(len(delta),size=(5000,len(delta)))].mean(1)
    print('CACHED_COMPARISON',json.dumps(dict(domain=d,n=len(names),
        point_macro=float(rows['point'].mean()),relative_macro=float(rows['relative'].mean()),
        paired_improvement=float(delta.mean()),improved_sequences=int((delta>0).sum()),
        paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        harmful_macro={v:float(np.mean([r[v]['sequences'][n]['harmful_rate'] for n in names])) for v in rows})))
dest=BASE/'relative_output_only_20260930'
if dest.exists():
    reports=[];errors=[]
    for i in range(8):
        log=dest/f'shard_{i}.log'
        if log.exists() and 'Traceback (most recent call last)' in log.read_text():errors.append(str(log))
        report=dest/f'shard_{i}/report.json'
        if report.exists():reports.extend(json.loads(report.read_text())['results'])
    print('OUTPUT_STATUS',json.dumps(dict(completed_sequences=len(reports),total_sequences=20,errors=errors)))
    final=dest/'report.json'
    if final.exists():
        new=json.loads(final.read_text());old=json.loads((BASE/'output_only_local_20260930/report.json').read_text())
        print('OUTPUT_RESULT',json.dumps(new['summary']))
        oldrows={x['sequence']:x for x in old['rows']}
        for d in ('all','lasot_train','tnl2k_train'):
            selected=[x for x in new['rows'] if d=='all' or x['domain']==d]
            dif=np.array([x['gain']-oldrows[x['sequence']]['gain'] for x in selected])
            rng=np.random.RandomState(42);boot=dif[rng.randint(len(dif),size=(5000,len(dif)))].mean(1)
            print('OUTPUT_PAIRED',json.dumps(dict(domain=d,improvement=float(dif.mean()),
                improved_sequences=int((dif>0).sum()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())))
