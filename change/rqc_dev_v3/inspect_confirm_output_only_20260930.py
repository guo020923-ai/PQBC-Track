"""Read-only status for the fixed development closed-loop run."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'change/rqc_dev_v3';dest=base/'confirm_output_only_20260930'
manifest=json.loads((base/'locked_refiner_protocol_20260930/final_manifest.json').read_text())
rows=[];errors=[]
for i in range(8):
    file=dest/f'shard_{i}.log'
    if not file.exists():continue
    content=file.read_text()
    if 'Traceback (most recent call last)' in content:errors.append(str(file))
    for line in content.splitlines():
        if line.startswith('OUTPUT_SEQUENCE '):rows.append(json.loads(line.split(' ',1)[1]))
print('CLOSED_STATUS',json.dumps(dict(completed_sequences=len(rows),total_sequences=len(manifest),
    completed_frames=sum(r['frames'] for r in rows),total_frames=sum(r['frames']-1 for r in manifest),errors=errors)))
launch=base/'launch_confirm_output_only_20260930.log'
if launch.exists():print('LAUNCH_TAIL',launch.read_text()[-3000:])
report=dest/'report.json'
if report.exists():print('RESULT',json.dumps(json.loads(report.read_text())['summary']))
