"""Read-only official-source collection status, never launches training."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent;dest=BASE/'official_center_cache_20260930'
rows=[];errors=[]
for i in range(8):
    log=dest/f'shard_{i:02d}.log'
    if log.exists():
        content=log.read_text()
        if 'Traceback (most recent call last)' in content:errors.append(str(log))
    part=dest/f'shard_{i:02d}'
    for file in part.glob('sequence_*.pt'):rows.append(file)
print('OFFICIAL_COLLECTION_STATUS',json.dumps(dict(completed_sequences=len(rows),total_sequences=100,
    errors=errors,training_started=False)))
completed=dest/'completed.json'
if completed.exists():print('COMPLETE',completed.read_text())
