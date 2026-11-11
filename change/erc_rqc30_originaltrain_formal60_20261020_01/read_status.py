"""Read only: no checker, launcher, model forward or mutation."""
import json,os
import rqc30_train_contract as c
os.chdir(c.ROOT)
c.verify()
records={n:json.loads((c.HERE/n).read_text()) if (c.HERE/n).exists() else None for n in
 ('launch_record_01.json','train_started_01.json','actual_optimizer_step_01.json',
  'controller_exit_01.json','train_failed_01.json','launch_failed_01.json','train60_01/completed_formal60.json')}
for name in ('launch_record_01.json','train_started_01.json','actual_optimizer_step_01.json'):
    if records[name]:assert records[name]['training_identity']==c.binding()
started=records['train_started_01.json'];processes={}
if started:
    for role in ('controller','main'):
        expected=started[role];current=c.process(expected['pid'])
        if current:assert all(current[k]==expected[k] for k in ('pid','tick','cwd','cmdline'))
        processes[role]=current
path=c.HERE/'train60_01/stats.jsonl'
stats=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
assert [r['epoch'] for r in stats]==list(range(1,len(stats)+1))
log=c.ROOT/'terminal_logs/train_erc_rqc30_originaltrain_formal60_20261020_01.log'
tail=log.read_text(errors='replace')[-16000:] if log.exists() else ''
latest=[s for s in tail.splitlines() if s.startswith('BALANCED_BATCH ')]
print('ORIGINAL_TRAIN_STATUS_JSON='+json.dumps(dict(records=records,processes=processes,
    completed_epochs=len(stats),recent_stats=stats[-3:],latest_batch=latest[-1:])),flush=True)
