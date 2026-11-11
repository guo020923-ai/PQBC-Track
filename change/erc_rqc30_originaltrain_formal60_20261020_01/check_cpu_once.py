"""Once-only CPU admission: unchanged functional model plus actual full TRAIN membership."""
import contextlib,io,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
import rqc30_train_contract as c
import torch
from importlib import import_module
start=time.monotonic()
assert not (c.HERE/'CPU_started_01.json').exists()
c.write_new(c.HERE/'CPU_started_01.json',dict(time=time.time(),optimizer_steps=0,GPU_operations=0))
try:
    c.idle();c.verify_split();c.validate_source();c.verify_switch_budget()
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):
        import_module(c.paper.__name__+'.checks').main()
    cpu_stdout=capture.getvalue()
    with (c.HERE/'CPU_stdout_01.txt').open('x') as f:f.write(cpu_stdout)
    line=next(x for x in cpu_stdout.splitlines() if x.startswith('BALANCED_ERC_CPU_REPORT='))
    suite=json.loads(line.partition('=')[2])
    assert suite['CPU_suite_pass'] and suite['optimizer_steps']==suite['GPU_operations']==0
    spec,cfg,plan,params=c.settings()
    pools=c.FitPools(cfg,plan);original=c.Pools(cfg)
    old_plan=json.loads((c.ROOT/'change/erc_rqc30_balanced_formal60_20261019_01/split_plan_01.json').read_text())
    restored={}
    for i in pools.vl:
        names=pools.datasets[i].sequence_list
        assert names==original.datasets[i].sequence_list and not pools.blocked[i]
        blocked={r['sequence'] for r in old_plan['blocked'] if r['source']==i}
        assert blocked.issubset(set(names))
        domain=pools.datasets[i].get_name()
        restored[domain]=dict(count=len(names),restored=len(blocked),
             exact_original_membership=True,sequence_list=list(names))
    assert {k:v['count'] for k,v in restored.items()}==dict(lasot=1120,tnl2k=1299,otb99=51)
    try:c.FitPools(cfg,old_plan)
    except ValueError:pass
    else:raise AssertionError('Old held-out plan accepted')
    locked=json.loads((c.HERE/'engineering_plan_01.json').read_text())
    for row in locked['selected']:
        clip=c.rebuild(pools,row)
        assert all(clip['visible'][frame] for frame in row['observation_frames'])
    assert c.FitPools.choose_clip is c.Pools.choose_clip
    assert not torch.cuda.is_initialized()
    ready=dict(suite,source_sha256_by_file={str((c.pc.Path(c.pc.__file__).parent/n).relative_to(c.ROOT)):h
             for n,h in c.pc.code_identity().items()},
             semantics=c.pc.semantics(),actual_full_TRAIN_membership=restored,
             old_holdout_plan_rejected=True,new_training_started=False,
             model_image_forwards=0,seconds=time.monotonic()-start)
    c.write_new(c.HERE/'CPU_admission_01.json',ready)
    c.write_new(c.HERE/'CPU_exit_01.json',dict(exit_code=0,seconds=time.monotonic()-start,automatic_retry=False))
    c.verify_implementation()
    print('ORIGINAL_TRAIN_CPU_PASS='+json.dumps(dict(pass_all=True,seconds=time.monotonic()-start,
        restored={k:{a:b for a,b in v.items() if a!='sequence_list'} for k,v in restored.items()},
        optimizer_steps=0,GPU_operations=0,model_image_forwards=0)),flush=True)
except Exception as exc:
    c.write_new(c.HERE/'CPU_failed_01.json',dict(error=repr(exc),automatic_retry=False))
    raise
