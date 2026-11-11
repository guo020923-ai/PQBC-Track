"""Eight single-GPU collection workers, verified merge, then fixed pilot only."""
import sys,os,time,json,subprocess
from pathlib import Path
import torch
from prepare_locked_refiner_protocol_20260930 import load_role
ROOT=Path(__file__).resolve().parents[2]


def main():
    os.chdir(str(ROOT));torch.set_num_threads(1)
    base=ROOT/'change/rqc_dev_v3';protocol=base/'locked_refiner_protocol_20260930'
    manifest=load_role(protocol,'fit')+load_role(protocol,'dev')
    dest=base/'locked_refiner_parallel_20260930';dest.mkdir(exist_ok=False)
    worker=base/'run_locked_refiner_parallel_20260930.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    running=[];handles=[];start=time.time()
    try:
        for i in range(8):
            log=(dest/f'shard_{i:02d}.log').open('x');handles.append(log)
            cmd=[sys.executable,'-u',str(worker),'collect','--protocol',str(protocol),
                 '--output',str(dest/f'shard_{i:02d}'),'--shard-index',str(i),'--shard-count','8']
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
            running.append(process)
        (dest/'pids.json').write_text(json.dumps([p.pid for p in running]))
        print('SHARDS_STARTED',json.dumps([p.pid for p in running]),flush=True)
        while any(p.poll() is None for p in running):
            failed=[(i,p.returncode) for i,p in enumerate(running) if p.poll() not in (None,0)]
            if failed:raise RuntimeError('Shard failure '+str(failed))
            time.sleep(5)
        assert all(p.returncode==0 for p in running)
    finally:
        for p in running:
            if p.poll() is None:p.terminate()
        for h in handles:h.close()
    allrows=[];records=[];identities=[]
    for i in range(8):
        part=dest/f'shard_{i:02d}'
        records.append(json.loads((part/'completed.json').read_text()))
        cache=torch.load(str(part/'paired_cache.pt'),map_location='cpu')
        identity=cache['identity'];identities.append(identity)
        assert identity['manifest']==manifest[i::8] and identity['shard_index']==i and identity['shard_count']==8
        assert not identity['smoke']
        allrows.extend(cache['rows'])
    for key in ('checkpoint_sha256','script_sha256','feature_source_hashes','tracker','samples_per_sequence'):
        assert all(x[key]==identities[0][key] for x in identities)
    assert len({(r['sequence'],r['frame']) for r in allrows})==len(allrows)
    assert {(r['sequence'],r['role']) for r in allrows}=={(r['name'],r['role']) for r in manifest}
    order={r['name']:i for i,r in enumerate(manifest)}
    allrows.sort(key=lambda r:(order[r['sequence']],r['frame']))
    # Compare every completed serial shard, if available, before admitting merge.
    serial=base/'locked_refiner_cache_20260930';lookup={(r['sequence'],r['frame']):r for r in allrows};compared=0
    for file in sorted(serial.glob('sequence_*.pt')):
        for old in torch.load(str(file),map_location='cpu')['rows']:
            row=lookup[(old['sequence'],old['frame'])]
            for key in ('features','boundary_samples','boundary_valid','box','gt','origin'):
                assert torch.equal(row[key],old[key]),'Serial/parallel mismatch: '+key
            assert row['side']==old['side'];compared+=1
    cachepath=dest/'paired_cache.pt'
    torch.save(dict(rows=allrows,identity=dict(identities[0],manifest=manifest,shard_index=None,shard_count=8)),cachepath)
    summary=dict(sequences=len(manifest),samples=len(allrows),tracked_frames=sum(r['tracked_frames'] for r in records),
        collection_seconds=time.time()-start,serial_exact_compared=compared,final_holdout_loaded=False)
    (dest/'completed.json').write_text(json.dumps(summary,indent=2));print('MERGE_COMPLETE',json.dumps(summary),flush=True)
    with (dest/'train.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(base/'run_locked_refiner_20260930.py'),'train',
            '--protocol',str(protocol),'--cache',str(cachepath),'--output',str(dest/'pilot')],
            check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
    print('PILOT_COMPLETE',str(dest/'pilot/report.json'),flush=True)


if __name__=='__main__':main()
