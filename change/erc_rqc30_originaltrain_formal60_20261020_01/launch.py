"""Single full60 launch; no evaluation dispatch, no retry or architecture loop."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import rqc30_train_contract as c

TRAIN_LOG = c.ROOT / 'terminal_logs/train_erc_rqc30_originaltrain_formal60_20261020_01.log'
CONTROL_LOG = c.ROOT / 'terminal_logs/launch_erc_rqc30_originaltrain_formal60_20261020_01.log'


def env():
    value = os.environ.copy()
    value.update(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        value.pop(key, None)
    return value


def main():
    assert sys.argv[1:] in (['--dispatch'], ['--control'], ['--engineer'])
    os.chdir(c.ROOT)
    if sys.argv[1] == '--engineer':
        c.idle();c.verify_implementation()
        for name in ('engineering_controller_started_01.json','engineering_started_01.json','preflight_01.json','engineering_controller_exit_01.json'):
            assert not (c.HERE/name).exists(),'Engineering already attempted'
        policy=json.loads((c.HERE/'training_policy_01.json').read_text())
        auth=json.loads((c.HERE/'authorization_01.json').read_text())
        assert auth['maximum_engineering_launches'] == 1 and policy['engineering_wallclock_limit_seconds']==1545
        log_path=c.ROOT/'terminal_logs/engineer_erc_rqc30_originaltrain_formal60_20261020_01.log'
        assert not log_path.exists()
        start=time.monotonic()
        with log_path.open('xb') as log:
            child=subprocess.Popen([c.PYTHON,'-u',str(c.HERE/'runner.py'),'--mode','engineer'],
                cwd=str(c.ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env())
        original=c.process(child.pid)
        assert original and original['cwd']==str(c.ROOT), 'Cannot establish own engineering child identity'
        c.write_new(c.HERE/'engineering_controller_started_01.json',dict(time=time.time(),main=original,
            controller=c.process(os.getpid()),budget_seconds=1545))
        timed_out=False
        while child.poll() is None:
            if time.monotonic()-start>=1545:
                current=c.process(child.pid)
                if current and original:
                    assert all(current[k]==original[k] for k in ('pid','tick','cwd','cmdline'))
                    os.kill(child.pid,signal.SIGTERM)
                timed_out=True;break
            time.sleep(2)
        try:code=child.wait(timeout=60)
        except subprocess.TimeoutExpired:code=None
        record=dict(time=time.time(),exit_code=code,timed_out=timed_out,
            seconds=time.monotonic()-start,original=original,preflight_exists=(c.HERE/'preflight_01.json').exists())
        c.write_new(c.HERE/'engineering_controller_exit_01.json',record)
        print('BALANCED_ENGINEERING_EXIT',json.dumps(record),flush=True)
        assert code==0 and not timed_out and record['preflight_exists'],'Engineering did not pass; no training'
        return
    policy = c.verify()
    identity = c.binding()
    if sys.argv[1] == '--dispatch':
        c.idle()
        for p in (c.HERE / 'launch_once.lock', c.HERE / 'launch_record_01.json', c.HERE / 'train_started_01.json',
                  c.HERE / 'train60_01', TRAIN_LOG, CONTROL_LOG):
            assert not p.exists(), 'Already launched or preexisting output: ' + str(p)
        assert not list(c.HERE.glob('*failed_01.json'))
        engineered=json.loads((c.HERE/'engineering_controller_exit_01.json').read_text())
        assert engineered['exit_code']==0 and not engineered['timed_out']
        preflight = json.loads((c.HERE / 'preflight_01.json').read_text())
        assert preflight['engineering_pass'] and preflight['optimizer_steps'] == 0 and preflight['identity'] == identity
        c.write_new(c.HERE / 'launch_once.lock', dict(maximum_launches=1, time=time.time(),
                    authorization_sha256=c.digest(c.HERE / 'authorization_01.json'), epochs=60, automatic_retry=False))
        with CONTROL_LOG.open('xb') as log:
            child = subprocess.Popen([c.PYTHON, '-u', str(c.HERE / 'launch.py'), '--control'], cwd=str(c.ROOT),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env(), start_new_session=True)
        record = dict(dispatched_at=time.time(), controller=c.process(child.pid), training_identity=identity,
                      preflight_sha256=c.digest(c.HERE / 'preflight_01.json'),
                      log=str(TRAIN_LOG), controller_log=str(CONTROL_LOG), epochs=60,
                      old_ERC_loaded=False, automatic_test=False, automatic_retrain=False)
        c.write_new(c.HERE / 'launch_record_01.json', record)
        print('BALANCED_DISPATCHED', json.dumps(record), flush=True)
        return
    assert (c.HERE / 'launch_once.lock').exists() and not (c.HERE / 'train_started_01.json').exists()
    start = time.monotonic()
    with TRAIN_LOG.open('xb') as log:
        child = subprocess.Popen([c.PYTHON, '-u', str(c.HERE / 'runner.py'), '--mode', 'train'],
                    cwd=str(c.ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env())
    original = c.process(child.pid)
    assert original and original['cwd']==str(c.ROOT), 'Cannot establish own training child identity'
    c.write_new(c.HERE / 'train_started_01.json', dict(time=time.time(), main=original,
                controller=c.process(os.getpid()), training_identity=identity, command=child.args))
    timed_out = False
    while child.poll() is None:
        if time.monotonic() - start >= policy['controller_wallclock_limit_seconds']:
            current = c.process(child.pid)
            assert current and original and all(current[k] == original[k] for k in ('pid', 'tick', 'cwd', 'cmdline'))
            os.kill(child.pid, signal.SIGTERM)
            timed_out = True
            break
        time.sleep(5)
    if timed_out:
        try:
            code = child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            code = None  # No SIGKILL or blanket worker kill.
    else:
        code = child.wait()
    c.write_new(c.HERE / 'controller_exit_01.json', dict(time=time.time(), exit_code=code,
                original_main=original, wallclock_seconds=time.monotonic() - start, timed_out=timed_out,
                completed=(c.HERE / 'train60_01/completed_formal60.json').exists(), automatic_retry=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        c.write_new(c.HERE / 'launch_failed_01.json', dict(error=repr(error), time=time.time(), automatic_retry=False))
        raise
