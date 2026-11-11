"""Unique new run binding; never shadow the legacy module named common."""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
paper = importlib.import_module('change.erc_rqc30_originaltrain_model_20261020_01')
pm = importlib.import_module(paper.__name__ + '.model')
pg = importlib.import_module(paper.__name__ + '.geometry')
po = importlib.import_module(paper.__name__ + '.objective')
pc = importlib.import_module(paper.__name__ + '.checkpoint')
pr = importlib.import_module(paper.__name__ + '.runtime')
pp = importlib.import_module(paper.__name__ + '.provider')
sys.path.insert(0, str(HERE))
import source30_data as infrastructure

Model = pm.BalancedSecondOrderEvidenceERC
Pools = infrastructure.Pools
FitPools = infrastructure.FitPools30
Stream = infrastructure.FrameStream
GraphSampler = infrastructure.GraphSampler
collate = infrastructure.collate
rebuild = infrastructure.rebuild
source_snapshot = infrastructure.source_snapshot
assert_unchanged = infrastructure.assert_unchanged
SOURCE = infrastructure.SOURCE
SPLIT = HERE / 'data_policy_01.json'
KIND = 'formal60_RQC30_original_TRAIN_mean_gain_risk_ERC_20261020_01'
PYTHON = os.environ.get('PYTHON_BIN', sys.executable)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, tick=fields[19], state=fields[0], ppid=int(fields[1]),
                    cwd=os.readlink(p / 'cwd'), cmdline=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def idle():
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    assert not gpu.strip(), 'GPU busy; no takeover'
    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent > 1 and parent not in ancestors:
        ancestors.add(parent)
        q = process(parent)
        if not q:
            break
        parent = q['ppid']
    for p in Path('/proc').iterdir():
        if p.name.isdigit() and int(p.name) not in ancestors:
            q = process(int(p.name))
            if q and 'python' in q['cmdline'] and q['cwd'].startswith(str(ROOT)):
                raise RuntimeError('Project Python task still active: ' + repr(q))


def settings():
    spec, cfg = infrastructure.settings()
    assert spec['NATURAL_INPUT']['WARMUP_FRAMES'] == 16 and spec['NATURAL_INPUT']['CLIP_FRAMES'] == 192
    plan = json.loads(SPLIT.read_text())
    assert plan['TRAIN_only'] and plan['source_sha256'] == pc.SOURCE_SHA
    assert plan['training_membership']=='original_loader_train_no_ERC_holdout'
    assert all(not plan[k] for k in ('blocked','monitor','confirmation','diagnostic'))
    return spec, cfg, plan, infrastructure.parameters()


def dependencies():
    # Bind every local module already imported by this actual source pipeline.
    paths = set()
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if str(p).startswith(str(ROOT) + '/') and p.suffix == '.py' and p.parent != HERE:
                paths.add(p)
    paths.add(infrastructure.CONFIG)
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def own_code():
    return {p.name: digest(p) for p in sorted(HERE.glob('*.py'))}


def binding():
    semantics = dict(pc.semantics())
    # Old definition-level flags are not run authorization or efficacy evidence.
    semantics.pop('training_authorized')
    semantics.pop('empirical_gain_proven')
    return dict(kind=KIND, semantics=semantics, implementation=pc.checkpoint_identity(),
                run_code=own_code(), dependencies_sha256=digest(HERE / 'dependency_manifest_01.json'),
                policy_sha256=digest(HERE / 'training_policy_01.json'),
                authorization_sha256=digest(HERE / 'authorization_01.json'),
                split_sha256=pc.SPLIT_SHA, source_sha256=pc.SOURCE_SHA,
                fresh_ERC=True, old_ERC_loaded=False, full60_authorized=True,
                native_history_feedback=False, empirical_gain_proven=False,
                engineering_plan_sha256=digest(HERE / 'engineering_plan_01.json'),
                CPU_report_sha256=digest(HERE / 'CPU_admission_01.json'),
                baseline_reference_acceptance_sha256=digest(HERE / 'baseline_reference_acceptance_01.json'),
                historical_baseline_provenance_limited=True,
                switch_budget_sha256=verify_switch_budget())


def verify_implementation():
    ready = json.loads((HERE / 'CPU_admission_01.json').read_text())
    assert ready['CPU_suite_pass'] and ready['optimizer_steps'] == 0 and ready['GPU_operations'] == 0
    prefix = 'change/erc_rqc30_originaltrain_model_20261020_01/'
    expected = {name[len(prefix):]: sha for name,sha in ready['source_sha256_by_file'].items() if name.startswith(prefix)}
    assert pc.code_identity() == expected
    assert pm.MODEL_PARAMETERS == 71881 and pc.semantics()['summary'] == 1014


def verify_split():
    new=json.loads(SPLIT.read_text())
    assert new['training_membership']=='original_loader_train_no_ERC_holdout'
    assert new['TRAIN_only'] and new['source_sha256']==pc.SOURCE_SHA
    assert all(not new[k] for k in ('blocked','monitor','confirmation','diagnostic'))
    assert not new['additional_ERC_sequence_exclusion'] and not new['original_test_split_changed']
    assert digest(SPLIT)==pc.SPLIT_SHA
    return new


def validate_source():
    audit = json.loads((HERE / 'reuse_audit_01.json').read_text())
    accepted = json.loads((HERE / 'baseline_reference_acceptance_01.json').read_text())
    assert audit['status'] == 'REUSE_AUDIT_BLOCKED'
    assert len(audit['issues']) == 1 and audit['issues'][0]['code'] == 'BASELINE_GENERATION_SOURCE_BINDING_MISSING'
    assert accepted['accepted_option'] == 'A' and accepted['historical_baseline_may_be_used_with_disclosed_generation_provenance_limit']
    assert not accepted['baseline_GPU_rerun_authorized']
    assert audit['RQC30_checkpoint_sha256'] == pc.SOURCE_SHA == digest(SOURCE)
    original = SOURCE.parent
    assert digest(original / 'protocol.json') == audit['RQC30_protocol_sha256']
    assert digest(original / 'stats.jsonl') == audit['RQC30_stats_sha256']
    protocol = json.loads((original / 'protocol.json').read_text())
    assert protocol['epochs'] == 30 and protocol['output_only']
    assert protocol['source_checkpoint_sha256'] == audit['official_E300_sha256']
    assert digest(protocol['source_checkpoint']) == audit['official_E300_sha256']
    assert digest(infrastructure.CONFIG) == protocol['config_sha256'] == audit['RQC30_config_sha256']
    for name, sha in protocol['source_files'].items():
        assert digest(ROOT / 'change/rqc_dev_v3' / name) == sha, 'Original RQC source changed: '+name
    for rel, record in audit['official_code'].items():
        assert digest(ROOT / rel) == record['isolated_sha256']
    assert all(v['all_frames_pass'] and v['finite_positive_integer_writer_pass'] for v in audit['datasets'].values())
    return protocol


def verify_switch_budget():
    ledger=json.loads((HERE/'authorization_01.json').read_text())
    prior=['erc_rqc30_secondorder_formal60_20261019_01',
           'erc_rqc30_gainrisk_formal60_20261019_01',
           'erc_rqc30_balanced_formal60_20261019_01']
    seconds=0.
    engineering=0.
    for name in prior:
        path=ROOT/'change'/name
        exit_record=json.loads((path/'controller_exit_01.json').read_text())
        assert exit_record['exit_code']==-15 and not exit_record['timed_out']
        assert not exit_record['completed']
        seconds+=exit_record['wallclock_seconds']
        er=json.loads((path/'engineering_controller_exit_01.json').read_text())
        assert er['exit_code']==0 and not er['timed_out']
        engineering+=er['seconds']
    assert abs(seconds-ledger['previous_training_seconds'])<1e-6
    assert abs(engineering-ledger['previous_engineering_seconds'])<1e-5
    assert seconds+ledger['training_seconds']<=86400
    assert engineering+ledger['engineering_seconds']<=1800
    for pid in (4435,4431,8923,8919,11337,11331):
        assert not process(pid),'Prior task active; do not replace'
    return digest(HERE/'authorization_01.json')


def verify():
    verify_implementation()
    verify_switch_budget()
    assert digest(SOURCE) == pc.SOURCE_SHA
    verify_split()
    validate_source()
    for rel, sha in json.loads((HERE / 'dependency_manifest_01.json').read_text()).items():
        assert digest(ROOT / rel) == sha, 'Frozen dependency changed: ' + rel
    auth = json.loads((HERE / 'authorization_01.json').read_text())
    policy = json.loads((HERE / 'training_policy_01.json').read_text())
    assert auth['training_epochs'] == policy['epochs'] == policy['scheduler_horizon'] == 60
    assert auth['maximum_training_launches'] == policy['maximum_launches'] == 1
    assert auth['no_automatic_repairs_or_retries'] and auth['new_authorization']
    assert auth['engineering_seconds'] == policy['engineering_wallclock_limit_seconds'] == 1545
    assert auth['training_seconds'] == policy['controller_wallclock_limit_seconds'] == 75930
    locked = json.loads((HERE / 'engineering_plan_01.json').read_text())
    assert locked['all_lists_fixed_before_forward'] and locked['model_image_forward_count'] == 0
    assert len(locked['selected']) == 3 and locked['observations'] == 72
    assert locked['source_sha256'] == pc.SOURCE_SHA and locked['split_sha256'] == pc.SPLIT_SHA
    return policy


def validate_checkpoint(payload, model, expected):
    assert payload['kind'] == KIND and payload['training_identity'] == expected == binding()
    assert 1 <= payload['epoch'] <= 60 and payload['optimizer_steps_total'] > 0
    canonical = dict(payload, kind=pc.KIND, identity=expected['implementation'])
    pc.validate_payload(canonical, model, expected['implementation'])


class Statistics:
    COUNTS = {'valid_samples', 'changed_saved_boxes', 'integer_improved_frames', 'integer_harmed_frames',
              'oracle_positive_frames', 'coefficient_identifiable', 'solver_not_converged', 'coefficient_numerically_invalid'}

    def __init__(self):
        self.total, self.weights, self.batches = {}, {}, 0

    def add(self, stats):
        n = int(stats['valid_samples'])
        if not n:
            return
        self.batches += 1
        for key, value in stats.items():
            if value is None:
                continue
            value = float(value)
            assert math.isfinite(value), key
            if key in self.COUNTS:
                self.total[key] = self.total.get(key, 0) + int(value)
            else:
                self.total[key] = self.total.get(key, 0.) + value * n
                self.weights[key] = self.weights.get(key, 0) + n

    def result(self):
        return dict({k: v / self.weights[k] if k in self.weights else v for k, v in self.total.items()}, batches=self.batches)


def inputs_from_rows(raw, fixed, hw, actions):
    assert set(raw) == {'boundary', 'valid', 'source_context', 'executed_delta'}
    target = pg.candidates(fixed, hw, actions)
    delta = (target[..., 2:] / fixed[:, None, 2:]).log()
    torch.testing.assert_close(raw['executed_delta'].double(), delta, rtol=1e-6, atol=1e-7)
    return raw


@torch.no_grad()
def metrics(pred, fixed, truth, hw, actions):
    t = pg.action_targets(fixed, truth, hw, actions)
    row = torch.arange(len(fixed), device=fixed.device)
    index = pred['index']
    gf, gz, g = (t[k][row, index] for k in ('float_gain', 'integer_gain', 'gain'))
    best = t['gain'].masked_fill(~pred['allowed'], -torch.inf).max(1).values
    boxes = t['boxes'][row, index]
    return dict(valid_samples=len(row), float_IoU_gain=float(gf.mean()), integer_IoU_gain=float(gz.mean()),
                conservative_gain=float(g.mean()), KEEP_rate=float(index.eq(0).double().mean()),
                changed_saved_boxes=int(boxes.long().ne(fixed.long()).any(1).sum()),
                integer_improved_frames=int(gz.gt(0).sum()), integer_harmed_frames=int(gz.lt(0).sum()),
                oracle_positive_frames=int(best.gt(0).sum()), oracle_conservative_gain=float(best.mean()),
                decision_regret=float((best - g).mean()))


def objective(model, inputs, fixed, truth, hw):
    pred = model(**inputs)
    loss, detail, boxes = po.objective(model, pred, inputs, fixed, truth, hw)
    assert detail['valid_samples'] == len(fixed)
    extra = metrics(pred, fixed, truth, hw, model.actions)
    for key in ('float_IoU_gain', 'integer_IoU_gain', 'conservative_gain'):
        assert abs(extra[key] - detail[key]) < 1e-12
    return loss, dict(detail, **extra), boxes, pred
