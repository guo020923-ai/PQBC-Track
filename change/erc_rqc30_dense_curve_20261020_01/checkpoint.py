"""Strict new semantic identity. Old ERC/locator/diagnostic weights are refused."""
import hashlib
from pathlib import Path
import torch
from .model import MODEL_CLASS,MODEL_PARAMETERS
from .geometry import action_lattice

KIND='rqc30_original_TRAIN_dense_spatial_curve_ERC_20261020_01'
SOURCE_SHA='1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd'
SPLIT_SHA='49cf9ecf63c9f6a1a89731a659f208f38ce162de325660a332e416eaed9ef10b'


def semantics():
    return dict(kind=KIND,model_class=MODEL_CLASS,parameters=MODEL_PARAMETERS,source_sha256=SOURCE_SHA,
        split_sha256=SPLIT_SHA,roi_channels=951,roi_side=16,roi_span=1.5,summary=1152,
        channel_order=[768,128,32,18,2,2,1],template_bank='FG64_BG64_row_major_8x8_cosine',
        coefficients=5,utility='dTx-xTQx; Q=LLT; KEEP=0',actions=25,
        losses='fixed24_squared_conservative_gain_risk',coefficient_teacher='none',
        coefficient_parameterization='initial_Jacobian_balanced_Cholesky_v1',
        action_unit=.05,gain_unit=.01,duplicate_actions='retained_with_nominal_weight',
        empty_roi='raw5_zero_without_support_change',activation='exact_GELU_explicit_erf',
        native_history_feedback=False,
        training_authorized=False,empirical_gain_proven=False)


def code_identity():
    here=Path(__file__).resolve().parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob('*.py'))}


def checkpoint_identity():
    return dict(semantics=semantics(),code_sha256=code_identity())


def validate_payload(payload,model,expected_identity):
    if not isinstance(payload,dict):raise ValueError('Expected checkpoint dictionary')
    if expected_identity!=checkpoint_identity():
        raise ValueError('Expected identity does not match this model/source/code')
    if payload.get('kind')!=KIND or payload.get('identity')!=expected_identity:
        raise ValueError('Wrong ERC semantic/source/code identity')
    if not isinstance(payload,dict) or type(payload.get('epoch')) is not int or payload.get('diagnostic_only',False) or payload['epoch']<=0:
        raise ValueError('Diagnostic/untrained payload cannot be used as trained ERC')
    state=payload.get('model',{});expected=model.state_dict()
    if not isinstance(state,dict) or state.keys()!=expected.keys() or any(not torch.is_tensor(state[k]) or state[k].shape!=expected[k].shape or state[k].dtype!=expected[k].dtype or state[k].layout!=expected[k].layout for k in state):
        raise ValueError('Wrong checkpoint tensor schema')
    if not all(torch.isfinite(v).all() for v in state.values()) or not torch.equal(state['actions'].cpu(),action_lattice()):
        raise ValueError('Nonfinite state/wrong action order')
    model.load_state_dict(state,strict=True)
