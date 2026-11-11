"""Strict new semantics; failed curve/locator/diagnostic weights are not reusable."""
import hashlib
import sys
from pathlib import Path
import torch
from .model import MODEL_CLASS, FeatureReuseCurriculumERC
from .geometry import action_lattice

KIND = 'RQC30_original_TRAIN_feature_reuse_curriculum_ERC_20261021_01'
SOURCE_SHA = '1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd'
TRAIN_SHA = '49cf9ecf63c9f6a1a89731a659f208f38ce162de325660a332e416eaed9ef10b'


def code_identity():
    # An in-memory CPU-only tester may supply the exact same byte manifest.
    loader = globals().get('__loader__')
    if hasattr(loader, 'source_manifest'):
        return dict(loader.source_manifest)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}


def semantics():
    # Do not perturb the caller's initialization RNG while enumerating parameters.
    with torch.random.fork_rng(devices=[]):
        count = FeatureReuseCurriculumERC().parameter_count
    return dict(kind=KIND, model=MODEL_CLASS, parameters=count, source_sha256=SOURCE_SHA,
        TRAIN_policy_sha256=TRAIN_SHA, source_grid=16, raw_RGB_encoder=False,
        additional_Backbone_forward=False, template='legal_initial_existing_FG_BG_tokens',
        evidence='ordered_4edge_2region_5point_search24_head8_similarity2_native_extent2',
        descriptor=384, gain_unit=.01, utility='0.01*(shared_score(candidate)-shared_score(KEEP))',
        gain_target='min(float_IoU_gain,official_integer_box_IoU_gain)',
        regression='24_supported_nonKEEP_squared_error_in_percentage_points_no_oracle_reweighting',
        boundary_GT_auxiliary=False, loss_weights=[1.], candidate_count=25, action_logwh_limit=.05,
        original_support=True, output_only=True, source_epoch=30,
        no_native_history_feedback=True, no_acceptance_threshold=True,
        boundary_evidence_used_at_deployment=True, pure_quadratic_curve=False,
        training_curriculum='epochs1_40_equal5_anchor_views_epochs41_60_natural_only',
        anchor_offsets_logwh=[[0.,0.],[-.05,0.],[.05,0.],[0.,-.05],[0.,.05]],
        model_input_GT=False, perturbation_center='natural_RQC_center_unchanged',
        training_authorized=False, empirical_gain_proven=False)


def identity():
    return dict(semantics=semantics(), code_sha256=code_identity())


def validate_payload(payload, model, expected):
    if expected != identity() or not isinstance(payload,dict):
        raise ValueError('Wrong implementation/source binding')
    if payload.get('kind') != KIND or payload.get('identity') != expected:
        raise ValueError('Wrong model/source/TRAIN/code identity; no partial loading')
    if type(payload.get('epoch')) is not int or not 1 <= payload['epoch'] <= 60 or payload.get('diagnostic_only',False):
        raise ValueError('Not a completed formal epoch')
    if type(payload.get('optimizer_steps_total')) is not int or payload['optimizer_steps_total'] <= 0:
        raise ValueError('Actual optimizer steps required')
    state, schema = payload.get('model',{}), model.state_dict()
    if not isinstance(state,dict) or state.keys() != schema.keys():
        raise ValueError('Wrong state keys')
    for key, value in state.items():
        if not torch.is_tensor(value) or value.shape != schema[key].shape or value.dtype != schema[key].dtype or value.layout != schema[key].layout or not torch.isfinite(value).all():
            raise ValueError('Wrong/nonfinite state: '+key)
    if not torch.equal(state['actions'].cpu(),action_lattice()):
        raise ValueError('Wrong action order')
    # No load is performed until the ENTIRE payload has passed validation.
    model.load_state_dict(state,strict=True)

