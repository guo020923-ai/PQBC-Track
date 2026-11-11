"""Production target-reference conditioned ERC (TRC-v1).

Separate deployment-consistent evidence; original Head losses stay native.
D compares spatially normalized branch distributions, not geometric means.
No candidate boxes/IoU/current-search GT enter inference.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA = 'erc-trc-production-v1'
CHANNELS = ('log_c', 'log_t_deploy', 'target_cosine', 'distractor_cosine',
            'background_cosine', 'template_cosine', 'hann', 'normalized_d')


def deploy_contrastive_logits(raw_logits):
    """Official SOFTMAX_ONE inference: [target, max(dis,bg,0), 0].

The third output is a ZERO logit, NOT the raw background logit. This auxiliary
definition is shared at train/test; the official two-logit training loss is
not replaced. Exactly three UNCOLLAPSED prototype logits must be supplied.
"""
    if raw_logits.ndim != 3 or raw_logits.shape[-1] != 3:
        raise ValueError('Expected raw [B,N,3] target/distractor/background logits')
    zero = torch.zeros_like(raw_logits[..., :1])
    negative = torch.cat([raw_logits[..., 1:], zero], -1).max(-1, keepdim=True).values
    return torch.cat([raw_logits[..., :1], negative, zero], -1)


def normalized_branch_difference(log_c, log_t, d_max=4.0):
    """Log ratio of spatial probability distributions; scale-invariant.

    Arithmetic mass normalization does not let many near-zero background
    probabilities determine the offset as a geometric-mean subtraction does.
    Uses true deployment log-probabilities, not epsilon-floored T.
    """
    if d_max <= 0:
        raise ValueError('d_max must be positive')
    log_pc = log_c - torch.logsumexp(log_c.flatten(1), -1)[:, None, None]
    log_pt = log_t - torch.logsumexp(log_t.flatten(1), -1)[:, None, None]
    raw = log_pc - log_pt
    return raw.clamp(-d_max, d_max), raw


def build_evidence(c, search, prompts, template, template_mask, logit_scale,
                   flag, hann, eps=1e-6, d_max=4.0):
    """Read-only evidence extraction; no current-search label or bbox argument.

The template IMAGE/target mask is the known reference. Template TOKENS are
from the current joint backbone pass, not an immutable independent encoding.
The prototype cosine triplet is retained BEFORE max/softmax compression.
"""
    if c.ndim != 3 or not torch.isfinite(c).all() or (c < 0).any() or (c > 1).any():
        raise ValueError('C must be finite [B,H,W] probabilities')
    batch, height, width = c.shape
    if search.ndim != 3 or search.shape[:2] != (batch, height * width):
        raise ValueError('Search token/response shape mismatch')
    if prompts.shape != (batch, 3, search.shape[-1]):
        raise ValueError('Expected three uncompressed prototype embeddings')
    if template.ndim != 3 or template.shape[0] != batch or template.shape[-1] != search.shape[-1]:
        raise ValueError('Template token shape mismatch')
    if template_mask.shape != template.shape[:2] or template_mask.dtype != torch.bool:
        raise ValueError('Expected known template target mask [B,Ntemplate], bool')
    if flag.numel() != batch or not ((flag >= 0) & (flag <= 2)).all():
        raise ValueError('Explicit official flag 0/1/2 required for each sample')
    active = flag.reshape(batch).eq(2)
    counts = template_mask.sum(-1)
    if (active & counts.eq(0)).any():
        raise ValueError('Active VL tracking requires a non-empty target reference')
    if hann.shape != (height, width) or not torch.isfinite(hann).all() or (hann < 0).any():
        raise ValueError('Hann shape/value mismatch')
    if not all(torch.isfinite(x).all() for x in (search, prompts, template, logit_scale)):
        raise ValueError('Nonfinite evidence')
    search_unit = F.normalize(search, dim=-1)
    cosine = search_unit @ F.normalize(prompts, dim=-1).transpose(-2, -1)
    raw_logits = logit_scale.exp() * cosine
    if not torch.isfinite(raw_logits).all():
        raise ValueError('Nonfinite scaled prototype logits')
    logits = deploy_contrastive_logits(raw_logits)
    t = logits.softmax(-1)[..., 0].reshape(batch, height, width)
    reference = (template * template_mask[..., None]).sum(1) / counts.clamp_min(1)[:, None]
    e_template = (search_unit * F.normalize(reference, dim=-1)[:, None]).sum(-1)
    e_template = e_template.reshape(batch, height, width)
    log_c = c.float().clamp_min(torch.finfo(torch.float32).tiny).log()
    log_t = logits.float().log_softmax(-1)[..., 0].reshape(batch, height, width)
    d, d_unclipped = normalized_branch_difference(log_c, log_t, d_max)
    h = hann.to(c).expand(batch, -1, -1)
    channels = torch.cat([
        (log_c.clamp(-8., 0.) / 8.)[:, None],
        (log_t.clamp(-16., 0.) / 16.)[:, None],
        cosine.transpose(1, 2).reshape(batch, 3, height, width),
        e_template[:, None], h[:, None], (d / d_max)[:, None],
    ], 1)
    return {
        'channels': channels.detach(), 'c': c, 't': t, 'd': d, 'hann': h,
        'log_c': log_c, 'log_t': log_t, 'd_unclipped': d_unclipped,
        'raw_logits': raw_logits, 'deploy_logits': logits,
        'template_cosine': e_template, 'prototype_cosine': cosine,
        'active': active,
        'clipping': {
            'c_epsilon_floor': (c < eps).float().mean().detach(),
            't_epsilon_floor': (t < eps).float().mean().detach(),
            'log_c_clip': (log_c < -8.).float().mean().detach(),
            'log_t_clip': (log_t < -16.).float().mean().detach(),
            'd_clip': (d_unclipped.abs() > d_max).float().mean().detach(),
        },
    }


class TargetReferenceERCPrototype(nn.Module):
    """Small global+local reliability estimator, not a permission/veto gate.

Zero final projection => rC=rT=0.5 => alpha=0 => factor=1. Zero final
weights delay encoder gradients until after the first update, not forever.
"""
    def __init__(self, hidden=32, alpha_max=0.5, reliability_temperature=1.0):
        super().__init__()
        if not 0 < alpha_max or not 0 < reliability_temperature:
            raise ValueError('Positive coefficient/temperature required')
        self.alpha_max = float(alpha_max)
        self.temperature = float(reliability_temperature)
        self.encoder = nn.Sequential(
            nn.Conv2d(len(CHANNELS), hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.reliability = nn.Conv2d(2 * hidden, 2, 1)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)

    @property
    def reachable_alpha_supremum(self):
        # Sigmoid reliabilities have difference in (-1,1), not all real values.
        return self.alpha_max * math.tanh(1. / self.temperature)

    def forward(self, evidence, mode='FULL_ERC'):
        if mode not in ('FULL_ERC', 'ERC_OFF'):
            raise ValueError('Unsupported mode')
        features = self.encoder(evidence['channels'].detach())
        global_features = features.mean((-2, -1), keepdim=True).expand_as(features)
        logits = self.reliability(torch.cat([features, global_features], 1))
        reliabilities = logits.sigmoid()
        alpha = self.alpha_max * torch.tanh(
            (reliabilities[:, 0] - reliabilities[:, 1]) / self.temperature)
        active = evidence['active'][:, None, None]
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))
        correction = alpha * evidence['d']
        factor = correction.exp()
        if mode == 'ERC_OFF':
            # Explicit bypass. Production integration must multiply this into
            # native S0, NEVER recompute OFF through an auxiliary normalizer.
            alpha = torch.zeros_like(alpha)
            correction = torch.zeros_like(correction)
            factor = torch.ones_like(factor)
        c, t, h = evidence['c'], evidence['t'], evidence['hann']
        # Hann-zero positions remain zero in the actual score.
        base = (c * h) * t
        # Separate logs avoid clamping tiny H*C*T to one shared floor.
        log_score = evidence['log_c'] + evidence['log_t']
        log_score = log_score + h.clamp_min(1e-30).log() + correction
        return {'reliability_logits': logits, 'reliability': reliabilities,
                'alpha': alpha, 'factor': factor, 'base': base,
                'score': base * factor, 'log_score': log_score,
                'support': h > 0, 'active': evidence['active']}


@torch.no_grad()
def local_rank_reliability_targets(log_c, log_t, relevance, support, active, tolerance=1e-6):
    """Comparable spatial labels: weighted fraction of correctly ordered pairs.

Y is the existing GT CENTER heatmap, not candidate IoU. At each location i,
compare branch b's response ordering with Y ordering against locations j.
Weight by |Yi-Yj|, ignore tied relevance/Hann-zero pairs, assign score ties
0.5. The SAME definition is applied separately to C and deploy-T. No current
GT is broadcast as a frame-level label. Empty/constant Y => no supervision.
"""
    if log_c.shape != log_t.shape or log_c.shape != relevance.shape or log_c.shape != support.shape:
        raise ValueError('Spatial supervision shapes must match')
    if not torch.isfinite(relevance).all() or (relevance < 0).any():
        raise ValueError('Invalid relevance heatmap')
    batch, height, width = log_c.shape
    y = relevance.detach().flatten(1)
    difference = y[:, :, None] - y[:, None, :]
    supported = support.flatten(1).bool()
    pairs = supported[:, :, None] & supported[:, None, :] & (difference.abs() > tolerance)
    pairs &= active.reshape(batch, 1, 1)
    weight = difference.abs() * pairs
    denominator = weight.sum(-1)
    valid = denominator > 0
    labels = []
    for response in (log_c, log_t):
        value = response.detach().flatten(1)
        signed_margin = (value[:, :, None] - value[:, None, :]) * difference.sign()
        correct = (signed_margin > tolerance).to(value)
        correct += 0.5 * (signed_margin.abs() <= tolerance).to(value)
        target = (weight * correct).sum(-1) / denominator.clamp_min(1e-12)
        labels.append(torch.where(valid, target, torch.full_like(target, .5)))
    # Avoid overwhelming a small positive response region with background cells.
    positive, negative = valid & (y > 0), valid & (y <= 0)
    class_count = positive.any(-1).float() + negative.any(-1).float()
    point_weight = torch.zeros_like(y)
    for group in (positive, negative):
        point_weight += group / group.sum(-1, keepdim=True).clamp_min(1)
    point_weight /= class_count.clamp_min(1)[:, None]
    return {'targets': torch.stack(labels, 1).reshape(batch, 2, height, width),
            'valid': valid.reshape(batch, height, width),
            'point_weight': point_weight.reshape(batch, height, width),
            'pair_weight': weight, 'relevance_difference': difference}


def trc_losses(output, labels):
    """Two terms only: comparable reliability BCE + deploy-score pair ranking.

Unit weights are reference defaults, not a claim of empirically optimal
hyperparameters. No hard argmax gradient, IoU label, permission or alpha L2.
"""
    point = F.binary_cross_entropy_with_logits(
        output['reliability_logits'], labels['targets'], reduction='none').mean(1)
    per_frame = (point * labels['point_weight']).flatten(1).sum(-1)
    valid_frames = labels['valid'].flatten(1).any(-1)
    count = valid_frames.sum().clamp_min(1)
    reliability = (per_frame * valid_frames).sum() / count
    log_score = output['log_score'].flatten(1)
    pair_margin = log_score[:, :, None] - log_score[:, None, :]
    weight = labels['pair_weight'] * (labels['relevance_difference'] > 0)
    ranking = (F.softplus(-pair_margin) * weight).sum((-2, -1))
    ranking /= weight.sum((-2, -1)).clamp_min(1e-12)
    fusion = (ranking * valid_frames).sum() / count
    return {'reliability': reliability, 'fusion': fusion, 'total': reliability + fusion}



class TargetReferenceERC(TargetReferenceERCPrototype):
    """Native Head adapter. Generic distillation fields retain ORIGINAL train T."""
    DESIGN = 'target_reference_erc_trc_v1'

    def __init__(self, feat_sz, hidden=32, alpha_max=.5, temperature=1.,
                 d_max=4., inference_mode='FULL_ERC', force_off=False):
        super().__init__(hidden, alpha_max, temperature)
        self.design = self.DESIGN
        self.feat_sz, self.d_max = int(feat_sz), float(d_max)
        self.force_off = bool(force_off)
        self.set_inference_mode(inference_mode)
        window = torch.hann_window(feat_sz, periodic=False)
        self.register_buffer('hann_window', torch.outer(window, window).reshape(1, -1))

    def set_inference_mode(self, mode):
        mode = {'OFF': 'ERC_OFF', 'FULL': 'FULL_ERC'}.get(str(mode).upper(), str(mode).upper())
        if mode not in ('ERC_OFF', 'FULL_ERC'):
            raise ValueError('TRC supports only ERC_OFF/FULL_ERC')
        self.inference_mode = mode

    def forward(self, c, native_t, *, search, prompts, template, template_mask,
                logit_scale, flag):
        evidence = build_evidence(c, search, prompts, template, template_mask,
                                  logit_scale, flag,
                                  self.hann_window.reshape(self.feat_sz, self.feat_sz),
                                  d_max=self.d_max)
        mode = 'ERC_OFF' if self.force_off else self.inference_mode
        packet = super().forward(evidence, mode)
        packet.update({'evidence': evidence, 'alpha_limit': self.reachable_alpha_supremum})
        batch = c.shape[0]
        c_flat, native_t_flat = c.flatten(1), native_t.flatten(1)
        native_base = c_flat * native_t_flat
        native_windowed = native_base * self.hann_window
        zero = c.new_zeros(batch, 1)
        # These native generic fields are used by existing Head distillation.
        # Auxiliary deployment T is isolated in packet['evidence']['t'].
        result = {
            'features': evidence['channels'], 'cls_response': c_flat,
            'cont_response': native_t_flat,
            'windowed_cls_score': c_flat * self.hann_window,
            'windowed_cont_score': native_t_flat * self.hann_window,
            'base_score': native_base, 'windowed_base_score': native_windowed,
            'calibrated_score': (evidence['c'] * evidence['t'] * packet['factor']).flatten(1),
            'windowed_calibrated_score': packet['score'].flatten(1),
            'factor': packet['factor'].flatten(1),
            'log_ratio': evidence['d'].flatten(1),
            'alpha': packet['alpha'].flatten(1), 'raw_alpha': packet['alpha'].flatten(1),
            'z_alpha': packet['reliability_logits'][:, 0].flatten(1)
                       - packet['reliability_logits'][:, 1].flatten(1),
            'reliability': packet['reliability'].mean(1).flatten(1),
            'reliability_cls': packet['reliability'][:, 0].flatten(1),
            'reliability_cont': packet['reliability'][:, 1].flatten(1),
            'reliability_base': zero, 'reliability_gate': zero,
            'intervention_gate': zero, 'delta': zero,
            'active_mask': evidence['active'], 'trc_output': packet,
        }
        return result
