"""DR4: train the actual hard decision, both before and after the gate.

Uses the existing DR3 response-only network. No candidate-box/IoU supervision,
dataset-specific inference rules, or extra candidates. DR4a optionally uses
detached GT center/extent for spatial targets; box IoU remains diagnostic only.
Discrete masks/targets are detached;
gradients flow through the scores of the current hard competitors.
"""
import torch
from torch.nn import functional as F

from .erc_target_reference_realized_action_dense_loss import _actual_action_oracle
from .erc_target_reference_trajectory_safe_dense_loss import _domain_mean


def masked_peak(score, mask):
    # Empty masks must not introduce +/-inf into masked losses/backprop.
    peak = score.masked_fill(~mask, torch.finfo(score.dtype).min).max(-1).values
    return torch.where(mask.any(-1), peak, torch.zeros_like(peak))


def compute_dr4_loss(pred, gt_heatmap, cfg, num_search_frames=1, dataset_ids=None,
                     gt_bbox=None):
    if cfg.MODEL.HEAD.ERC.DR2_ENABLE_MEMORY:
        raise ValueError('DR4 requires stateless gates in BOTH training and inference')
    raw = pred['erc_dr3_raw_correction'].float().flatten(1)
    if cfg.MODEL.HEAD.ERC.DR4_EXACT_DEPLOY_SCORE:
        raw = raw.double()
    gate = pred['erc_dr3_current_gate'].float().reshape(-1)
    active = pred['erc_active_mask'].reshape(-1).bool()
    support = pred['erc_dr1_support_mask'].bool().flatten(1)
    target = gt_heatmap.detach().to(raw).flatten(1).clamp_min(0)
    target = target / target.amax(-1, keepdim=True).clamp_min(1e-6)
    target_valid = target.amax(-1).gt(0)
    continuous = cfg.TRAIN.ERC_DR4A_CONTINUOUS_TARGET
    if continuous:
        if gt_bbox is None:
            raise ValueError('DR4a continuous supervision needs training GT boxes')
        from .erc_dr4a_continuous_supervision import spatial_quality
        height, width = pred['erc_dr3_raw_correction'].shape[-2:]
        target_map, target_valid = spatial_quality(gt_bbox, height, width, raw.dtype)
        target = target_map.flatten(1)
    base = pred['erc_windowed_base_score'].detach().to(raw).flatten(1)
    base = base.clamp_min(torch.finfo(raw.dtype).tiny).log()
    base = base.masked_fill(~support, torch.finfo(raw.dtype).min)
    valid = active & support.any(-1) & target_valid
    zero = (raw.sum() + gate.sum()) * 0
    row = torch.arange(raw.shape[0], device=raw.device)
    bi = base.argmax(-1)
    quality0 = target[row, bi]
    margin = float(cfg.TRAIN.ERC_DR4_MARGIN)
    min_gain = float(cfg.TRAIN.ERC_DR4_MIN_GAIN)
    if margin <= 0 or min_gain <= 0:
        raise ValueError('DR4 margin and minimum response gain must be positive')

    # Prefer an actually competing BETTER response, not necessarily one fixed
    # centre cell or the average/logsumexp of a large safe region.
    better = support & target.gt(quality0[:, None] + min_gain)
    other = support & ~better
    initial_gap = masked_peak(base, other) - masked_peak(base, better)
    # Necessary amplitude bound only, NOT an oracle guarantee. Do not demand
    # corrections outside the existing residual's pairwise 2*MAX_ALPHA span.
    eligible = valid & better.any(-1) & other.any(-1)
    eligible &= (initial_gap + margin).lt(2 * float(cfg.MODEL.HEAD.ERC.MAX_ALPHA))
    proposal = base + raw
    deployed = base + gate[:, None] * raw
    if cfg.MODEL.HEAD.ERC.DR4_EXACT_DEPLOY_SCORE:
        # Match the model's float32 exp followed by float64 score multiplication.
        # Do not recompute exp(q*raw) in a different precision inside the loss.
        deployed = base + pred['erc_factor'].reshape_as(raw).double().log()
    proposal_gap = masked_peak(proposal, other) - masked_peak(proposal, better)
    deployed_gap = masked_peak(deployed, other) - masked_peak(deployed, better)
    proposal_element = F.relu(margin + proposal_gap)
    deployed_element = F.relu(margin + deployed_gap)

    # Current proposal supplies gate targets; discrete oracle is not a gradient
    # path. No temporal blend can turn a nonzero target into an ineffective q.
    with torch.no_grad():
        gate_target, actionable, _, _, _, actions = _actual_action_oracle(
            base, raw, target, cfg.TRAIN.ERC_DR3_ACTION_LEVELS, valid, min_gain)
        gate_supervised = valid & actions.ne(actions[0:1]).any(0)
    gate_element = F.smooth_l1_loss(gate, gate_target, reduction='none')

    # Preserve ordering against genuinely worse response locations. At zero
    # correction this loss is zero, including baseline ties; no soft
    # distribution penalty opposes every small beneficial score movement.
    # Use the SAME quality tolerance for correction and preservation in DR4a.
    worse = support & target.lt(quality0[:, None] - (min_gain if continuous else 1e-8))
    nonworse = support & ~worse
    preserve_margin = (masked_peak(base, nonworse) - masked_peak(base, worse))
    preserve_margin = preserve_margin.clamp(min=0, max=margin).detach()
    noharm_element = F.relu(preserve_margin + masked_peak(deployed, worse)
                           - masked_peak(deployed, nonworse))
    noharm_mask = valid & worse.any(-1) & nonworse.any(-1)
    reduce = lambda x, mask: _domain_mean(
        x, mask, dataset_ids, cfg.TRAIN.ERC_DR3_DOMAIN_WORST_WEIGHT, zero)
    losses = {
        'proposal': reduce(proposal_element, eligible),
        'deploy': reduce(deployed_element, eligible),
        'gate': reduce(gate_element, gate_supervised),
        'noharm': reduce(noharm_element, noharm_mask),
        'minimal': reduce(raw.abs().mean(-1), valid),
    }
    total = sum(float(getattr(cfg.TRAIN, 'ERC_DR4_' + k.upper() + '_WEIGHT')) * v
                for k, v in losses.items())
    with torch.no_grad():
        index = deployed.argmax(-1)
        gain = target[row, index] - quality0
        changed = valid & index.ne(bi)
        beneficial = changed & gain.gt(1e-8)
        harmful = changed & gain.lt(-1e-8)
        supervised_preserve = gate_supervised & ~actionable
        def rate(mask):
            return mask.float().sum() / valid.float().sum().clamp_min(1)
        status = {'Loss/erc_total': total.detach()}
        status['ERC/dr4a_continuous_target'] = raw.new_tensor(float(continuous))
        status['ERC/dr4_target_baseline_quality_sum'] = quality0[valid].sum()
        status['ERC/dr4_noharm_supervised_count'] = noharm_mask.sum().float()
        status.update({'Loss/erc_dr4_' + k: v.detach() for k, v in losses.items()})
        for key, mask in [('valid', valid), ('potential', eligible),
                          ('actionable', actionable), ('changed', changed),
                          ('beneficial', beneficial), ('harmful', harmful),
                          ('preserve_supervised', supervised_preserve)]:
            status['ERC/dr4_' + key + '_count'] = mask.float().sum()
            status['ERC/dr4_' + key + '_rate'] = rate(mask)
        status['ERC/dr4_response_gain_sum'] = gain[valid].sum()
        status['ERC/dr4_response_gain_per_valid'] = gain[valid].sum() / valid.sum().clamp_min(1)
        if continuous:
            # Legacy response_gain keys remain for log compatibility; this
            # explicit name prevents interpreting spatial quality as IoU.
            status['ERC/dr4a_spatial_quality_gain'] = gain[valid].sum() / valid.sum().clamp_min(1)
        status['ERC/dr4_changed_gain_sum'] = gain[changed].sum()
        status['ERC/dr4_gate_actionable_sum'] = gate[actionable].sum()
        status['ERC/dr4_gate_preserve_sum'] = gate[supervised_preserve].sum()
        status['ERC/dr4_gate_mean'] = gate[valid].mean() if valid.any() else zero.detach()
        for name, mask in [('potential', eligible), ('actionable', actionable)]:
            status['ERC/dr4_' + name + '_deploy_gap_sum'] = deployed_gap[mask].sum()
        status['ERC/dr4_margin_satisfied_count'] = (eligible & deployed_gap.le(-margin)).float().sum()
        status['ERC/dr4_raw_abs'] = raw[valid].abs().mean() if valid.any() else zero.detach()
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for domain in ids[valid].unique():
                mask = valid & ids.eq(domain)
                prefix = 'ERC/dr4_domain_' + str(int(domain))
                status[prefix + '_count'] = mask.float().sum()
                status[prefix + '_gain_sum'] = gain[mask].sum()
                status[prefix + '_response_gain'] = gain[mask].mean()
    if gt_bbox is not None:
        from .erc_dr4_box_diagnostics import deployed_box_diagnostics
        status.update(deployed_box_diagnostics(pred, gt_bbox, gt_heatmap, valid, index))
    return total, status
