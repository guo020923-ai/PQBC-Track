"""Fixed24 mean-gain risk. Independently bound replacement experiment.

Unchanged output geometry/25-action support, direction + PSD curvature model,
source and per-observation min(float, integer) labels. Only supervision changes.
No tracker/model construction, optimizer, checkpoint loader, or launch entry.

For fixed supported action a, d E[L]/d score_a is proportional to
score_a - E[g_a/.01 | inputs, supported]. This is an unrestricted score-risk
property, not a guarantee for the constrained PSD model or real generalization.
Do not add oracle-choice/coefficient losses while claiming this property.
"""
import torch

OBJECTIVE_KIND = 'fixed24_squared_conditional_conservative_gain_v1'
TRAINING_AUTHORIZED = False
ACTION_COUNT = 25
GAIN_UNIT = .01


def loss_from_targets(prediction, executed_delta, utility):
    score, allowed = prediction['utility'], prediction['allowed']
    if score.ndim != 2 or score.shape[1] != ACTION_COUNT:
        raise ValueError('Expected Bx25 utility in gain/.01 units')
    if allowed.shape != score.shape or allowed.dtype != torch.bool:
        raise ValueError('Wrong action support')
    if executed_delta.shape != (*score.shape, 2):
        raise ValueError('Wrong actual-execution geometry')
    y = utility.detach().to(score)
    if y.shape != score.shape or not torch.isfinite(y).all() or not torch.isfinite(score).all():
        raise ValueError('Targets and scores must be finite Bx25 tensors')
    if not allowed[:,0].all() or not y[:,0].eq(0).all() or not score[:,0].eq(0).all():
        raise ValueError('KEEP must be allowed with exact zero target and score')
    if y.numel() and y.abs().max() > 1 / GAIN_UNIT + 1e-6:
        raise ValueError('IoU gain must lie in [-1,1], expressed in .01 units')
    if len(score) == 0:
        return score.sum()*0, dict(valid_samples=0, loss=0., gain_risk=0.)
    active = allowed.clone()
    active[:,0] = False
    # Fixed /24, not per-frame supported-count normalization: do not reweight
    # outcomes according to how many other actions happen to be supported.
    residual = torch.where(active, score-y, torch.zeros_like(score))
    loss = .5 * residual.square().sum(1).mean() / (ACTION_COUNT-1)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite squared gain risk')
    return loss, dict(loss=float(loss.detach()), gain_risk=float(loss.detach()))
