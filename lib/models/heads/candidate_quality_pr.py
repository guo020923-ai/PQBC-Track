"""PR: native-response prior plus centered residual, final-score supervision.

Different from JC absolute IoU fitting and R1 auxiliary utility regression.
No fixed-blend test tuning. Scores are ranking logits, not calibrated IoU.
"""
import torch
from .candidate_quality_jc import CandidateQualityJC, candidates, select_candidate
from .candidate_quality_pd import training_loss as final_margin_loss


def prior_residual_scores(prior, raw):
    # Native candidates arrive sorted. Keep precision for close response values.
    p = prior.detach().double().clamp_min(1e-30)
    log_prior = p.log() - p[:, :1].log()
    residual = raw.double() - raw[:, :1].double()
    return log_prior + residual, log_prior, residual


class CandidateQualityPR(CandidateQualityJC):
    def forward(self, batch, use_reliability=False):
        raw = super().forward(batch, use_reliability)['residual']
        final, prior, residual = prior_residual_scores(batch['scores'],raw)
        return dict(decision_score=final,log_prior=prior,residual=residual)


def training_loss(pred, iou_target, temperature=.1):
    # Critically, supervise the exact prior+residual deployed at test time.
    loss,parts=final_margin_loss(pred,iou_target,temperature)
    parts.pop('saturation_rate')  # Unbounded logits: .95 is not saturation.
    score=pred['decision_score'].detach()
    winner=score.argmax(-1,keepdim=True)
    parts.update(residual_abs=pred['residual'].detach().abs().mean(),
                 prior_abs=pred['log_prior'].abs().mean(),
                 selected_prior_gap=-pred['log_prior'].gather(1,winner).mean(),
                 selected_residual=pred['residual'].detach().gather(1,winner).mean())
    return loss.float(),parts
