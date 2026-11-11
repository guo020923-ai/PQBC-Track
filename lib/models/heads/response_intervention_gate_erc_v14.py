"""ERC-v14 with an independent response-level intervention gate.

ERC-v13 uses the magnitude of the signed branch coefficient ``beta`` both to
decide whether a frame should be changed and to decide which evidence branch
should be trusted.  The paired closed-loop audit showed that those two jobs
must be separated.  ERC-v14 therefore keeps the v13 response-only branch and
regional predictors, but applies their correction through a learned scalar
gate ``q``:

    S_ERC(x) = S0(x) * exp(q * beta * G(x) * D(x)).

``q`` predicts whether the current response pair has a reachable correction;
``beta`` keeps only the branch direction and strength.  The module still sees
only response evidence and the Hann prior.  It consumes no boxes, geometry,
IoU targets, or search features.
"""

import math

import torch
from torch import nn

from .response_branch_spatial_gate_erc_v13 import (
    ResponseBranchSpatialGateERCV13,
)


class ResponseInterventionGateERCV14(ResponseBranchSpatialGateERCV13):
    DESIGN = 'response_intervention_gate_v14'

    def __init__(self, *args, intervention_prior=0.20, **kwargs):
        super().__init__(*args, **kwargs)
        prior = float(intervention_prior)
        if not 0.0 < prior < 1.0:
            raise ValueError('ERC-v14 intervention_prior must be in (0, 1)')
        self.design = self.DESIGN
        self.intervention_prior = prior
        self.intervention_head = nn.Linear(
            self.beta_head.in_features, 1
        )
        nn.init.zeros_(self.intervention_head.weight)
        nn.init.constant_(
            self.intervention_head.bias, math.log(prior / (1.0 - prior))
        )

    def forward(self, cls_response, cont_response, flag=None):
        output = super().forward(cls_response, cont_response, flag=flag)
        gate_logit = self.intervention_head(
            output['v13_decision_features']
        )
        predicted_gate = gate_logit.sigmoid()
        enabled_mask = output['active_mask'] & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC'
        )
        effective_gate = torch.where(
            enabled_mask[:, None], predicted_gate,
            torch.zeros_like(predicted_gate),
        )

        # Recompute the whole calibrated path.  This deliberately does not use
        # v13's already-applied correction, so q is the single and exact switch
        # between the official response and the branch-spatial correction.
        predicted_beta = output['v13_beta']
        effective_beta = effective_gate * predicted_beta
        correction_map = (
            effective_beta[:, :, None]
            * output['v13_spatial_gate']
            * output['v13_d_map']
        )
        factor = correction_map.exp().flatten(1)
        base_score = output['base_score']
        windowed_base_score = output['windowed_base_score']
        calibrated_score = base_score * factor
        windowed_calibrated_score = windowed_base_score * factor

        output.update({
            'features': output['v13_decision_features'],
            'alpha': correction_map.flatten(1),
            'raw_alpha': predicted_beta,
            'z_alpha': output['v13_beta_logit'],
            'reliability': effective_gate,
            'reliability_gate': effective_gate,
            'intervention_gate': effective_gate,
            'gate_logit': gate_logit,
            'gate_decision_features': output['v13_decision_features'],
            'delta': correction_map.flatten(1),
            'factor': factor,
            'calibrated_score': calibrated_score,
            'windowed_calibrated_score': windowed_calibrated_score,
            'global_calibrated_score': base_score,
            'windowed_global_calibrated_score': windowed_base_score,
            'alpha_global': effective_beta,
            'alpha_region': correction_map,
            'alpha_final': correction_map,
            'v13_effective_beta': effective_beta,
            'v13_correction_map': correction_map,
            'v14_gate_logit': gate_logit,
            'v14_intervention_gate': predicted_gate,
            'v14_effective_gate': effective_gate,
            'v14_effective_beta': effective_beta,
        })
        if self.training:
            # Teacher-forcing predictions for every possible response center
            # are computed inside the model forward.  The actor can then
            # gather the detached Oracle center without invoking model
            # parameters outside DDP's forward graph.
            spatial_features = output['v13_spatial_features']
            encoder_input = output['v13_encoder_input']
            batch_size, _, height, width = spatial_features.shape
            candidate_count = height * width
            local_spatial = spatial_features.flatten(2).transpose(1, 2)
            local_evidence = encoder_input.flatten(2).transpose(1, 2)
            global_features = output['v13_global_features'][:, None, :].expand(
                -1, candidate_count, -1
            )
            anchor_features = self.decision_projection(torch.cat([
                global_features, local_spatial, local_evidence
            ], dim=-1))
            output['v14_anchor_beta_logit'] = self.beta_head(
                anchor_features
            ).squeeze(-1)
            output['v14_anchor_sigma_logits'] = self.sigma_head(
                anchor_features
            )
        return output
