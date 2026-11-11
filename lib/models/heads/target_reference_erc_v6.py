"""TRC-v6: one deployment-aligned utility policy for KEEP/C/T.

The previous v5g/v5h policy combined three independently supervised decisions:
a global gate, a C/T direction classifier, and a branch-specific veto.  TRC-v6
instead predicts the response-level utility of the two reachable corrections.
KEEP has a fixed reference utility.  Gate, direction, and intervention strength
are therefore derived from one action distribution and the same straight-
through path is used by training and deployment.

This module remains an ERC: it only reads response/reference evidence and only
calibrates C/T fusion.  It does not read candidate boxes or predict IoU.
"""

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.heads.target_reference_erc import build_evidence
from lib.models.heads.target_reference_erc_v5 import TargetReferenceERCV5


class TargetReferenceERCV6(TargetReferenceERCV5):
    DESIGN = 'target_reference_erc_trc_v6_joint_action_utility'

    def __init__(self, feat_sz, hidden=32, alpha_max=0.5, temperature=1.0,
                 d_max=4.0, inference_mode='FULL_ERC', force_off=False,
                 utility_threshold=0.05):
        if utility_threshold <= 0:
            raise ValueError('utility_threshold must be positive')
        super().__init__(
            feat_sz, hidden, alpha_max, temperature, d_max,
            inference_mode, force_off,
        )
        decision_dim = self.permission_head[0].in_features
        # Remove the obsolete independently trained Gate/direction/strength
        # heads.  This also prevents unused parameters entering the optimizer.
        del self.permission_head
        del self.direction_head
        del self.strength_head
        self.utility_head = nn.Sequential(
            nn.Linear(decision_dim, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )
        self.branch_strength_head = nn.Sequential(
            nn.Linear(decision_dim, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )
        for head in (self.utility_head, self.branch_strength_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.utility_threshold = float(utility_threshold)
        self.design = self.DESIGN

    def forward(self, c, native_t, *, search, prompts, template, template_mask,
                logit_scale, flag):
        evidence = build_evidence(
            c, search, prompts, template, template_mask, logit_scale, flag,
            self.hann_window.reshape(self.feat_sz, self.feat_sz),
            d_max=self.d_max,
        )
        features = self.encoder(evidence['channels'].detach())
        response_statistics = self._response_statistics(evidence)
        decision_features = torch.cat([
            self._learned_statistics(features), response_statistics,
        ], -1)

        branch_utility = self.utility_head(decision_features)
        keep_utility = branch_utility.new_full(
            (branch_utility.shape[0], 1), self.utility_threshold
        )
        action_logits = torch.cat([keep_utility, branch_utility], -1)
        action_probability = (action_logits / self.temperature).softmax(-1)
        hard_action = action_logits.argmax(-1)
        hard_policy = F.one_hot(hard_action, 3).to(action_probability)
        action_policy = (
            hard_policy + action_probability - action_probability.detach()
            if self.training else hard_policy
        )

        branch_strength_logit = self.branch_strength_head(decision_features)
        branch_strength = branch_strength_logit.sigmoid()
        signed_strength = torch.stack([
            branch_strength.new_zeros(branch_strength.shape[0]),
            branch_strength[:, 0], -branch_strength[:, 1],
        ], -1)
        alpha_scalar = self.alpha_max * (
            action_policy * signed_strength
        ).sum(-1)
        soft_alpha_scalar = self.alpha_max * (
            action_probability * signed_strength
        ).sum(-1)
        active = evidence['active']
        alpha_scalar = torch.where(
            active, alpha_scalar, torch.zeros_like(alpha_scalar)
        )
        soft_alpha_scalar = torch.where(
            active, soft_alpha_scalar, torch.zeros_like(soft_alpha_scalar)
        )
        alpha = alpha_scalar[:, None, None].expand_as(evidence['d'])
        soft_alpha = soft_alpha_scalar[:, None, None].expand_as(alpha)
        correction = alpha * evidence['d']
        factor = correction.exp()
        mode = 'ERC_OFF' if self.force_off else self.inference_mode
        if mode == 'ERC_OFF':
            alpha = torch.zeros_like(alpha)
            correction = torch.zeros_like(correction)
            factor = torch.ones_like(factor)

        base = (evidence['c'] * evidence['hann']) * evidence['t']
        log_base = (
            evidence['log_c'] + evidence['log_t']
            + evidence['hann'].clamp_min(1e-30).log()
        )
        score = base * factor
        selected_strength = torch.where(
            hard_action.eq(1), branch_strength[:, 0],
            torch.where(
                hard_action.eq(2), branch_strength[:, 1],
                branch_strength.new_zeros(branch_strength.shape[0]),
            ),
        )
        hard_permission = hard_action.ne(0) & active
        direction_probability = action_probability[:, 1:]
        direction_probability = direction_probability / (
            direction_probability.sum(-1, keepdim=True).clamp_min(1e-12)
        )
        packet = {
            'features': features,
            'response_statistics': response_statistics,
            'decision_features': decision_features,
            'branch_utility': branch_utility,
            'utility_threshold': self.utility_threshold,
            'action_logits': action_logits,
            'action_probability': action_probability,
            'hard_action': torch.where(
                active, hard_action, torch.zeros_like(hard_action)
            ),
            'hard_permission': hard_permission.to(branch_utility),
            'permission_probability': 1.0 - action_probability[:, 0],
            'permission_threshold': self.utility_threshold,
            'permission_temperature': self.temperature,
            'direction_probability': direction_probability,
            'hard_direction_index': (hard_action - 1).clamp(0, 1),
            'branch_strength_logit': branch_strength_logit,
            'branch_strength': branch_strength,
            'strength': selected_strength,
            'alpha_scalar': alpha_scalar,
            'soft_alpha_scalar': soft_alpha_scalar,
            'alpha': alpha,
            'soft_alpha': soft_alpha,
            'factor': factor,
            'base': base,
            'log_base': log_base,
            'score': score,
            'log_score': log_base + correction,
            'support': evidence['hann'] > 0,
            'active': active,
            'evidence': evidence,
            'alpha_limit': self.alpha_max,
        }

        batch = c.shape[0]
        c_flat, native_t_flat = c.flatten(1), native_t.flatten(1)
        native_base = c_flat * native_t_flat
        native_windowed = native_base * self.hann_window
        return {
            'features': evidence['channels'],
            'cls_response': c_flat,
            'cont_response': native_t_flat,
            'windowed_cls_score': c_flat * self.hann_window,
            'windowed_cont_score': native_t_flat * self.hann_window,
            'base_score': native_base,
            'windowed_base_score': native_windowed,
            'calibrated_score': (
                evidence['c'] * evidence['t'] * factor
            ).flatten(1),
            'windowed_calibrated_score': score.flatten(1),
            'factor': factor.flatten(1),
            'log_ratio': evidence['d'].flatten(1),
            'alpha': alpha.flatten(1),
            'raw_alpha': alpha_scalar[:, None],
            'z_alpha': (
                branch_utility[:, 0] - branch_utility[:, 1]
            )[:, None],
            'reliability': (1.0 - action_probability[:, :1]),
            'reliability_cls': action_probability[:, 1:2],
            'reliability_cont': action_probability[:, 2:3],
            'reliability_base': action_probability[:, :1],
            'reliability_gate': (1.0 - action_probability[:, :1]),
            'intervention_gate': (1.0 - action_probability[:, :1]),
            'gate_logit': branch_utility.max(-1).values[:, None],
            'delta': branch_strength_logit.mean(-1, keepdim=True),
            'active_mask': evidence['active'],
            'trc_output': packet,
        }
