"""TRC-v5g: branch-conditional permission for deployable C/T actions.

The original Gate still decides whether a frame is broadly correctable.  A
second two-output veto estimates whether the *chosen* C or T correction is
permitted.  This removes the invalid assumption that "some correction exists"
implies that the currently predicted branch direction is safe to execute.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.heads.target_reference_erc_v5f import TargetReferenceERCV5F


class TargetReferenceERCV5G(TargetReferenceERCV5F):
    DESIGN = 'target_reference_erc_trc_v5g_branch_conditional_permission'

    def __init__(self, feat_sz, hidden=32, alpha_max=0.5, temperature=1.0,
                 d_max=4.0, inference_mode='FULL_ERC', force_off=False,
                 permission_threshold=0.15,
                 permission_initial_probability=0.05,
                 direction_t_threshold=0.35,
                 branch_permission_threshold=0.5,
                 branch_permission_initial_probability=0.5):
        if not 0 < branch_permission_threshold < 1:
            raise ValueError('branch_permission_threshold must be in (0, 1)')
        if not 0 < branch_permission_initial_probability < 1:
            raise ValueError(
                'branch_permission_initial_probability must be in (0, 1)'
            )
        super().__init__(
            feat_sz, hidden, alpha_max, temperature, d_max,
            inference_mode, force_off, permission_threshold,
            permission_initial_probability, direction_t_threshold,
        )
        decision_dim = self.permission_head[0].in_features
        self.branch_permission_head = nn.Sequential(
            nn.Linear(decision_dim, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.branch_permission_head[-1].weight)
        initial_logit = self.temperature * math.log(
            branch_permission_initial_probability
            / (1.0 - branch_permission_initial_probability)
        )
        nn.init.constant_(self.branch_permission_head[-1].bias, initial_logit)
        self.branch_permission_threshold = float(branch_permission_threshold)
        self.design = self.DESIGN

    def forward(self, c, native_t, *, search, prompts, template, template_mask,
                logit_scale, flag):
        output = super().forward(
            c, native_t, search=search, prompts=prompts, template=template,
            template_mask=template_mask, logit_scale=logit_scale, flag=flag,
        )
        packet = output['trc_output']
        branch_logits = self.branch_permission_head(packet['decision_features'])
        branch_probability = torch.sigmoid(branch_logits / self.temperature)
        row = torch.arange(branch_logits.shape[0], device=branch_logits.device)
        hard_direction = packet['hard_direction_index']
        selected_branch_probability = branch_probability[row, hard_direction]
        hard_branch_permission = selected_branch_probability.gt(
            self.branch_permission_threshold
        )
        hard_permission = (
            packet['hard_permission'].bool() & hard_branch_permission
        ).to(selected_branch_probability)

        soft_permission = (
            packet['permission_probability'] * selected_branch_probability
        )
        permission = (
            hard_permission + soft_permission - soft_permission.detach()
            if self.training else hard_permission
        )
        hard_policy = F.one_hot(hard_direction, 2).to(branch_probability)
        direction_policy = (
            hard_policy + packet['direction_probability']
            - packet['direction_probability'].detach()
            if self.training else hard_policy
        )
        direction = direction_policy[:, 0] - direction_policy[:, 1]
        raw_alpha_scalar = (
            self.alpha_max * permission * direction * packet['strength']
        )
        soft_signed_permission = packet['permission_probability'] * (
            branch_probability[:, 0] * packet['direction_probability'][:, 0]
            - branch_probability[:, 1] * packet['direction_probability'][:, 1]
        )
        raw_soft_alpha_scalar = (
            self.alpha_max * soft_signed_permission * packet['strength']
        )
        active = packet['active']
        alpha_scalar = torch.where(
            active, raw_alpha_scalar, torch.zeros_like(raw_alpha_scalar)
        )
        soft_alpha_scalar = torch.where(
            active, raw_soft_alpha_scalar,
            torch.zeros_like(raw_soft_alpha_scalar),
        )
        alpha = alpha_scalar[:, None, None].expand_as(packet['evidence']['d'])
        soft_alpha = soft_alpha_scalar[:, None, None].expand_as(alpha)
        correction = alpha * packet['evidence']['d']
        factor = correction.exp()
        mode = 'ERC_OFF' if self.force_off else self.inference_mode
        if mode == 'ERC_OFF':
            alpha = torch.zeros_like(alpha)
            correction = torch.zeros_like(correction)
            factor = torch.ones_like(factor)

        action_c = (
            packet['permission_probability'] * branch_probability[:, 0]
            * packet['direction_probability'][:, 0]
        )
        action_t = (
            packet['permission_probability'] * branch_probability[:, 1]
            * packet['direction_probability'][:, 1]
        )
        action_probability = torch.stack([
            1.0 - action_c - action_t, action_c, action_t,
        ], -1).clamp_min(0.0)
        hard_action = torch.where(
            hard_permission.bool(), hard_direction + 1,
            torch.zeros_like(hard_direction),
        )
        score = packet['base'] * factor
        packet.update({
            'branch_permission_logits': branch_logits,
            'branch_permission_probability': branch_probability,
            'branch_permission_threshold': self.branch_permission_threshold,
            'selected_branch_permission_probability':
                selected_branch_probability,
            'hard_branch_permission': hard_branch_permission.to(factor),
            'hard_permission': hard_permission,
            'action_probability': action_probability,
            'hard_action': hard_action,
            'alpha_scalar': alpha_scalar,
            'soft_alpha_scalar': soft_alpha_scalar,
            'alpha': alpha,
            'soft_alpha': soft_alpha,
            'factor': factor,
            'score': score,
            'log_score': packet['log_base'] + correction,
        })
        evidence = packet['evidence']
        output.update({
            'calibrated_score': (
                evidence['c'] * evidence['t'] * factor
            ).flatten(1),
            'windowed_calibrated_score': score.flatten(1),
            'factor': factor.flatten(1),
            'alpha': alpha.flatten(1),
            'raw_alpha': alpha_scalar[:, None],
            'reliability_cls': action_probability[:, 1:2],
            'reliability_cont': action_probability[:, 2:3],
            'reliability_base': action_probability[:, :1],
            'trc_output': packet,
        })
        return output
