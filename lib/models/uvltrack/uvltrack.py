from torch import nn

from lib import registry
from lib.models.backbones import *
from lib.models.heads import *


class UVLTrack(nn.Module):
    """ This is the base class for Transformer Tracking, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, box_head):
        """ Initializes the model.
        """
        super().__init__()
        # self.language_backbone = language_backbone
        self.backbone = backbone
        self.box_head = box_head

    def forward(self, template, search, text, template_mask, context_mask, flag):
        # text_feature = self.language_backbone(text) # b, s, c  b, s  FT
        backbone_info = self.backbone(template, search, text, flag)
        backbone_info['template_mask'] = template_mask
        backbone_info['context_mask'] = context_mask
        head_info = self.box_head(backbone_info)
        return head_info
    
    def forward_prompt_init(self, template, search, text, template_mask, context_mask, flag):
        backbone_info = self.backbone(template, search, text, flag)
        backbone_info['template_mask'] = template_mask
        backbone_info['context_mask'] = context_mask
        prompt = self.box_head.forward_prompt(backbone_info)
        return prompt
    
    def forward_prompt(self, out_dict, template_mask, context_mask):
        backbone_info = out_dict
        backbone_info['template_mask'] = template_mask
        backbone_info['context_mask'] = context_mask
        prompt = self.box_head.forward_prompt(backbone_info)
        return prompt
        
    
    def forward_test(self, template, search, text, prompt, flag, template_mask=None):
        backbone_info = self.backbone(template, search, text, flag)
        backbone_info['prompt'] = prompt
        if template_mask is not None:
            backbone_info['template_mask'] = template_mask
        head_info = self.box_head(backbone_info)
        return head_info
        
@registry.MODELS.register('uvltrack')
def build_model(cfg):
    # language_backbone = registry.BACKBONES[cfg.MODEL.BACKBONE.LANGUAGE.TYPE](cfg)
    backbone = registry.BACKBONES[cfg.MODEL.BACKBONE.TYPE](cfg)
    head = registry.HEADS[cfg.MODEL.HEAD.TYPE](cfg)  # a simple corner head
    model = UVLTrack(
        # language_backbone,
        backbone,
        head
    )
    if cfg.MODEL.RQC.ENABLED:
        if cfg.MODEL.HEAD.ERC.ENABLED:
            raise ValueError('RQC pilot requires legacy ERC disabled')
        rqc_design = cfg.MODEL.RQC.DESIGN
        if rqc_design == 'candidate_quality_v3':
            from lib.models.heads.candidate_quality_v3 import CandidateQualityV3
            rqc_type = CandidateQualityV3
        elif rqc_design == 'candidate_quality_jq':
            from lib.models.heads.candidate_quality_jq import JointQuality
            rqc_type = JointQuality
        elif rqc_design == 'candidate_quality_bq':
            from lib.models.heads.candidate_quality_bq import BoundaryQuality
            rqc_type = BoundaryQuality
        elif rqc_design == 'candidate_quality_pairgain':
            from lib.models.heads.candidate_quality_pairgain import PairGain
            rqc_type = PairGain
        elif rqc_design == 'candidate_quality_pr':
            from lib.models.heads.candidate_quality_pr import CandidateQualityPR
            rqc_type = CandidateQualityPR
        elif rqc_design == 'candidate_quality_pd':
            from lib.models.heads.candidate_quality_pd import CandidateQualityPD
            rqc_type = CandidateQualityPD
        elif rqc_design == 'candidate_quality_pc':
            from lib.models.heads.candidate_quality_pc import CandidateQualityPC
            rqc_type = CandidateQualityPC
        elif rqc_design == 'candidate_quality_jc':
            from lib.models.heads.candidate_quality_jc import CandidateQualityJC
            rqc_type = CandidateQualityJC
        elif rqc_design == 'candidate_quality_r1':
            from lib.models.heads.candidate_quality_r1 import CandidateQualityR1
            rqc_type = CandidateQualityR1
        elif rqc_design == 'candidate_quality_v4':
            from lib.models.heads.candidate_quality_v4 import CandidateQualityV4
            rqc_type = CandidateQualityV4
        elif rqc_design == 'candidate_quality_v5':
            from lib.models.heads.candidate_quality_v5 import CandidateQualityV5
            rqc_type = CandidateQualityV5
        elif rqc_design == 'candidate_quality_v5_advantage':
            from lib.models.heads.candidate_quality_v5_advantage import (
                CandidateQualityV5Advantage,
            )
            rqc_type = CandidateQualityV5Advantage
        else:
            raise ValueError('Unknown RQC design: %s' % rqc_design)
        rqc_kwargs = dict(
            feature_dim=cfg.MODEL.HIDDEN_DIM,
            reference=cfg.MODEL.RQC.REFERENCE,
            scale=cfg.MODEL.RQC.SCALE,
        )
        if rqc_design == 'candidate_quality_v5':
            rqc_kwargs['erc_scale'] = cfg.MODEL.RQC.ERC_SCALE
        elif rqc_design == 'candidate_quality_v5_advantage':
            rqc_kwargs['advantage_scale'] = cfg.MODEL.RQC.ADVANTAGE_SCALE
        model.rqc = rqc_type(**rqc_kwargs)
    return model
