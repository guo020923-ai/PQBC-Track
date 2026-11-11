"""Standalone tracker: native baseline proposals + learned ERC/RQC ranking.

Raw online prompt update stays native; only final selection is replaced.
No old tracker/YAML/checkpoint is modified. No GT access in this module.
"""
import torch
from lib.test.tracker.uvltrack import UVLTrack
from lib.models.heads.candidate_quality_v2 import CandidateQualityV2,candidates

class CandidateQualityTracker(UVLTrack):
    def __init__(self,params,dataset_name):
        super().__init__(params,dataset_name)
        if self.erc_enabled or not self.cfg.MODEL.HEAD.SOFTMAX_ONE:
            raise ValueError('Prototype requires official baseline, legacy ERC disabled, SOFTMAX_ONE')
        saved=torch.load(params.quality_checkpoint,map_location='cpu')
        assert saved.get('architecture')=='candidate_quality_v2', 'Wrong checkpoint version'
        self.quality_model=CandidateQualityV2(saved['feature_dim'],saved['hidden']).cuda().eval()
        self.quality_model.load_state_dict(saved['net'],strict=True)
        self.quality_k=saved['k'];self.use_reliability=saved['use_reliability']
        self.quality_enabled=True
        original=self.network.forward_test
        def ranked_forward(*args,**kwargs):
            out=original(*args,**kwargs)
            if not self.quality_enabled or int(self.flag.item())!=2:return out
            with torch.no_grad():
                batch=candidates(out,self.template_mask,self.quality_k)
                pred=self.quality_model(batch,self.use_reliability)
                local=pred['quality'].argmax(-1)
                winner=batch['indices'].gather(1,local[:,None])
                # Existing baseline tracker multiplies factor after Hann*C*T.
                # One-hot routing selects the quality winner; no score mixing threshold.
                factor=torch.zeros_like(out['cls_score_test'].flatten(1))
                factor.scatter_(1,winner,1.)
                out['erc_factor']=factor
            return out
        self.network.forward_test=ranked_forward
        # Existing tracker uses this switch only for factor application.
        self.erc_enabled=True

def get_tracker_class():return CandidateQualityTracker
