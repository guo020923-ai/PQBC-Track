"""Inference wrapper; never writes ERC outputs into the frozen source history."""
import torch
from change.erc_rqc30_dense_curve_20261020_01.runtime import clone, equal, history
from .geometry import apply_size


class OutputOnlyERC:
    def __init__(self, source, model, provider):
        self.source, self.model, self.provider = source, model.eval(), provider
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def initialize(self, image, info):
        # Only legal first-frame annotation/language enters native initialize.
        self.source.initialize(image,info)

    @torch.no_grad()
    def track(self, image, info=None):
        fixed_output = clone(self.source.track(image,None))
        before = history(self.source)
        inputs, fixed, hw = self.provider(self.source,image,fixed_output)
        pred = self.model(**inputs)
        result = clone(fixed_output)
        if len(pred['index']) != 1:
            raise ValueError('Tracking runtime is one image per native trajectory')
        if pred['index'].item() != 0:
            result['target_bbox'] = apply_size(fixed,pred['delta'],hw)[0].cpu().tolist()
        else:
            assert equal(result,fixed_output)
        assert equal(before,history(self.source)), 'ERC/provider changed native history'
        self.last_prediction = pred
        self.last_rqc_box, self.last_output_box = clone(fixed_output['target_bbox']),clone(result['target_bbox'])
        return result
