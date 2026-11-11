"""Joint-only crop metadata fix: use native sample_target's Python precision.

No change to sampling, images, crop labels, RNG, or official/shared processing.
The frozen prototype's metadata module is retained byte-for-byte.
"""
import math
from center_formal_contract_20260930 import (
    MetadataTrackProcessing,install_metadata as install_original_metadata)

class ExactMetadataTrackProcessing(MetadataTrackProcessing):
    def _get_jittered_box(self,box,mode):
        result=super()._get_jittered_box(box,mode)
        if self._center_recording and mode=='search':
            # Native sample_target uses Python floats before multiplying w*h.
            # float(tensor_w*tensor_h) rounds the product in float32; ceil can
            # choose an adjacent crop side near an integer-size boundary.
            x,y,w,h=result.tolist()
            side=math.ceil(math.sqrt(w*h)*self.search_area_factor[mode])
            center=result[:2]+result[2:]/2
            size=self.output_sz[mode]
            self._center_meta[-1].update(side=float(side),
                origin=(center-side/2).clone(),
                annotation_origin=(center-side*(size-1)/(2*size)).clone())
        return result

def install_metadata(processing):
    if isinstance(processing,ExactMetadataTrackProcessing):return processing
    obj=install_original_metadata(processing)
    assert isinstance(obj,MetadataTrackProcessing)
    obj.__class__=ExactMetadataTrackProcessing
    return obj
