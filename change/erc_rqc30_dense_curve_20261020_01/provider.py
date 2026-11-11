"""Captured-source adapter only. No tracker/weights/dataset construction."""
import math
import torch
from .inputs import build_inputs
from .geometry import action_lattice
from .checkpoint import SOURCE_SHA,SPLIT_SHA

class RQC30Provider:
    def __init__(self,source_attestation):
        if source_attestation.get('source_sha256')!=SOURCE_SHA or source_attestation.get('split_sha256')!=SPLIT_SHA:
            raise ValueError('Wrong fixed RQC30/original TRAIN identity')
        if source_attestation.get('epoch')!=30 or source_attestation.get('variant')!='directional_center_guard_joint':
            raise ValueError('Wrong RQC30 source semantic variant')
        self.source_attestation=dict(source_attestation)

    @torch.no_grad()
    def __call__(self,source,image,fixed_output,previous_native):
        if not hasattr(source,'flag') or int(source.flag.item())!=2:
            raise ValueError('ERC requires vision-language branch')
        out=source.observed;device=out['search'].device
        previous=torch.as_tensor(previous_native,device=device,dtype=torch.double)
        if previous.shape!=(4,) or not torch.isfinite(previous).all() or not previous[2:].gt(0).all():
            raise ValueError('Invalid previous native geometry')
        side=math.ceil(math.sqrt(float(previous[2]*previous[3]))*source.params.search_factor)
        origin=torch.round(previous[:2]+previous[2:]/2-side/2)[None]
        fixed=torch.tensor([fixed_output['target_bbox']],device=device,dtype=torch.double)
        hw=torch.tensor([image.shape[:2]],device=device,dtype=torch.double)
        old=source.last_inputs
        inputs=build_inputs(out['search'],source.initial_template,source.initial_foreground,
            source._capture.maps['conv_bbox'],fixed,origin,
            torch.tensor([side],device=device,dtype=torch.double),hw,old['source_context'],
            source.observed_rgb,source.template,out['bbox_map'],action_lattice().to(device),
            inherited_valid=old['valid'],inherited_executed_delta=old['executed_delta'])
        return inputs,fixed,hw
