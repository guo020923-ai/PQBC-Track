"""FR1: foreground-reference evidence features; geometry uses raw search only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ForegroundReferenceERC(nn.Module):
    DESIGN = 'foreground_reference_evidence_erc_fr1'

    def __init__(self, dim, hidden=64, max_delta=.1, feat_sz=16,
                 force_off=False, inference_mode='FULL_ERC'):
        super().__init__()
        if hidden % 4 or not 0 < max_delta <= 1:
            raise ValueError('FR1 requires hidden divisible by four and bounded residual')
        self.design = self.DESIGN
        self.max_delta = max_delta
        self.force_off = force_off
        self.inference_mode = inference_mode
        self.query = nn.Linear(dim, hidden)
        self.foreground = nn.Linear(dim, hidden)
        self.language = nn.Linear(dim, hidden)
        self.competition = nn.Linear(dim, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        h = np.hanning(feat_sz)
        self.register_buffer('hann_window', torch.from_numpy(np.outer(h,h)).flatten(), persistent=False)

    def set_inference_mode(self, mode):
        if mode not in ('FULL_ERC', 'ERC_OFF'):
            raise ValueError('FR1 supports FULL_ERC / ERC_OFF, not GLOBAL_ONLY')
        self.inference_mode = mode

    def forward(self, search, template, template_mask, language, prompts, flag):
        batch = search.shape[0]
        active = flag.reshape(batch).eq(2)
        if template_mask is None:
            if active.any():
                raise ValueError('FR1 VL tracking requires the initial template foreground mask')
            mask = torch.zeros(template.shape[:2], dtype=torch.bool, device=template.device)
        else:
            mask = template_mask.reshape(batch, -1).bool()
        if mask.shape != template.shape[:2]:
            raise ValueError('FR1 template token/mask shape mismatch')
        valid = mask.any(-1)
        # Avoid all-masked attention NaNs. Empty foreground samples are disabled.
        safe = mask.clone(); safe[~valid, 0] = True
        q = self.query(search.detach())
        kv = self.foreground(template.detach())
        attended, _ = self.attention(q, kv, kv, key_padding_mask=~safe, need_weights=False)
        lang = self.language(language.detach().reshape(batch, -1, search.shape[-1]).mean(1))
        competition = self.competition((prompts[:,0] - .5*(prompts[:,1]+prompts[:,2])).detach())
        raw = self.up(F.gelu(self.norm(q + attended + lang[:,None] + competition[:,None])))
        scale = search.detach().norm(dim=-1,keepdim=True).clamp_min(1e-6)
        delta = self.max_delta * scale * raw / (scale + raw.norm(dim=-1,keepdim=True))
        enabled = active & valid
        if self.force_off or self.inference_mode == 'ERC_OFF': enabled = torch.zeros_like(enabled)
        elif self.inference_mode != 'FULL_ERC': raise ValueError(self.inference_mode)
        delta = torch.where(enabled[:,None,None], delta, torch.zeros_like(delta))
        return search + delta, {'delta':delta, 'active':active & valid,
                               'relative_norm':delta.norm(dim=-1)/scale.squeeze(-1),
                               'foreground_count':mask.sum(-1)}


def forward_fr1(head, out):
    """Original prompts/geometry, adapted evidence; native 2-logit train loss kept."""
    flag = out['flag']; raw = out['search']; batch = raw.shape[0]
    raw_cont, prompts = head.contractive_learning(out)
    adapted, info = head.erc(raw, out['template'], out.get('template_mask'),
                             out['txt_token'], prompts, flag)
    x = raw.transpose(1,2).reshape(batch,-1,head.feat_sz,head.feat_sz).contiguous()
    xp = adapted.transpose(1,2).reshape_as(x).contiguous()
    tokens = torch.cat([out['vis_token'],out['txt_token'],(out['vis_token']+out['txt_token'])/2],1)
    token = tokens[torch.arange(batch,device=raw.device), flag.reshape(-1).long(),:,None,None]
    cls = head.conv_cls(xp*token if head.cls_tokenize else xp).sigmoid().squeeze(1)
    # Extra raw response is used for diagnostics, teacher anchoring and prompt updates.
    raw_cls = head.conv_cls(x*token if head.cls_tokenize else x).sigmoid().squeeze(1)
    logits = head.logit_scale.exp() * (F.normalize(adapted,dim=-1) @ F.normalize(prompts,dim=-1).transpose(-2,-1))
    z = torch.zeros_like(logits[...,:1])
    negatives = torch.cat([logits[...,1:],z],-1).max(-1,keepdim=True).values if head.softmax_one else logits[...,1:].max(-1,keepdim=True).values
    cont = torch.cat([logits[...,:1], negatives],-1)
    if out.get('prompt') is not None and head.softmax_one: cont = torch.cat([cont,z],-1)
    def deploy_probability(a):
        if head.softmax_one and a.shape[-1]==2: a=torch.cat([a,torch.zeros_like(a[...,:1])],-1)
        return a.softmax(-1)[...,0].reshape_as(cls)
    t = deploy_probability(cont); rt = deploy_probability(raw_cont)
    offset = head.conv_offset(x)
    if head.offset_sigmoid: offset=offset.sigmoid()
    tr = head.conv_bbox(x).sigmoid(); gr = head.conv_bbox_grounding(x).sigmoid()
    size = torch.where(flag.reshape(batch,1,1,1).eq(1),gr,tr)
    native_t = cont.softmax(-1)[...,0].reshape_as(cls)
    boxes,bbox = head.convert2bbox(cls,offset,size,(cls*native_t).flatten(1))
    hann = head.erc.hann_window
    score = cls.flatten(1).double()*hann*t.flatten(1).double()
    raw_score = raw_cls.flatten(1).double()*hann*rt.flatten(1).double()
    out.update(cls_score=cls if not head.joint_cls else cls*native_t,
               cls_score_test=cls,cont_score=cont,prompts=prompts,bbox_map=boxes,
               pred_boxes=bbox,erc_pred_boxes=bbox,
               erc_factor=torch.ones_like(cls.flatten(1)),
               erc_cls_response=raw_cls.flatten(1),erc_cont_response=rt.flatten(1),
               erc_windowed_base_score=raw_score,erc_windowed_score=score,
               erc_active_mask=info['active'],erc_fr1=info,
               erc_prompt_base_score=(raw_cls*rt).flatten(1))
    return out
