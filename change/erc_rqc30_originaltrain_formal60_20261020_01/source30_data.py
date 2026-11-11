"""Restore original TRAIN membership; no added ERC split; adapter for fresh full60."""
import json
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'change/erc_action_comparison_natural_20261011'))
import natural_common as nc
from natural_data import Pools,FrameStream,CleanCollector,collate,source_snapshot,assert_unchanged
from comparison_runtime import JointCenterEvaluationTracker
sys.path.insert(0,str(ROOT/'change/erc_action_comparison_natural_20261011/batched_20261011'))
from graph_sampler import GraphSampler

NAME='baseline_base_head_directional_center_guard_bs16_30ep_20261001'
CONFIG=ROOT/'experiments/uvltrack'/(NAME+'.yaml')
SOURCE=ROOT/'change/directional_center_joint_training_20261001'/NAME/'DirectionalCenterJoint_ep0030.pth.tar'
Collector30=CleanCollector

def settings():
    spec,cfg=nc.specification()
    assert spec['SOURCE_CHECKPOINT_SHA256']=='1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd'
    return spec,cfg

def parameters():
    p=JointCenterEvaluationTracker('tnl2k',SOURCE,0,epoch=30).get_parameters(30)
    assert Path(p.center_checkpoint)==SOURCE and p.cfg.TEST.EPOCH==30
    return p

class FixedSequenceRNG:
    def __init__(self,sid,seed):self.sid=sid;self.rng=random.Random(seed)
    def randrange(self,n):assert self.sid<n;return self.sid
    def choice(self,items):return self.rng.choice(items)

class OriginalTrainPools(Pools):
    """Original loader-defined TRAIN pool. No ERC-specific sequence exclusion.

    A historical held-out plan is rejected instead of silently ignored. This
    module restores membership only; natural causal clips remain unchanged.
    """
    def __init__(self,cfg,plan=None):
        if plan is not None:
            if plan.get('training_membership')!='original_loader_train_no_ERC_holdout':
                raise ValueError('Original TRAIN membership policy required')
            if any(plan.get(k) for k in ('blocked','monitor','confirmation','diagnostic')):
                raise ValueError('ERC-added held-out subsets are forbidden')
        super().__init__(cfg)
        self.blocked={i:set() for i in self.vl}
        self.membership={self.datasets[i].get_name():list(self.datasets[i].sequence_list) for i in self.vl}
        assert all(not v for v in self.blocked.values())

FitPools30=OriginalTrainPools


def rebuild(pools,row):
    clip=pools.choose_clip(row['source'],FixedSequenceRNG(row['sid'],row['seed']),192,16)
    assert clip['name']==row['sequence'] and clip['begin']==row['begin'] and clip['end']==row['end']
    return clip
