"""Original test entry, only substitute post-sequence memory lifecycle adapter."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import test_center_joint_20260930 as original
from center_joint_lifecycle_20261001 import MemorySafeCenterEvaluationTracker

if __name__=='__main__':
    original.JointCenterEvaluationTracker=MemorySafeCenterEvaluationTracker
    original.main()
