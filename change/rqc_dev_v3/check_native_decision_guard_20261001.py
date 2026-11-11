"""Synthetic decision-regret gradient tests. No fitting or optimizer update."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from head_native_decision_guard_20261001 import decision_guard

def output(winner,teacher=False):
    cls=torch.full((1,16),.1);cls[0,winner]=.9
    boxes=torch.tensor([[[.6,.4,.2,.2]]*16]);boxes[0,5]=torch.tensor([.4,.4,.2,.2])
    boxes.requires_grad_(True);cls.requires_grad_(True)
    return dict(cls_score_test=cls,cont_score=torch.zeros(1,16,3,requires_grad=True),
        bbox_map=boxes,flag=torch.tensor([[2]]))

def main():
    torch.set_num_threads(1)
    data=dict(center_gt=torch.tensor([[[30.,30.,20.,20.]]]),center_hw=torch.tensor([[[100.,100.]]]),
        center_side=torch.tensor([[100.]]),center_origin=torch.zeros(1,1,2),center_flip=torch.zeros(1,1,dtype=torch.bool))
    student=output(6);teacher=output(5)
    loss,stats=decision_guard(student,teacher,data);assert loss>0
    loss.backward()
    assert student['cls_score_test'].grad[0,5]<0 and student['cls_score_test'].grad[0,6]>0
    assert student['bbox_map'].grad.abs().sum()>0
    assert all(teacher[k].grad is None for k in ('cls_score_test','cont_score','bbox_map'))
    identical=output(5);zero,_=decision_guard(identical,output(5),data);assert zero==0
    improved=output(6);worse_teacher=output(5)
    with torch.no_grad():
        improved['bbox_map'][0,6]=torch.tensor([.4,.4,.2,.2])
        worse_teacher['bbox_map'][0,5]=torch.tensor([.44,.4,.2,.2])
    zero,_=decision_guard(improved,worse_teacher,data);assert zero==0
    inactive=output(6);inactive['flag'].zero_();zero,_=decision_guard(inactive,output(5),data);assert zero==0
    print('NATIVE_DECISION_GUARD_CHECK_PASS',json.dumps(dict(positive_regret=float(loss.detach()),
        recovery_ranking_gradient=True,selected_box_gradient=True,teacher_gradient_none=True,
        identical_no_penalty=True,student_improvement_not_penalized=True,inactive_zero=True,
        optimizer_steps=0,stats=stats)))
if __name__=='__main__':main()
