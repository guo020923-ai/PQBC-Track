"""Memory-lifetime-only adapter; trained model and prediction code unchanged."""
import gc,json,os
import torch
from center_joint_tracker_20260930 import JointCenterTracker,JointCenterEvaluationTracker

class MemorySafeCenterTracker(JointCenterTracker):
    def close(self):
        if getattr(self,'_closed',False):return
        # Do NOT store a bound method back into its own network instance: that
        # creates network -> bound method -> network, retaining CUDA parameters
        # until a later Python cyclic-GC pass, which is not CUDA-memory-aware.
        network=getattr(self,'network',None)
        original=getattr(self,'_original_forward',None)
        if original is not None:
            assert original.__self__ is network
            assert original.__func__ is type(network).forward_test
            network.__dict__.pop('forward_test',None)  # restore class descriptor
        capture=getattr(self,'_capture',None)
        if capture is not None:capture.__exit__()
        # Tracking is finished. Drop all search/template/prompt/model tensors
        # even if an exception traceback briefly retains the tracker object.
        self.__dict__.clear()
        self._closed=True

class MemorySafeCenterEvaluationTracker(JointCenterEvaluationTracker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tracker_class=MemorySafeCenterTracker

    def run_sequence(self,seq,debug=None):
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        try:
            return super().run_sequence(seq,debug)
        finally:
            # Parent closes the live tracker even on sequence exceptions. GC
            # handles partial-construction cycles; empty_cache releases unused
            # allocator blocks for the other processes sharing the same GPU.
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                free,total=torch.cuda.mem_get_info()
                print('GPU_RELEASE',json.dumps(dict(pid=os.getpid(),sequence=seq.name,
                    allocated_mib=round(torch.cuda.memory_allocated()/2**20,2),
                    reserved_mib=round(torch.cuda.memory_reserved()/2**20,2),
                    free_mib=round(free/2**20,2))),flush=True)
