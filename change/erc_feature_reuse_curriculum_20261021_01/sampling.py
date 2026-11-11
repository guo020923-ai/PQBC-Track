"""Exact causal CPU RGB prefetch plus existing native batch-one graph replay.

Unlike the old sampler, there is NO legacy ERC model/decision/evidence shortcut.
Images are still decoded for the native source, never passed into the new ERC.
"""
import time
import math
import importlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch


def read_rgb(path):
    import cv2
    image=cv2.imread(str(path))
    if image is None:raise ValueError('Missing RGB frame: '+str(path))
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


class PrefetchFrames:
    def __init__(self,stream):
        self.stream=stream;self.original_next=stream.next_sample
        self.executor=ThreadPoolExecutor(max_workers=2)
        self.clip=None;self.pending={};self.depth=8

    def clear(self):
        for future in self.pending.values():future.cancel()
        self.pending.clear();self.clip=None

    def enable(self):
        self.clear();self.stream.next_sample=self.next_sample

    def disable(self):
        self.stream.next_sample=self.original_next;self.clear()

    def close(self):
        self.disable();self.executor.shutdown(wait=True)

    def image(self,frame):
        s=self.stream
        if self.clip is not s.clip:self.clear();self.clip=s.clip
        for i in range(frame,min(frame+self.depth,s.clip['end'])):
            if i not in self.pending:self.pending[i]=self.executor.submit(read_rgb,s.clip['paths'][i])
        return self.pending.pop(frame).result()

    def next_sample(self):
        from .bridge import base_class
        s=self.stream
        if s.clip is None:s._new_clip()
        while s.cursor<s.clip['end']:
            frame=s.cursor;s.cursor+=1;image=self.image(frame)
            before=np.asarray(s.collector.state,dtype=np.float64).copy()
            used_prompt=s.collector.prompt.detach().clone();started=time.monotonic()
            warmup=frame<=s.clip['begin']+s.spec['WARMUP_FRAMES']
            fixed_output=base_class().track(s.collector,image) if warmup else s.collector.track(image)
            s.seconds+=time.monotonic()-started;s.frames+=1
            if warmup or not s.clip['visible'][frame]:continue
            # No current search GT before source inference and feature construction.
            device=s.collector.last_inputs['search'].device
            truth=s.clip['annotations'][frame].detach().clone()[None].to(device).double()
            if not torch.isfinite(truth).all() or not truth[:,2:].gt(0).all():continue
            fixed=torch.tensor([fixed_output['target_bbox']],device=device,dtype=torch.double)
            assert s.collector.state==s.collector.last_native_box
            side=math.ceil(math.sqrt(before[2]*before[3])*s.collector.params.search_factor)
            return dict(inputs={k:v.detach().clone() for k,v in s.collector.last_inputs.items()},
                native_bbox_map=s.collector.observed['bbox_map'].detach().clone(),
                fixed=fixed,truth=truth,hw=torch.tensor([image.shape[:2]],device=device,dtype=torch.double),
                meta=dict(domain=s.clip['domain'],name=s.clip['name'],begin=s.clip['begin'],frame=frame,
                    source=s.source,previous_state=before.tolist(),
                    crop_origin=np.rint(before[:2]+before[2:]/2.-side/2.).tolist(),side=side,
                    prompt=s.collector.prompt.detach().clone(),forward_prompt_updated=not torch.equal(used_prompt,s.initial_prompt)))
        return None


class GraphSampler:
    def __init__(self,streams):
        # load_legacy has set the original read-only graph module import path.
        NativeGraph=importlib.import_module('graph_sampler').GraphForward
        self.streams=streams
        self.adapters={i:NativeGraph(s.collector) for i,s in streams.items()}
        self.fast_frames={i:PrefetchFrames(s) for i,s in streams.items()}
        for i,s in streams.items():
            s.collector._original_forward=self.adapters[i];self.fast_frames[i].enable()

    def collect(self,tickets):
        rows=[]
        for flag,i in tickets:
            if flag==2:rows.extend(self.streams[i].next_pair())
        return rows

    def close(self):
        for i,s in self.streams.items():
            s.collector._original_forward=self.adapters[i].native
            self.fast_frames[i].close()

