"""Paired feature acquisition only. No fitting, training, or checkpoint writes."""
import json,hashlib
import torch
from head_quality_features_20260930 import HeadQualityCapture
from audit_jq_natural_20260930 import main

class Collector:
    def attach(self,tracker):
        self.net=tracker.network;self.cap=HeadQualityCapture(self.net.box_head)
        self.cap.__enter__();self.rows=[];self.last={}
        def input_hook(module,inputs):self.last['old']=inputs[0].detach().cpu()
        def model_hook(module,inputs,out):
            b=inputs[0]
            self.last['new']=self.cap.gather(b['indices']).cpu()
            self.last['scalar']=torch.cat((b['boxes'],b['score_features'],b['evidence']),-1).detach().cpu()
            self.last['prior']=b['scores'].detach().cpu()
            self.last['prediction']=out['quality'].detach().cpu()
        self.handles=[self.net.rqc.quality[0].register_forward_pre_hook(input_hook),
                      self.net.rqc.register_forward_hook(model_hook)]
    def record(self,metadata,frame,ious):
        # Uniform temporal sampling; never select based on GT gain or errors.
        if frame%5:return
        assert self.last['old'].shape==(1,10,198)
        assert self.last['new'].shape==(1,10,288)
        row={k:v[0].clone() for k,v in self.last.items()}
        row.update(sequence=metadata['name'],domain=metadata['domain'],frame=frame,
                   iou=torch.as_tensor(ious,dtype=torch.float32))
        self.rows.append(row)
    def finish(self,dest):
        for h in self.handles:h.remove()
        self.cap.__exit__()
        assert self.rows
        for row in self.rows:
            for k in ('old','new','scalar','prior','prediction','iou'):
                assert torch.isfinite(row[k]).all() and not row[k].requires_grad
        path=dest/'paired_features.pt';assert not path.exists()
        torch.save(dict(rows=self.rows,source='JQ E30 native-response natural trajectory',
            note='Old learned198 features vs raw Head288; matched-budget fitting must control dimensions and scalar inputs. Train-source, not independent benchmark.'),path)
        summary=dict(samples=len(self.rows),sequences=len(set(r['sequence'] for r in self.rows)),
            bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            optimizer_steps=0,trained=False,uniform_frame_stride=5,
            domains={d:sum(r['domain']==d for r in self.rows) for d in set(r['domain'] for r in self.rows)})
        (dest/'collection.json').write_text(json.dumps(summary,indent=2))
        print('PAIRED_COLLECTION_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    with torch.no_grad():main('jq_paired_features_20260930',Collector())
