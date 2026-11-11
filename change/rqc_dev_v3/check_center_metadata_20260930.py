"""Native augmentation equivalence and official E300 cache checks, no optimization."""
import sys,random,json,copy
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from center_formal_contract_20260930 import install_metadata,file_hash,decode_training_boxes
from lib.train.data.processing import TrackProcessing
from lib.train.data import transforms as tfm
from lib.utils import TensorDict
from prepare_locked_refiner_protocol_20260930 import load_role
from center_safe_refiner_20260930 import CenterSafeRefiner,center_safe_loss
from local_box_refiner_20260930 import decode_image_boxes
from lib.test.tracker.uvltrack import UVLTrack
from lib.utils.box_ops import clip_box

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n)
def main():
    torch.set_num_threads(1)
    image=np.random.RandomState(13).randint(0,256,size=(240,320,3),dtype=np.uint8)
    original=TrackProcessing(search_area_factor={'template':2.,'search':4.},output_sz={'template':128,'search':256},
        center_jitter_factor={'template':0.,'search':3.5},scale_jitter_factor={'template':0.,'search':.5},
        center_jitter_factor_grounding=3.5,scale_jitter_factor_grounding=.5,mode='sequence',settings=None,
        dynamic_cls=True,gaussian_iou=.7,
        transform=tfm.Transform(tfm.ToTensorAndJitter(.2),tfm.RandomHorizontalFlip_Norm(.5),tfm.Normalize([.485,.456,.406],[.229,.224,.225])),
        joint_transform=tfm.Transform(tfm.RandomHorizontalFlip(.5),tfm.ToGrayscale(.05)))
    modified=install_metadata(copy.deepcopy(original));valid_checks=0;flips=set()
    for i in range(30):
        payload=TensorDict(template_images=[image.copy()],search_images=[image.copy(),image.copy()],
            template_anno=[torch.tensor([2.,5.,40.,50.])],
            search_anno=[torch.tensor([3.,7.,38.,48.]),torch.tensor([230.,160.,40.,60.])],flag=torch.tensor([[2]]))
        seed(i);a=original.track_process(TensorDict(copy.deepcopy(dict(payload))));rng_a=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        seed(i);b=modified.track_process(TensorDict(copy.deepcopy(dict(payload))));rng_b=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        assert bool(a['valid'])==bool(b['valid'])
        assert rng_a[0]==rng_b[0] and torch.equal(rng_a[2],rng_b[2])
        assert rng_a[1][0]==rng_b[1][0] and np.array_equal(rng_a[1][1],rng_b[1][1]) and rng_a[1][2:]==rng_b[1][2:]
        if not a['valid']:continue
        for k,v in a.items():
            if torch.is_tensor(v):assert torch.equal(v,b[k]),'Native processing changed: '+k
        flips.update(b['center_flip'].tolist());valid_checks+=1
        # Direct native TEST decoder comparison, including flip and boundary clip.
        predicted=torch.tensor([[.15,.7,.2,.25],[.9,.1,.3,.2]])
        active=torch.ones(2,dtype=torch.bool)
        decoded=decode_training_boxes(predicted,b,active)
        expected=[]
        for j,meta in enumerate(modified._center_meta):
            box=predicted[j].clone()
            if b['center_flip'][j]:box[0]=1-box[0]
            stub=SimpleNamespace(state=meta['extract_box'].tolist(),params=SimpleNamespace(search_size=256))
            mapped=UVLTrack.map_box_back(stub,(box*meta['side']).tolist(),256/meta['side'])
            h,w=meta['hw'].tolist();expected.append(clip_box(mapped,h,w,margin=10))
        torch.testing.assert_close(decoded,torch.tensor(expected),rtol=2e-5,atol=.0001)
    assert valid_checks>=10 and flips=={True,False}
    cache_path=ROOT/'change/rqc_dev_v3/official_center_cache_20260930/paired_cache.pt'
    cache=torch.load(str(cache_path),map_location='cpu');identity=cache['identity'];rows=cache['rows']
    assert identity['tracker']=='baseline_base' and not identity['smoke']
    source=ROOT/'checkpoints/train/uvltrack/baseline_base/UVLTrack_ep0300.pth.tar'
    assert identity['checkpoint_sha256']==file_hash(source)
    manifest=load_role(ROOT/'change/rqc_dev_v3/locked_refiner_protocol_20260930','fit')+load_role(ROOT/'change/rqc_dev_v3/locked_refiner_protocol_20260930','dev')
    assert {(r['sequence'],r['role']) for r in rows}=={(m['name'],m['role']) for m in manifest}
    assert len(rows)==len({(r['sequence'],r['frame']) for r in rows})
    fit=[r for r in rows if r['role']=='fit'][:64]
    features=torch.stack([r['features'] for r in fit]).requires_grad_(True)
    boxes=torch.stack([r['box'] for r in fit]).requires_grad_(True)
    truth=torch.stack([r['gt'] for r in fit]).requires_grad_(True)
    origin=torch.stack([r['origin'] for r in fit]);side=torch.tensor([r['side'] for r in fit]);hw=torch.tensor([r['image_hw'] for r in fit])
    model=CenterSafeRefiner();out=model(features,boxes)
    baseline=decode_image_boxes(boxes.detach(),origin,side,hw)
    refined=decode_image_boxes(out['boxes'],origin,side,hw);assert torch.equal(refined,baseline)
    loss,stats=center_safe_loss(refined,baseline,truth,hw,torch.ones(len(fit),dtype=torch.bool));loss.backward()
    assert torch.isfinite(loss) and model.net[-1].weight.grad.abs().sum()>0
    assert features.grad is None and boxes.grad is None and truth.grad is None
    print('CENTER_METADATA_CACHE_PASS',json.dumps(dict(augmentation_equivalent=valid_checks,flip_and_padding_checked=True,
        rng_unchanged=True,test_decoder_equivalent=True,sequences=len(manifest),cache_samples=len(rows),official_checkpoint_sha256=identity['checkpoint_sha256'],
        gradient_finite=True,upstream_isolated=True,zero_equivalent=True,optimizer_steps=0)))

if __name__=='__main__':main()
