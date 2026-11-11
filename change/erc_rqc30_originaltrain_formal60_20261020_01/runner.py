"""One engineering admission followed by one fixed full60; no short-fit gate."""
import argparse
from collections import Counter
import gc
import io
import json
import os
import random
import time
import cv2
import numpy as np
import torch
import rqc30_train_contract as c


def release(streams, sampler=None):
    if sampler is not None:
        sampler.close()
    for stream in streams.values():
        stream.close()
    gc.collect()
    torch.cuda.empty_cache()



def engineer(spec, cfg, plan, params):
    # The 72 real observations and all exclusion evidence were locked before GPU use.
    c.verify_implementation()
    assert not (c.HERE / 'engineering_started_01.json').exists(), 'Do not repeat engineering'
    c.write_new(c.HERE / 'engineering_started_01.json', dict(time=time.time(), optimizer_steps=0, maximum_seconds=1545))
    start=time.monotonic()
    locked=json.loads((c.HERE/'engineering_plan_01.json').read_text())
    assert locked['all_lists_fixed_before_forward'] and len(locked['selected'])==3
    random.seed(20261016);np.random.seed(20261016);torch.manual_seed(20261016);torch.cuda.manual_seed_all(20261016)
    fit=c.FitPools(cfg,plan)
    model=c.Model().cuda()
    assert sum(p.numel() for p in model.parameters())==71881
    results=[];total_changes=total_updates=0
    def budget():
        assert time.monotonic()-start<1545,'Engineering wallclock budget exhausted'
    for entry in locked['selected']:
        budget()
        source=entry['source'];clip=c.rebuild(fit,entry);domain=clip['domain']
        assert clip['name'] not in locked['exclude_by_domain'][domain]
        selected_frames=entry['observation_frames'];assert len(selected_frames)==24
        trackers=[c.infrastructure.Collector30(params,domain) for _ in range(3)]
        baseline,zero_source,forced_source=trackers
        attestation=dict(source_sha256=c.pc.SOURCE_SHA,split_sha256=c.pc.SPLIT_SHA,
                         epoch=30,variant='directional_center_guard_joint')
        zero=c.pr.OutputOnlyERC(zero_source,c.Model().cuda(),c.pp.RQC30Provider(attestation))
        forced_model=c.Model().cuda()
        original=forced_model.decision
        def force_nonzero(**inputs):
            prediction=original(**inputs)
            support=prediction['allowed'].clone();support[:,0]=False
            index=support.long().argmax(1)
            index=torch.where(support.any(1),index,torch.zeros_like(index))
            return dict(prediction,index=index,delta=forced_model.actions[index])
        forced_model.decision=force_nonzero
        forced=c.pr.OutputOnlyERC(forced_source,forced_model,c.pp.RQC30Provider(attestation))
        snapshots=[];samples=[];changed=updates=raw_frames=0
        try:
            for tracker in trackers:
                tracker.initialize(clip['initial_image'],dict(init_bbox=clip['initial_box'],language=clip['language']))
                snapshots.append(c.source_snapshot(tracker))
            previous_prompt=baseline.prompt.clone()
            for frame in range(clip['begin']+1,selected_frames[-1]+1):
                budget();image=cv2.imread(str(clip['paths'][frame]));assert image is not None
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                a=baseline.track(image);b=zero.track(image);d=forced.track(image)
                raw_frames+=1
                np.testing.assert_array_equal(a['target_bbox'],b['target_bbox'])
                for tracker in trackers[1:]:
                    assert c.pr.equal(c.pr.history(baseline),c.pr.history(tracker)),'Native/prompt history changed'
                updates+=int(not torch.equal(previous_prompt,baseline.prompt))
                previous_prompt=baseline.prompt.clone()
                if frame not in selected_frames:continue
                assert clip['visible'][frame] and frame>clip['begin']+16
                for key,value in zero.last_inputs.items():
                    torch.testing.assert_close(value,zero_source.last_inputs[key],rtol=0,atol=0)
                    torch.testing.assert_close(value,baseline.last_inputs[key],rtol=0,atol=0)
                changed+=int(not np.array_equal(np.asarray(a['target_bbox']).astype(int),np.asarray(d['target_bbox']).astype(int)))
                # GT is read AFTER causal forward; no fitting, selection or optimizer.
                truth=clip['annotations'][frame].detach().clone()[None].cuda().double()
                fixed=torch.tensor([a['target_bbox']],device='cuda',dtype=torch.double)
                hw=torch.tensor([image.shape[:2]],device='cuda',dtype=torch.double)
                inputs={k:v.detach().clone() for k,v in zero.last_inputs.items()}
                with torch.no_grad():
                    train_pred=model(**inputs);deploy_pred=zero.last_prediction
                    for key in train_pred:
                        torch.testing.assert_close(train_pred[key],deploy_pred[key],rtol=0,atol=0)
                    target=c.pg.action_targets(fixed,truth,hw,model.actions)
                    torch.testing.assert_close(inputs['executed_delta'].double(),target['executed_delta'],rtol=1e-6,atol=1e-7)
                    assert torch.equal(target['gain'],torch.minimum(target['float_gain'],target['integer_gain']))
                samples.append(dict(inputs=inputs,fixed=fixed,truth=truth,hw=hw,meta=dict(frame=frame,prompt=baseline.prompt.clone())))
            assert len(samples)==24 and changed>0,'Missing required observations or nonzero saved-box case'
            for tracker,snapshot in zip(trackers,snapshots):c.assert_unchanged(tracker,snapshot)
        finally:
            for tracker in trackers:tracker.close()
            del zero,forced,forced_model,baseline,zero_source,forced_source,trackers,tracker
            gc.collect();torch.cuda.empty_cache()
        # Gradient check uses the FIRST 8 of the SAME locked observations, not new selections.
        budget()
        inputs,fixed,truth,hw=c.collate(samples[:8])
        inputs={k:v.detach().clone().requires_grad_(True) if v.is_floating_point() else v.detach().clone() for k,v in inputs.items()}
        truth=truth.detach().clone().requires_grad_(True)
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.train();loss,detail,boxes,pred=c.objective(model,inputs,fixed,truth,hw)
        assert pred['index'].eq(0).all() and torch.equal(boxes,fixed)
        loss.backward();grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        assert np.isfinite(grad_norm) and grad_norm>0
        assert all(v.grad is None for v in inputs.values()) and truth.grad is None
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
        model.zero_grad(set_to_none=True)
        # Original sampler serial/graph parity: first FOUR predetermined observation frames.
        stream=c.Stream(fit,source,params,random.Random(1),spec,forced_clips=[clip])
        graph_snapshot=c.source_snapshot(stream.collector)
        sampler=c.GraphSampler({source:stream})
        try:
            budget();graph=sampler.collect([(2,source),(2,source)])
            assert len(graph)==4 and [r['meta']['frame'] for r in graph]==selected_frames[:4]
            for serial,accelerated in zip(samples[:4],graph):
                for key in ('fixed','truth','hw'):torch.testing.assert_close(serial[key],accelerated[key],rtol=0,atol=0)
                torch.testing.assert_close(serial['meta']['prompt'],accelerated['meta']['prompt'],rtol=0,atol=0)
                for key in serial['inputs']:
                    torch.testing.assert_close(serial['inputs'][key],accelerated['inputs'][key],rtol=0,atol=0)
            c.assert_unchanged(stream.collector,graph_snapshot)
        finally:release({source:stream},sampler)
        model.eval()
        results.append(dict(domain=domain,source=source,sequence=entry['sequence'],observations=24,
            raw_trajectory_frames=raw_frames,forced_integer_changes=changed,prompt_updates=updates,
            gradient_observations=8,gradient_norm=grad_norm,objective=detail,graph_serial_observations=4))
        total_changes+=changed;total_updates+=updates
        del samples,inputs,fixed,truth,hw,graph
        budget()
    assert len(results)==3 and total_changes>0 and total_updates>0,'No actual prompt update boundary'
    c.write_new(c.HERE/'dependency_manifest_01.json',c.dependencies())
    identity=c.binding();c.verify()
    # In-memory schema fixture only; it is never saved as a real trained checkpoint.
    payload=dict(kind=c.KIND,training_identity=identity,epoch=1,optimizer_steps_total=1,model=model.state_dict())
    memory=io.BytesIO();torch.save(payload,memory);memory.seek(0)
    loaded=torch.load(memory,map_location='cpu');c.validate_checkpoint(loaded,model,identity)
    for bad in (dict(loaded,kind='old_ERC'),dict(loaded,training_identity={}),dict(loaded,diagnostic_only=True)):
        try:c.validate_checkpoint(bad,model,identity)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Wrong checkpoint identity accepted')
    probe=c.Statistics();probe.add(dict(valid_samples=2,integer_improved_frames=1,loss=3.))
    probe.add(dict(valid_samples=2,integer_improved_frames=0,loss=1.))
    assert probe.result()['loss']==2. and probe.result()['integer_improved_frames']==1
    report=dict(engineering_pass=True,optimizer_steps=0,model_parameters=71881,real_observations=72,
        per_domain=results,train_deploy_predictions_bitwise_equal=True,source_input_GT_isolation=True,
        fixed_RQC_zero_equivalence=True,nonzero_native_history_prompt_exact=True,
        graph_serial_exact=True,strict_checkpoint_checked=True,monitor_smoke_observations=0,
        confirmation_forward_evaluations=0,official_test_forward_evaluations=0,identity=identity,
        all_lists_fixed_before_forward=True,empirical_gain_proven=False,seconds=time.monotonic()-start)
    budget();c.write_new(c.HERE/'preflight_01.json',report)
    print('BALANCED_PREFLIGHT_PASS',json.dumps(report),flush=True)


def train(spec, cfg, plan, params):
    policy = c.verify()
    ready = json.loads((c.HERE / 'preflight_01.json').read_text())
    identity = c.binding()
    assert ready['engineering_pass'] and ready['optimizer_steps'] == 0 and ready['identity'] == identity
    assert (c.HERE / 'launch_once.lock').exists()
    output = c.HERE / 'train60_01'
    output.mkdir(exist_ok=False)
    c.write_new(output / 'protocol.json', dict(training_identity=identity, policy=policy,
                preflight_sha256=c.digest(c.HERE / 'preflight_01.json'),
                efficacy_not_proven_at_launch=True, no_automatic_test=True))
    seed = policy['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fit = c.FitPools(cfg, plan)
    model = c.Model().cuda()
    assert sum(p.numel() for p in model.parameters()) == 71881
    assert model.curve[-1].weight.eq(0).all() and model.curve[-1].bias[:2].eq(0).all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=policy['LR'], weight_decay=policy['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    streams, snapshots, sampler = {}, {}, None
    total_steps, start, history = 0, time.monotonic(), []
    try:
        for source in fit.vl:
            streams[source] = c.Stream(fit, source, params, random.Random(seed + source), spec)
            snapshots[source] = c.source_snapshot(streams[source].collector)
        sampler = c.GraphSampler(streams)
        with (output / 'stats.jsonl').open('x') as log:
            for epoch in range(1, 61):
                ep = time.monotonic()
                model.train()
                aggregate = c.Statistics()
                domains = {i: c.Statistics() for i in fit.vl}
                histogram, steps, maximum_grad = [0] * 25, 0, 0.
                tickets = fit.epoch_plan(30000, random.Random(seed + epoch))
                counts = Counter(flag for flag, _ in tickets)
                for source, stream in streams.items():
                    stream.rng = random.Random(seed + epoch * 100 + source)
                    stream.clip = None
                frames_before = sampler.forward_frames
                for offset in range(0, 30000, 16):
                    rows = sampler.collect(tickets[offset:offset + 16])
                    if not rows:
                        continue
                    assert all(r['meta']['name'] not in fit.blocked[r['meta']['source']] for r in rows)
                    inputs, fixed, truth, hw = c.collate(rows)
                    inputs = c.inputs_from_rows(inputs, fixed, hw, model.actions)
                    optimizer.zero_grad(set_to_none=True)
                    loss, detail, _, pred = c.objective(model, inputs, fixed, truth, hw)
                    assert torch.isfinite(loss)
                    loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
                    assert np.isfinite(norm)
                    optimizer.step()
                    steps += 1
                    total_steps += 1
                    aggregate.add(detail)
                    maximum_grad = max(maximum_grad, norm)
                    histogram = [a + b for a, b in zip(histogram, torch.bincount(pred['index'].cpu(), minlength=25).tolist())]
                    with torch.no_grad():
                        for source in fit.vl:
                            mask = torch.tensor([r['meta']['source'] == source for r in rows], device='cuda')
                            if mask.any():
                                subset = {k: v[mask] for k, v in pred.items()}
                                domains[source].add(c.metrics(subset, fixed[mask], truth[mask], hw[mask], model.actions))
                    if total_steps == 1:
                        assert len(optimizer.state) > 0
                        assert any(not torch.equal(v.detach().cpu(), torch.zeros_like(v.cpu())) for v in [model.curve[-1].weight])
                        c.write_new(c.HERE / 'actual_optimizer_step_01.json', dict(process=c.process(os.getpid()),
                                    optimizer_steps_total=1, optimizer_state_parameters=len(optimizer.state),
                                    readout_nonzero=True, loss=float(loss.detach()), gradient=norm,
                                    training_identity=identity, seconds=time.monotonic() - start))
                    if offset % 1600 == 0:
                        print('BALANCED_BATCH', json.dumps(dict(epoch=epoch, tickets=offset + 16,
                              optimizer_steps=steps, optimizer_steps_total=total_steps,
                              LR=optimizer.param_groups[0]['lr'], gradient=norm,
                              seconds=time.monotonic() - ep, **detail)), flush=True)
                assert aggregate.total['valid_samples'] == 2 * counts[2] and steps > 0
                for source, stream in streams.items():
                    c.assert_unchanged(stream.collector, snapshots[source])
                train_seconds = time.monotonic() - ep
                validation = dict(role='disabled_no_added_holdout', evaluated=False, observations=0)
                c.verify()
                assert c.binding() == identity
                scheduler.step()
                row = dict(epoch=epoch, train=dict(aggregate.result(), optimizer_steps=steps,
                           optimizer_steps_total=total_steps, sampled_tickets=30000, modality_counts=dict(counts),
                           gradient_max=maximum_grad, action_histogram=histogram,
                           domain_statistics={fit.datasets[i].get_name(): s.result() for i, s in domains.items()},
                           causal_frames=sampler.forward_frames - frames_before, seconds=train_seconds),
                           monitor=validation, confirmation_evaluated=False, source_unchanged=True,
                           local_metrics_are_AUC=False, seconds=time.monotonic() - ep,
                           next_lr=optimizer.param_groups[0]['lr'])
                path = output / ('BalancedSecondOrderEvidenceERC_ep%04d.pth.tar' % epoch)
                payload = dict(kind=c.KIND, epoch=epoch, training_identity=identity,
                               model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                               optimizer_steps_total=total_steps, rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                               rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all())
                with path.open('xb') as f:
                    torch.save(payload, f)
                    f.flush()
                    os.fsync(f.fileno())
                history.append(row)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
                print('BALANCED_EPOCH', json.dumps(row), flush=True)
        assert [r['epoch'] for r in history] == list(range(1, 61))
        final = output / 'BalancedSecondOrderEvidenceERC_ep0060.pth.tar'
        c.validate_checkpoint(torch.load(final, map_location='cpu'), c.Model(), identity)
        c.verify()
        c.write_new(output / 'completed_formal60.json', dict(full_60_epochs=True, epochs=60,
                    optimizer_steps_total=total_steps, checkpoint=str(final), checkpoint_sha256=c.digest(final),
                    training_identity=identity, seconds=time.monotonic() - start,
                    confirmation_evaluations=0, official_test_evaluations=0,
                    automatic_test=False, automatic_retrain=False))
        print('BALANCED_COMPLETE', json.dumps(dict(epochs=60, optimizer_steps_total=total_steps,
              seconds=time.monotonic() - start)), flush=True)
    finally:
        release(streams, sampler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['engineer', 'train'], required=True)
    mode = parser.parse_args().mode
    os.chdir(c.ROOT)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    assert torch.cuda.device_count() == 1 and not torch.distributed.is_initialized()
    assert int(os.environ.get('WORLD_SIZE', '1')) == 1 and int(os.environ.get('LOCAL_RANK', '-1')) == -1
    c.idle()
    assert c.digest(c.SOURCE) == c.pc.SOURCE_SHA and c.digest(c.SPLIT) == c.pc.SPLIT_SHA
    spec, cfg, plan, params = c.settings()
    try:
        (engineer if mode == 'engineer' else train)(spec, cfg, plan, params)
    except Exception as error:
        c.write_new(c.HERE / (mode + '_failed_01.json'), dict(error=repr(error), automatic_retry=False, time=time.time()))
        raise


if __name__ == '__main__':
    main()
