import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import (Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, TNL2K, TNL2Ktest,
                               VisualGenome, OTB99, Object365, RefCOCOSeq, Lasotext, Lasot_test, WebUAV)
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.scheduler import WarmupMultiStepLR


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE,
                          'grounding': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.separate_grad_clip = cfg.TRAIN.SEPARATE_GRAD_CLIP
    settings.head_grad_clip_norm = cfg.TRAIN.HEAD_GRAD_CLIP_NORM
    settings.erc_grad_clip_norm = cfg.TRAIN.ERC_GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "LASOTEXT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "REFCOCOG_val", 
                        "COCO17", "VID", "TRACKINGNET", "TNL2K", "TNL2K_test", "VisualGenome", "OTB99", "OTB99_test",
                        "Object365", "REFCOCOG", "LASOT_test", "WEBUAV"]
        if name == "LASOT_test":
            datasets.append(Lasot_test(settings.env.lasot_dir, split='test', image_loader=image_loader))
        if name == "TNL2K_test":
            datasets.append(TNL2Ktest(settings.env.tnl2k_test_dir, split='test', image_loader=image_loader))
        if name == "REFCOCOG":
            datasets.append(RefCOCOSeq(settings.env.refcoco_dir, split='train', image_loader=image_loader))
        if name == "REFCOCOG_val":
            datasets.append(RefCOCOSeq(settings.env.refcoco_dir, split='val', image_loader=image_loader))
        if name == "Object365":
            datasets.append(Object365(settings.env.object365_dir, split='train', image_loader=image_loader))
        if name == "VisualGenome":
            datasets.append(VisualGenome(settings.env.visualgenome_dir, split='train', image_loader=image_loader))
        if name == "WEBUAV":
            datasets.append(WebUAV(settings.env.webuav_dir, split='train', image_loader=image_loader))
        if name == "OTB99":
            datasets.append(OTB99(settings.env.otb99_dir, split='train', image_loader=image_loader))
        if name == "OTB99_test":
            datasets.append(OTB99(settings.env.otb99_dir, split='test', image_loader=image_loader))
        if name == "TNL2K":
            datasets.append(TNL2K(settings.env.tnl2k_dir, split='train', image_loader=image_loader))
        if name == "LASOTEXT":
            datasets.append(Lasotext(settings.env.lasotext_dir, split='train', image_loader=image_loader))
        if name == "LASOT":
            datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "COCO17":
            datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.ToGrayscale(probability=0.05))
    
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    
    transform_grounding = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    dynamic_cls = getattr(cfg.TRAIN, "DYNAMIC_CLS", False)
    gaussian_iou = getattr(cfg.TRAIN, "GAUSSIAN_IOU", 0.3)
    center_jitter_factor_grounding = getattr(cfg.DATA.SEARCH, "CENTER_JITTER_GROUNDING", 3.5)
    scale_jitter_factor_grounding = getattr(cfg.DATA.SEARCH, "SCALE_JITTER_GROUNDING", 0.5)
    sort_vl_search_frames = (
        cfg.MODEL.HEAD.ERC.ENABLED
        and cfg.MODEL.HEAD.ERC.DESIGN in (
            'cross_domain_global_regional_reliability_erc_v1',
            'target_reference_trajectory_safe_dense_erc_dr2',
            'target_reference_realized_action_dense_erc_dr3',
        )
    )

    data_processing_train = processing.TrackProcessing(search_area_factor=search_area_factor,
                                                           output_sz=output_sz,
                                                           center_jitter_factor=settings.center_jitter_factor,
                                                           scale_jitter_factor=settings.scale_jitter_factor,
                                                           mode='sequence',
                                                           transform=transform_train,
                                                           grounding_transform=transform_grounding,
                                                           joint_transform=transform_joint,
                                                           settings=settings,
                                                           train_score=train_score,
                                                           dynamic_cls=dynamic_cls,
                                                           gaussian_iou=gaussian_iou,
                                                           center_jitter_factor_grounding=center_jitter_factor_grounding,
                                                           scale_jitter_factor_grounding=scale_jitter_factor_grounding)

    data_processing_val = processing.TrackProcessing(search_area_factor=search_area_factor,
                                                         output_sz=output_sz,
                                                         center_jitter_factor=settings.center_jitter_factor,
                                                         scale_jitter_factor=settings.scale_jitter_factor,
                                                         mode='sequence',
                                                         transform=transform_val,
                                                         grounding_transform=transform_grounding,
                                                         joint_transform=transform_joint,
                                                         settings=settings,
                                                         train_score=train_score,
                                                         dynamic_cls=dynamic_cls,
                                                         gaussian_iou=gaussian_iou,
                                                         center_jitter_factor_grounding=center_jitter_factor_grounding,
                                                         scale_jitter_factor_grounding=scale_jitter_factor_grounding)


    dataset_train = sampler.GroundingAndTrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                                        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                                        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                                        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                                        num_template_frames=settings.num_template, processing=data_processing_train,
                                                        frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5, mode=cfg.TRAIN.MODE,
                                                        bert_path=cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH, grounding_ratio=cfg.TRAIN.GROUNDING_RATIO,
                                                        vl_ratio=cfg.TRAIN.VL_RATIO,
                                                        sort_vl_search_frames=sort_vl_search_frames)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    loader_list = [loader_train]
    # # Validation samplers and loaders
    dataset_val_track = sampler.GroundingAndTrackingSampler(datasets=names2datasets(cfg.DATA.VALTRACK.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VALTRACK.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VALTRACK.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5, mode="tracking_test",
                                          bert_path=cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH)
    val_sampler_track = DistributedSampler(dataset_val_track) if settings.local_rank != -1 else None
    loader_val_track = LTRLoader(f"tr_{cfg.DATA.VALTRACK.DATASETS_NAME[0]}", dataset_val_track, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler_track,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)
    loader_list.append(loader_val_track)
    
    # Validation samplers and loaders
    for dataset_name in cfg.DATA.VAL.DATASETS_NAME:
        dataset_val = sampler.GroundingAndTrackingSampler(datasets=names2datasets([dataset_name], settings, opencv_loader),
                                            p_datasets=None,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_val,
                                            frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5, mode="grounding_test",
                                            bert_path=cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH)
        val_sampler_grounding = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val_grounding = LTRLoader(f'gr_{dataset_name}', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=False, stack_dim=1, sampler=val_sampler_grounding,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)
        loader_list.append(loader_val_grounding)

    # # Validation samplers and loaders
    dataset_val_vl = sampler.GroundingAndTrackingSampler(datasets=names2datasets(cfg.DATA.VALVL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VALVL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VALVL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5, mode="vl_test",
                                          bert_path=cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH,
                                          sort_vl_search_frames=sort_vl_search_frames)
    val_sampler_vl = DistributedSampler(dataset_val_vl) if settings.local_rank != -1 else None
    loader_val_vl = LTRLoader(f"vl_{cfg.DATA.VALVL.DATASETS_NAME[0]}", dataset_val_vl, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler_vl,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)
    loader_list.append(loader_val_vl)
    
    return loader_list


def get_optimizer_scheduler(net, cfg):
    head_params = [
        p for n, p in net.named_parameters()
        if ("backbone" not in n) and ("box_head.erc" not in n)
        and not n.startswith('rqc.') and not n.startswith('module.rqc.')
        and p.requires_grad
    ]
    erc_params = [
        p for n, p in net.named_parameters()
        if ("box_head.erc" in n) and p.requires_grad
    ]
    rqc_params = [
        p for n, p in net.named_parameters()
        if (n.startswith('rqc.') or n.startswith('module.rqc.')) and p.requires_grad
    ]
    backbone_params = [
        p for n, p in net.named_parameters()
        if ("backbone" in n) and p.requires_grad
    ]
    param_dicts = []
    if head_params:
        head_lr = cfg.TRAIN.HEAD_LR if cfg.TRAIN.HEAD_LR is not None else cfg.TRAIN.LR
        param_dicts.append({"params": head_params, "lr": head_lr, "name": "head"})
    if erc_params:
        erc_lr = cfg.TRAIN.ERC_LR if cfg.TRAIN.ERC_LR is not None else cfg.TRAIN.LR
        param_dicts.append({"params": erc_params, "lr": erc_lr, "name": "erc"})
    if cfg.MODEL.RQC.ENABLED:
        if not rqc_params or cfg.TRAIN.RQC_LR is None or cfg.TRAIN.RQC_LR <= 0:
            raise ValueError('Enabled RQC requires its own positive RQC_LR and parameter group')
        param_dicts.append({'params': rqc_params, 'lr': cfg.TRAIN.RQC_LR, 'name': 'rqc'})
    if backbone_params:
        param_dicts.append({
            "params": backbone_params,
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            "name": "backbone",
        })
    if not param_dicts:
        raise ValueError("No trainable parameters found")
    # for n, p in net.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    elif cfg.TRAIN.SCHEDULER.TYPE == "WarmMstep":
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                         gamma=cfg.TRAIN.SCHEDULER.GAMMA, warmup_iters=cfg.TRAIN.SCHEDULER.WARM_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "CosineAnnealingLR":
        cosine_horizon = (
            cfg.TRAIN.SCHEDULER.T_MAX
            if cfg.TRAIN.SCHEDULER.T_MAX is not None
            else cfg.TRAIN.EPOCH
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cosine_horizon, eta_min=0, last_epoch=-1
        )
    else:
        raise ValueError("Unsupported scheduler")
    for group in optimizer.param_groups:
        parameter_count = sum(parameter.numel() for parameter in group['params'])
        print("Optimizer group %s: params=%d lr=%g" % (
            group.get('name', 'unnamed'), parameter_count, group['lr']))
    return optimizer, lr_scheduler
