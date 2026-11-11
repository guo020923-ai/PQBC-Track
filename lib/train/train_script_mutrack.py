import os
import random
import numpy as np
# loss function related
from lib.utils.box_ops import giou_loss, GaussWeightedLoss
from torch.nn.functional import l1_loss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
import lib.models
import lib.train.actors
# for import modules
import importlib
from lib import registry


def load_initial_checkpoint(net, checkpoint_path):
    """Load model weights only, leaving the new optimizer state untouched."""
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Initial checkpoint not found: %s" % checkpoint_path)

    print("Loading initial checkpoint from %s" % checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
    non_erc_missing = [key for key in missing_keys if 'box_head.erc.' not in key and not key.startswith('rqc.')]
    if getattr(net, 'rqc', None) is not None:
        expected = {'rqc.' + key for key in net.rqc.state_dict()}
        checkpoint_has_rqc = any(key.startswith('rqc.') for key in state_dict)
        allowed_upgrade_missing = {
            key for key in expected if key.startswith('rqc.identity.')
        }
        valid_rqc_missing = (
            set(missing_keys) == expected
            if not checkpoint_has_rqc
            else set(missing_keys).issubset(allowed_upgrade_missing)
        )
        if not valid_rqc_missing or unexpected_keys:
            raise ValueError('Official initialization mismatch: missing=%s unexpected=%s'
                             % (missing_keys, unexpected_keys))
    if non_erc_missing:
        print("Warning: non-ERC keys missing from initial checkpoint: %s" % non_erc_missing)
    if unexpected_keys:
        print("Warning: unexpected keys in initial checkpoint: %s" % unexpected_keys)
    print("Initial checkpoint loaded; new-module keys initialized: %d" % len(missing_keys))


def configure_trainable_parameters(net, cfg):
    geometry_module_names = (
        'conv_offset',
        'conv_bbox',
        'conv_bbox_grounding',
    )
    if cfg.TRAIN.ERC_ONLY:
        if not cfg.MODEL.HEAD.ERC.ENABLED or net.box_head.erc is None:
            raise ValueError("TRAIN.ERC_ONLY requires MODEL.HEAD.ERC.ENABLED=true")
        for parameter in net.parameters():
            parameter.requires_grad = False
        for parameter in net.box_head.erc.parameters():
            parameter.requires_grad = True
        # ERC-only is stronger than FREEZE_BACKBONE: the complete pretrained
        # baseline (backbone and original head) stays fixed.  Accepting both
        # flags makes that intent explicit in experiment YAML files.
        mode = "ERC-only (entire baseline frozen)"
    elif cfg.TRAIN.FREEZE_BACKBONE:
        for parameter in net.backbone.parameters():
            parameter.requires_grad = False
        mode = "frozen-backbone"
    else:
        mode = "full-model"

    geometry_frozen = 0
    if cfg.TRAIN.FREEZE_HEAD:
        for parameter in net.box_head.parameters():
            parameter.requires_grad = False
        mode += "+frozen-head"
    if cfg.MODEL.RQC.ENABLED and cfg.TRAIN.FREEZE_RQC_BASE:
        if not hasattr(net.rqc, 'identity'):
            raise ValueError(
                'FREEZE_RQC_BASE requires an RQC design with identity ERC'
            )
        for parameter in net.rqc.parameters():
            parameter.requires_grad = False
        for parameter in net.rqc.identity.parameters():
            parameter.requires_grad = True
        mode += "+frozen-rqc-base"
    if not cfg.TRAIN.ERC_ONLY and cfg.TRAIN.FREEZE_BBOX_GEOMETRY:
        if not hasattr(net, 'box_head') or net.box_head is None:
            raise AttributeError(
                "TRAIN.FREEZE_BBOX_GEOMETRY requires net.box_head"
            )
        for module_name in geometry_module_names:
            if not hasattr(net.box_head, module_name):
                raise AttributeError(
                    "TRAIN.FREEZE_BBOX_GEOMETRY requires "
                    "net.box_head.%s" % module_name
                )
            module = getattr(net.box_head, module_name)
            if not isinstance(module, torch.nn.Module):
                raise TypeError(
                    "net.box_head.%s must be a torch.nn.Module, got %s"
                    % (module_name, type(module).__name__)
                )
            module_parameter_count = sum(
                parameter.numel() for parameter in module.parameters()
            )
            if module_parameter_count == 0:
                raise ValueError(
                    "net.box_head.%s has no parameters to freeze" % module_name
                )
            for parameter in module.parameters():
                parameter.requires_grad = False
            geometry_frozen += module_parameter_count
        mode += "+frozen-bbox-geometry"
        print(
            "Frozen bbox geometry modules %s: params=%d"
            % (', '.join(geometry_module_names), geometry_frozen)
        )

    trainable = sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in net.parameters())
    print("Training mode %s: trainable=%d total=%d" % (mode, trainable, total))


def reset_training_seed(settings):
    """Make loader-worker seeds independent of newly added module initialization."""
    if settings.base_seed is None:
        return
    rank_offset = max(settings.local_rank, 0)
    seed = settings.base_seed + rank_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Training RNG reset to seed %d" % seed)

def run(settings):
    settings.description = 'Training script for Mixformer'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    max_epochs = getattr(settings, 'max_epochs', None)
    if max_epochs is not None:
        if int(max_epochs) < 1 or int(max_epochs) > int(cfg.TRAIN.EPOCH):
            raise ValueError(
                'max_epochs must be in [1, YAML TRAIN.EPOCH=%d]'
                % int(cfg.TRAIN.EPOCH)
            )
        cfg.TRAIN.EPOCH = int(max_epochs)
        print(
            'Pilot max_epochs override: %d (YAML file remains unchanged)'
            % cfg.TRAIN.EPOCH
        )

    # update settings based on cfg
    update_settings(settings, cfg)
    settings.save_every_epoch = cfg.TRAIN.SAVE_EVERY_EPOCH

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_list = build_dataloaders(cfg, settings)

    # Create network
    net = registry.MODELS[settings.script_name](cfg).cuda()
    load_initial_checkpoint(net, cfg.TRAIN.INIT_CHECKPOINT)
    configure_trainable_parameters(net, cfg)
    reset_training_seed(settings)

    # wrap networks to distributed one
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
        
    # settings.save_every_epoch = True
    actor = registry.ACTORS[settings.script_name](net, cfg)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    trainer = LTRTrainer(actor, loader_list, optimizer, settings, lr_scheduler, use_amp=False)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
