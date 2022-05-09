import datetime
import inspect
import os
import numpy as np
import torch

from data_utils.SedData import SedData
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, BatchSampler
from data_utils.DataLoad import DataLoadDf
from engine import train
from train_sedt import get_parser
from utilities.Logger import create_logger, set_logger
from utilities.Scaler import Scaler
from utilities.distribute import is_main_process, init_distributed_mode
from utilities.utils import collate_fn, back_up_code
from utilities.BoxEncoder import BoxEncoder
from utilities.BoxTransforms import get_transforms as box_transforms
from sedt import build_model
import config as cfg


def get_pretrain_data(desed_dataset, extra_data=False):
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel)
    if extra_data:
        dcase2018_task5 = desed_dataset.initialize_and_get_df(cfg.dcase2018_task5)
        unlabel_df = unlabel_df.append(dcase2018_task5,ignore_index=True)
    return unlabel_df


if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)

    parser = get_parser()
    # sp-sedt related parameters
    parser.add_argument('--num_patches', default=10, type=int, help="number of query patches")
    parser.add_argument('--feature_recon', action='store_true', default=False)
    parser.add_argument('--query_shuffle', action='store_true', default=False)
    parser.add_argument('--fixed_patch_size', default=False, action='store_true',
                        help="use fixed size for each patch")
    parser.add_argument('--extra_data', default=False, action='store_true',
                        help="use dcase2018 task5 data to pretrain")
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank',default=0, type=int)
    f_args = parser.parse_args()
    assert f_args.dataname == "dcase", "only support dcase dataset now"
    f_args.lr_backbone = 0
    init_distributed_mode(f_args)
    if f_args.info is None:
        f_args.info = f"pretrain_enc_{f_args.enc_layers}"
    if f_args.feature_recon:
        f_args.info += "_feature_recon"
    if f_args.fixed_patch_size:
        f_args.info += "_fixed_patch_size"
    if f_args.extra_data:
        f_args.extra_data += "_extra_data"
    if f_args.log:
        set_logger(f_args.info)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Self-supervised Pre-training for Sound Event Detection Transformer")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f_args.gpus

    logger.info(vars(f_args))
    store_dir = os.path.join(cfg.dir_root, "dcase")
    saved_model_dir = os.path.join(store_dir, "model")
    os.makedirs(saved_model_dir, exist_ok=True)
    if f_args.back_up:
        back_up_code(store_dir, f_args.info)

    # ##############
    # DATA
    # ##############
    dataset = SedData("dcase", recompute_features=False, compute_log=False)
    unlabel_data = get_pretrain_data(dataset, extra_data=f_args.extra_data)

    # Normalisation per audio or on the full dataset
    add_axis_conv = 0
    scaler = Scaler()
    if f_args.extra_data:
        scaler_path = os.path.join(store_dir, "dcase_sp_bd.json")
    else:
        scaler_path = os.path.join(store_dir, "dcase_sp.json")
    num_class = 1
    label_encoder = BoxEncoder(num_class, seconds=cfg.max_len_seconds, generate_patch=True)
    encod_func = label_encoder.encode_strong_df

    if os.path.isfile(scaler_path):
        logger.info('loading scaler from {}'.format(scaler_path))
        scaler.load(scaler_path)
    else:
        transforms = box_transforms(cfg.max_frames, add_axis=add_axis_conv, crop_patch=f_args.self_sup,
                                    fixed_patch_size=f_args.fixed_patch_size)
        train_data = DataLoadDf(unlabel_data, label_encoder.encode_unlabel, transforms,
                                  num_patches=f_args.num_patches, fixed_patch_size=f_args.fixed_patch_size)
        scaler.calculate_scaler(train_data)
        scaler.save(scaler_path)

    logger.debug(f"scaler mean: {scaler.mean_}")
    transforms = box_transforms(cfg.max_frames, scaler, add_axis_conv, crop_patch=True,
                                fixed_patch_size=f_args.fixed_patch_size)
    train_data = DataLoadDf(unlabel_data, label_encoder.encode_unlabel, transforms,
                              num_patches=f_args.num_patches, fixed_patch_size=f_args.fixed_patch_size)
    strong_mask = slice(f_args.batch_size)
    weak_mask = slice(f_args.batch_size)

    if torch.cuda.device_count() > 1:
        train_sampler = DistributedSampler(train_data)
    else:
        train_sampler = RandomSampler(train_data)
    train_sampler = BatchSampler(train_sampler, f_args.batch_size, drop_last=True)
    training_loader = DataLoader(train_data, batch_sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)


    # ##############
    # Model
    # ##############
    model, criterion, postprocessors = build_model(f_args)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info("number of parameters in the model: {}".format(pytorch_total_params))
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": f_args.lr_backbone,
        },
    ]

    if f_args.pretrain:
        logger.info('loading the ptrtrained backbone for self-supervised training')
        model_fname = os.path.join(saved_model_dir, f_args.pretrain)
        state = torch.load(model_fname, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        load_dict = state['model']
        load_dict = {'backbone.0.' + k: v for k, v in load_dict.items() if
                     ('backbone.0.' + k in model_dict and "class_embed" not in k and "query_embed" not in k)}
        model_dict.update(load_dict)
        model.load_state_dict(model_dict)

    start_epoch = 0
    if f_args.resume:
        model_fname = os.path.join(saved_model_dir, f_args.resume)
        if torch.cuda.is_available():
            state = torch.load(model_fname)
        else:
            state = torch.load(model_fname, map_location=torch.device('cpu'))
        load_dict = state['model']['state_dict']
        model.load_state_dict(load_dict)
        start_epoch = state['epoch']
        logger.info('Resume training form epoch {}'.format(state['epoch']))

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[f_args.gpu])


    optim = torch.optim.AdamW(param_dicts, lr=f_args.lr,
                              weight_decay=f_args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, f_args.lr_drop)
    if f_args.resume:
        optim.load_state_dict(state['optimizer']['state_dict'])

    state = {
        'model': {"name": model.__class__.__name__,
                  'args': '',
                  "kwargs": '',
                  'state_dict': model.state_dict()},

        'optimizer': {"name": optim.__class__.__name__,
                      'args': '',
                      'state_dict': optim.state_dict()},
    }


    for epoch in range(start_epoch, f_args.epochs):
        model.train()

        loss_value = train(training_loader, model, criterion, optim, epoch, f_args.accumrating_gradient_steps,
                           mask_weak=weak_mask, normalize=f_args.normalize, mask_strong=strong_mask, max_norm=0.1)
        if f_args.adjust_lr:
            lr_scheduler.step()
        # Validation
        model = model.eval()

        # Update state
        if is_main_process():
            if torch.cuda.device_count() > 1:
                state['model']['state_dict'] = model.module.state_dict()
            else:
                state['model']['state_dict'] = model.state_dict()
            state['optimizer']['state_dict'] = optim.state_dict()
            state['epoch'] = epoch

            if f_args.checkpoint_epochs > 0 and (epoch + 1) % f_args.checkpoint_epochs == 0:
                model_fname = os.path.join(saved_model_dir, "pretrained_{}_loss_{}".format(f_args.info, epoch))
                torch.save(state, model_fname)