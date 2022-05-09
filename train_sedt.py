#!/usr/bin/env python
# encoding: utf-8
"""
@author: yzr
@file: train_sedt.py
@time: 2020/8/1 10:43
"""
import argparse
import datetime
import inspect
import os
from pprint import pprint
import numpy as np
import torch

from data_utils.SedData import SedData, get_dfs
from torch.utils.data import DataLoader
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from engine import train, evaluate
import config as cfg
from utilities.Logger import create_logger, set_logger
from utilities.Scaler import Scaler
from utilities.utils import SaveBest, collate_fn, back_up_code, EarlyStopping
from utilities.BoxEncoder import BoxEncoder
from utilities.BoxTransforms import get_transforms as box_transforms
from sedt import build_model

def get_parser():
    parser = argparse.ArgumentParser(description="")
    # dataset parameters
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--dataname', default='dcase', choices=['urbansed', 'dcase'])
    parser.add_argument('--synthetic', dest='synthetic', action='store_true', default=True,
                        help="using synthetic labels during training")
    parser.add_argument('--weak', dest='weak', action='store_false', default=True,
                        help="Not using weak labels during training")

    # train parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_weak', default=16, type=int)
    parser.add_argument('--accumrating_gradient_steps', default=1, type=int)
    parser.add_argument('--adjust_lr', action='store_false', default=True)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--eval', action="store_true", help='evaluate existing model')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--epochs_ls', default=400, type=int, help='number of epochs for learning stage')
    parser.add_argument('--checkpoint_epochs', default=0, type=int, help='save model every checkpoint_epochs')
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--fine_tune', action="store_true", default=False)
    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # data augmentation parameters
    parser.add_argument("--mix_up_ratio", type=float, default=0,
                        help="the ratio of data to be mixed up during training")
    parser.add_argument("--time_mask", action="store_true", default=False,
                        help="perform time mask during training")
    parser.add_argument("--freq_mask", action="store_true", default=False,
                        help="perform frequency mask during training")
    parser.add_argument("--freq_shift", action="store_true", default=False,
                        help="perform frequency shift during training")

    # model parameters
    parser.add_argument('--self_sup', dest='self_sup', action='store_true')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    parser.add_argument('--resume', default='', help='resume training from specific model')
    parser.add_argument("--dec_at", action="store_true", default=False, help="add audio tagging branch")
    parser.add_argument("--fusion_strategy", default=[1], nargs='+', type=int)
    parser.add_argument("--pooling", type=str, default=None, choices=('max', 'avg', 'attn', 'weighted_sum'))
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_false', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--idim', default=128, type=int,
                        help="Size of the transformer input")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_false', default=True)
    parser.add_argument('--input_layer', default="linear", type=str,
                        help="input layer type in the transformer")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--epsilon', default=1, type=float)
    parser.add_argument('--alpha', default=1, type=float)
    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--weak_loss_coef', default=1, type=float)
    parser.add_argument('--weak_loss_p_coef', default=1, type=float)
    parser.add_argument('--ce_loss_coef', default=1, type=float)

    parser.add_argument('--info', default=None, type=str)  # experiment information
    parser.add_argument('--back_up', action='store_true', default=False,
                        help="store current code")
    parser.add_argument('--log', action='store_false', default=True,
                        help="generate log file for this experiment")
    return parser


if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    parser = get_parser()
    f_args = parser.parse_args()
    if f_args.eval:
        f_args.epochs = 0
        assert f_args.info, "Don't give the model information to be evaluated"
    if f_args.info is None:
        f_args.info = f"{f_args.dataname}_atloss_{f_args.weak_loss_coef}_atploss_{f_args.weak_loss_p_coef}_enc_{f_args.enc_layers}_pooling_{f_args.pooling}_{f_args.fusion_strategy}"
        if f_args.pretrain:
            f_args.info += "_" + f_args.pretrain
    if f_args.log:
        set_logger(f_args.info)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Sound Event Detection Transformer")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f_args.gpus

    if 'dcase' in f_args.dataname:
        f_args.num_queries=20
    pprint(vars(f_args))
    store_dir = os.path.join(cfg.dir_root, f_args.dataname)
    saved_model_dir = os.path.join(store_dir, "model")
    os.makedirs(saved_model_dir, exist_ok=True)
    if f_args.back_up:
        back_up_code(store_dir, f_args.info)

    # ##############
    # DATA
    # ##############
    dataset = SedData(f_args.dataname, recompute_features=False, compute_log=False)
    dfs = get_dfs(dataset, f_args.dataname)


    # Normalisation per audio or on the full dataset
    add_axis_conv = 0
    scaler = Scaler()
    scaler_path = os.path.join(store_dir, f_args.dataname + ".json")
    if f_args.dataname == 'urbansed':
        label_encoder = BoxEncoder(cfg.urban_classes, seconds=cfg.max_len_seconds)
        transforms = box_transforms(cfg.umax_frames, add_axis=add_axis_conv)
        encod_func = label_encoder.encode_strong_df
        train_data = DataLoadDf(dfs['train'], encod_func, transforms)
        train_data = ConcatDataset([train_data])
    else:
        label_encoder = BoxEncoder(cfg.dcase_classes, seconds=cfg.max_len_seconds)
        transforms = box_transforms(cfg.max_frames, add_axis=add_axis_conv)
        encod_func = label_encoder.encode_strong_df
        weak_data = DataLoadDf(dfs["weak"], encod_func, transforms)
        train_synth_data = DataLoadDf(dfs["synthetic"], encod_func, transforms)
        train_data = ConcatDataset([weak_data, train_synth_data])
    if os.path.isfile(scaler_path):
        logger.info('loading scaler from {}'.format(scaler_path))
        scaler.load(scaler_path)
    else:
        scaler.calculate_scaler(train_data)
        scaler.save(scaler_path)
    logger.debug(f"scaler mean: {scaler.mean_}")


    if f_args.dataname == 'urbansed':
        transforms = box_transforms(cfg.umax_frames, scaler, add_axis_conv, time_mask=f_args.time_mask,
                                    freq_mask=f_args.freq_mask, freq_shift=f_args.freq_shift,)
        transforms_valid = box_transforms(cfg.umax_frames, scaler, add_axis_conv)
        train_data = DataLoadDf(dfs["train"], encod_func, transform=transforms, in_memory=cfg.in_memory)
        eval_data = DataLoadDf(dfs["eval"], encod_func, transform=transforms_valid, return_indexes=True)
        validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)

        train_dataset = [train_data]
        batch_sizes = [f_args.batch_size]
        strong_mask = slice(batch_sizes[0])
        weak_mask = None
    else:
        transforms = box_transforms(cfg.max_frames, scaler, add_axis_conv, time_mask=f_args.time_mask,
                                    freq_mask=f_args.freq_mask, freq_shift=f_args.freq_shift,)
        transforms_valid = box_transforms(cfg.max_frames, scaler, add_axis_conv)
        weak_data = DataLoadDf(dfs["weak"], encod_func, transforms, in_memory=cfg.in_memory)
        train_synth_data = DataLoadDf(dfs["synthetic"], encod_func, transforms, in_memory=cfg.in_memory)
        validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)
        eval_data = DataLoadDf(dfs["eval"], encod_func, transform=transforms_valid, return_indexes=True)
        train_dataset = [train_synth_data, weak_data]
        batch_sizes = [f_args.batch_size-f_args.n_weak, f_args.n_weak]
        weak_mask = slice(batch_sizes[0], f_args.batch_size)
        strong_mask = slice(batch_sizes[0])

    concat_dataset = ConcatDataset(train_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes)
    training_loader = DataLoader(concat_dataset, batch_sampler=sampler, collate_fn=collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=f_args.batch_size, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_data, batch_size=f_args.batch_size, collate_fn=collate_fn)
    validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
    eval_labels_df = dfs["eval"].drop("feature_filename", axis=1)


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
        logger.info('loading the self-supervised model')
        model_fname = os.path.join(saved_model_dir, f_args.pretrain)
        state = torch.load(model_fname, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        # There are no audio query in self-supervised model
        # only load the backbone, encoder, decoder and MLP for box prediction
        load_dict = state['model']['state_dict']
        model_dict["query_embed.weight"][1:, :] = load_dict["query_embed.weight"]
        load_dict = {k: v for k, v in load_dict.items() if (k in model_dict and "class_embed" not in k and "query_embed" not in k)}
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

    fusion_strategy = f_args.fusion_strategy
    best_saver = {}

    for at_m in fusion_strategy:
        best_saver[at_m] = SaveBest("sup")

    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, fusion_strategy=f_args.fusion_strategy,
                                            val_comp="sup", init_patience=cfg.es_init_wait)

    for epoch in range(start_epoch, f_args.epochs):
        model.train()
        if epoch == f_args.epochs_ls:
            logger.info("enter the fine-tuning stage")
            # load the best model of the learning stage
            try:
                model_fname = os.path.join(saved_model_dir, f"{f_args.info}_1_best")
                state = torch.load(model_fname)
                model.load_state_dict(state['model']['state_dict'])
            except:
                logger.info("No best model exists, fine-tune current model")
            # fix the learning rate as 1e-5
            f_args.adjust_lr = False
            f_args.fine_tune = True
            f_args.info += "_ft"

        loss_value = train(training_loader, model, criterion, optim, epoch, f_args.accumrating_gradient_steps,
                           mask_weak=weak_mask, fine_tune=f_args.fine_tune, normalize=f_args.normalize,
                           mask_strong=strong_mask, max_norm=0.1, mix_up_ratio=f_args.mix_up_ratio)
        if f_args.adjust_lr:
            lr_scheduler.step()

        # Update state
        state['model']['state_dict'] = model.state_dict()
        state['optimizer']['state_dict'] = optim.state_dict()
        state['epoch'] = epoch
        # Validation
        model = model.eval()
        logger.info("Metric on validation")
        metrics = evaluate(model, criterion, postprocessors, validation_dataloader, label_encoder, validation_labels_df,
                           at=True, fusion_strategy=fusion_strategy)

        if cfg.save_best:
            for at_m, eb in metrics.items():
                state[f'event_based_f1_{at_m}'] = eb
                if best_saver[at_m].apply(eb):
                    model_fname = os.path.join(saved_model_dir, f"{f_args.info}_{at_m}_best")
                    torch.save(state, model_fname)

                if cfg.early_stopping:
                    if early_stopping_call.apply(eb):
                        logger.warn("EARLY STOPPING")
                        break

        if f_args.checkpoint_epochs > 0 and (epoch + 1) % f_args.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, f"{f_args.info}_{epoch}")
            torch.save(state, model_fname)

    if cfg.save_best or f_args.eval:
        for at_m in fusion_strategy:
            model_fname = os.path.join(saved_model_dir, f"{f_args.info}_{at_m}_best")
            if torch.cuda.is_available():
                state = torch.load(model_fname)
            else:
                state = torch.load(model_fname, map_location=torch.device('cpu'))
            model.load_state_dict(state['model']['state_dict'])
            logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")

            model.eval()
            logger.info("Metric on validation")
            evaluate(model, criterion, postprocessors, validation_dataloader, label_encoder, validation_labels_df,
                     at=True, fusion_strategy=[at_m], cal_seg=True, cal_clip=True)

            logger.info("Metric on eval")
            evaluate(model, criterion, postprocessors, eval_dataloader, label_encoder, eval_labels_df,
                     at=True, fusion_strategy=[at_m], cal_seg=True, cal_clip=True)
