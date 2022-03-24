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
import sys
import time
import shutil
from pprint import pprint
import math
import numpy as np
import torch
from torch import nn

from data_utils.SedData import SedData
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler, RandomSampler, BatchSampler
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler, data_prefetcher
from evaluation_measures import compute_metrics, get_sedt_predictions, audio_tagging_results
import config as cfg
from utilities.Logger import create_logger, set_logger
from utilities.Scaler import Scaler
from utilities.utils import SaveBest, to_cuda_if_available, EarlyStopping, AverageMeter, MetricLogger, SmoothedValue, \
    collate_fn, is_main_process, reduce_dict
from utilities.BoxEncoder import BoxEncoder
from utilities.BoxTransforms import get_transforms as box_transforms
from sedt import build_model

import pandas as pd
def train(train_loader, model, criterion, optimizer, c_epoch, accumrating_gradient_steps,
          mask_weak=None, mask_strong=None, fine_tune=False, normalize=False, max_norm=0.1):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    metric_logger: MetricLogger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    log.debug("Nb batches: {}".format(len(train_loader)))
    end = time.time()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    prefetcher = data_prefetcher(train_loader)
    batch_input, target = prefetcher.next()
    i = -1
    while batch_input is not None:
        i += 1
        # measure data loading time
        data_time.update(time.time() - end)
        global_step = c_epoch * len(train_loader) + i + 1
        # Outputs
        if 'patches' in target[0]:
            patches = [t['patches'] for t in target]
            patches = torch.stack(patches, dim=0)
            outputs = model(batch_input.decompose(), patches)
        else:
            outputs = model(batch_input)

        loss_dict, _ = criterion(outputs, target, mask_weak, mask_strong, fine_tune, normalize)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        losses.backward()
        if (i + 1) % accumrating_gradient_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_input, target = prefetcher.next()
    # gather the stats from all processes
    log.info("Epoch:{} data_time:{:.3f}({:.3f}) batch_time:{:.3f}({:.3f})".
          format(c_epoch, data_time.val, data_time.avg, batch_time.val, batch_time.avg))
    log.info("Train averaged stats: \n" + str(metric_logger))
    return loss_value


def get_dfs(desed_dataset, dataname, unlabel_data=False):
    if dataname == 'urbansed':
        train_df = desed_dataset.initialize_and_get_df(cfg.urban_train_tsv)
        valid_df = desed_dataset.initialize_and_get_df(cfg.urban_valid_tsv)
        eval_df = desed_dataset.initialize_and_get_df(cfg.urban_eval_tsv)

        data_dfs = {
            "train": train_df,
            "validation": valid_df,
            "eval": eval_df
        }
    else:
        weak_df = desed_dataset.initialize_and_get_df(cfg.weak)
        synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic)
        validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir)
        eval_df = desed_dataset.initialize_and_get_df(cfg.eval_desed)
        data_dfs = {
            "weak": weak_df,
            "synthetic": synthetic_df,
            "validation": validation_df,
            "eval": eval_df
        }
        if unlabel_data:
            unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel)
            dcase2018_task5 = desed_dataset.initialize_and_get_df(cfg.dcase2018_task5)
            data_dfs["unlabel"] = unlabel_df.append(dcase2018_task5,ignore_index=True)
            # data_dfs["unlabel"] = unlabel_df

    return data_dfs






if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    parser = argparse.ArgumentParser(description="")
    # dataset parameters
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--dataname', default='urbansed', choices=['urbansed', 'dcase'])
    parser.add_argument("-syn", '--synthetic', dest='synthetic', action='store_true', default=True,
                        help="using synthetic labels during training")
    parser.add_argument("-w", '--weak', dest='weak', action='store_false', default=True,
                        help="Not using weak labels during training")


    # train parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_weak', default=64, type=int)
    parser.add_argument('--accumrating_gradient_steps', default=1, type=int)
    parser.add_argument('--adjust_lr', action='store_false', default=True)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--eval', action="store_true", help='evaluate existing model')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--epochs_ls', default=400, type=int, help='number of epochs for learning stage')
    parser.add_argument('--checkpoint_epochs', default=0, type=int, help='save model every checkpoint_epochs')
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--fine_tune', action="store_true", default=False)
    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # model parameters
    parser.add_argument('--self_sup', action='store_true', default=False)
    parser.add_argument('--feature_recon', action='store_true', default=False)
    parser.add_argument('--query_shuffle', action='store_true', default=False)
    parser.add_argument('--fixed_patch_size', default=False, action='store_true',
                        help="use fixed size for each patch")
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
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_patches', default=10, type=int,
                        help="number of query patches")
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
    f_args = parser.parse_args()
    if f_args.eval:
        f_args.epochs = 0
        assert f_args.info, "Don't give the model information to be evaluated"
    if f_args.info is None:
        f_args.info = f"{f_args.dataname}_atloss_{f_args.weak_loss_coef}_atploss_{f_args.weak_loss_p_coef}_enc_{f_args.enc_layers}_pooling_{f_args.pooling}_{f_args.fusion_strategy}"
    if f_args.self_sup:
        f_args.info = "pretrain_" + f_args.info
    if f_args.log:
        set_logger(f_args.info)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Sound Event Detection Transformer")
    logger.info(f"Starting time: {datetime.datetime.now()}")

    os.environ["CUDA_VISIBLE_DEVICES"] = f_args.gpus


    if f_args.self_sup:
        f_args.unlabel="unlabel"
        f_args.dataname = "dcase"
        f_args.lr_backbone = 0
        f_args.feature_recon = True
    if 'dcase' in f_args.dataname:
        f_args.num_queries=20
    pprint(vars(f_args))
    store_dir = os.path.join(cfg.dir_root, f_args.dataname)
    current_time = time.strftime("%Y%m%d-%H%M", time.localtime(time.time()))
    # ######################
    # back-up current code
    # ######################
    if f_args.back_up:
        saved_code_dir = os.path.join(store_dir, 'code')
        # code file path
        cur_code_dir = os.path.join(saved_code_dir, f'{current_time}_{f_args.info}')
        if os.path.exists(cur_code_dir):
            shutil.rmtree(cur_code_dir)
        os.makedirs(cur_code_dir)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(this_dir):
            old_path = os.path.join(this_dir, filename)
            if 'log' in old_path:
                continue
            new_path = os.path.join(cur_code_dir, filename)
            if os.path.isdir(old_path):
                shutil.copytree(old_path, new_path)
            else:
                shutil.copyfile(old_path, new_path)
    ###########################################################

    saved_model_dir = os.path.join(store_dir, "model")
    # os.makedirs(store_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    # ##############
    # DATA
    # ##############
    dataset = SedData(f_args.dataname, recompute_features=False, compute_log=False)
    dfs = get_dfs(dataset, f_args.dataname, unlabel_data=f_args.self_sup)


    # Normalisation per audio or on the full dataset
    add_axis_conv = 0
    scaler = Scaler()
    if f_args.self_sup:
        scaler_path = os.path.join(store_dir, f_args.dataname + "_sp.json")
    else:
        scaler_path = os.path.join(store_dir, f_args.dataname + ".json")
    if f_args.dataname == 'urbansed':
        label_encoder = BoxEncoder(cfg.urban_classes, seconds=cfg.max_len_seconds)
        transforms = box_transforms(cfg.umax_frames, add_axis=add_axis_conv)
        encod_func = label_encoder.encode_strong_df
        train_data = DataLoadDf(dfs['train'], encod_func, transforms)
        train_data = ConcatDataset([train_data])
    else:
        if f_args.self_sup:
            num_class = 1
        else:
            num_class = cfg.dcase_classes
        label_encoder = BoxEncoder(num_class, seconds=cfg.max_len_seconds, generate_patch=f_args.self_sup)
        transforms = box_transforms(cfg.max_frames, add_axis=add_axis_conv, crop_patch=f_args.self_sup,
                                    fixed_patch_size=f_args.fixed_patch_size)
        encod_func = label_encoder.encode_strong_df
        if f_args.self_sup:
            unlabel_data = DataLoadDf(dfs["unlabel"], label_encoder.encode_unlabel, transforms,
                                      num_patches=f_args.num_patches, fixed_patch_size=f_args.fixed_patch_size)
            train_data = unlabel_data
        else:
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
        transforms = box_transforms(cfg.umax_frames, scaler, add_axis_conv)
        train_data = DataLoadDf(dfs["train"], encod_func, transform=transforms, in_memory=cfg.in_memory)
        eval_data = DataLoadDf(dfs["eval"], encod_func, transform=transforms, return_indexes=True)
        validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms, return_indexes=True)

        train_dataset = [train_data]
        batch_sizes = [f_args.batch_size]
        strong_mask = slice(batch_sizes[0])
        weak_mask = None
    else:
        transforms = box_transforms(cfg.max_frames, scaler, add_axis_conv, crop_patch=f_args.self_sup,
                                    fixed_patch_size=f_args.fixed_patch_size)
        transforms_valid = box_transforms(cfg.max_frames, scaler, add_axis_conv)
        if f_args.self_sup:
            unlabel_data = DataLoadDf(dfs["unlabel"], label_encoder.encode_unlabel, transforms,
                                      num_patches=f_args.num_patches, fixed_patch_size=f_args.fixed_patch_size)
            train_dataset = unlabel_data
            strong_mask = slice(f_args.batch_size)
            weak_mask = slice(f_args.batch_size)
        else:
            weak_data = DataLoadDf(dfs["weak"], encod_func, transforms, in_memory=cfg.in_memory)
            train_synth_data = DataLoadDf(dfs["synthetic"], encod_func, transforms, in_memory=cfg.in_memory)
            validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)
            eval_data = DataLoadDf(dfs["eval"], encod_func, transform=transforms_valid, return_indexes=True)
            train_dataset = [train_synth_data, weak_data]
            batch_sizes = [f_args.batch_size-f_args.n_weak, f_args.n_weak]
            weak_mask = slice(batch_sizes[0], f_args.batch_size)
            strong_mask = slice(batch_sizes[0])



    if f_args.self_sup:
        if torch.cuda.device_count() > 1:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
        train_sampler = BatchSampler(train_sampler, f_args.batch_size, drop_last=True)
        training_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    else:
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
        logger.info('Initialized from the pre-training model')
        model_fname = os.path.join(saved_model_dir, f_args.pretrain)
        state = torch.load(model_fname, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        if "backbone" not in model_fname:
            # There are no audio query in self-supervised model
            # only load the backbone, encoder, decoder and MLP for box prediction
            load_dict = state['model']['state_dict']
            logger.info('loading the self-supervised model')
            model_dict["query_embed.weight"][1:, :] = load_dict["query_embed.weight"]
            load_dict = {k: v for k, v in load_dict.items() if (k in model_dict and "class_embed" not in k and "query_embed" not in k)}
        else:
            load_dict = state['model']
            logger.info('loading the ptrtrained backbone for self-supervised training')
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

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        model = DistributedDataParallel(model.cuda(),find_unused_parameters=True)
    else:
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
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup", init_patience=cfg.es_init_wait)

    for epoch in range(start_epoch, f_args.epochs):
        model.train()
        if epoch == f_args.epochs_ls:
            logger.info("enter the fine-tuning stage")
            # load the best model of the learning stage
            try:
                model_fname = os.path.join(saved_model_dir, f"{f_args.info}_2_best")
                state = torch.load(model_fname)
                model.load_state_dict(state['model']['state_dict'])
            except:
                logger.info("No best model exists, fine-tune current model")
            # fix the learning rate as 1e-5
            f_args.adjust_lr = False
            f_args.fine_tune = True

        loss_value = train(training_loader, model, criterion, optim, epoch, f_args.accumrating_gradient_steps,
                           mask_weak=weak_mask, fine_tune=f_args.fine_tune, normalize=f_args.normalize,
                           mask_strong=strong_mask, max_norm=0.1)
        if f_args.adjust_lr:
            lr_scheduler.step()
        # Validation
        model = model.eval()

        # Update state
        if torch.cuda.device_count() > 1:
            state['model']['state_dict'] = model.module.state_dict()
        else:
            state['model']['state_dict'] = model.state_dict()
        state['optimizer']['state_dict'] = optim.state_dict()
        state['epoch'] = epoch

        if f_args.checkpoint_epochs > 0 and (epoch + 1) % f_args.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "pretrained_{}_loss_{}".format(f_args.info, epoch))
            torch.save(state, model_fname)

        # Validation with real data
        if not f_args.self_sup:
            logger.info("\n ### Metric on validation  ### \n")
            audio_tag_dfs, dec_predictions = get_sedt_predictions(model, criterion, postprocessors, validation_dataloader,
                                                                  label_encoder, at=True, fusion_strategy=fusion_strategy)
            if not audio_tag_dfs.empty:
                clip_metric = audio_tagging_results(validation_labels_df, audio_tag_dfs)
                logger.info(f"AT Class-wise clip metrics \n {'='*50} \n {clip_metric}")

            logger.info(f"decoder output \n {'=' * 50}")
            for at_m, dec_pred in dec_predictions.items():
                logger.info(f"Fusion strategy: {at_m}")
                event_macro_f1, event_macro_p, clip_macro_f1 = compute_metrics(dec_pred, validation_labels_df,
                                                                                           cal_seg=False,
                                                                                           cal_clip=False)

                state['event_macro_f1'] = event_macro_f1
                state['clip_macro_f1'] = clip_macro_f1


                if cfg.save_best:
                    if best_saver[at_m].apply(event_macro_f1):
                        model_fname = os.path.join(saved_model_dir, f"{f_args.info}_{at_m}_best")
                        torch.save(state, model_fname)

            if cfg.early_stopping:
                if early_stopping_call.apply(event_macro_f1):
                    logger.warn("EARLY STOPPING")
                    break

    if not f_args.self_sup:
        if cfg.save_best or f_args.eval:
            for at_m in fusion_strategy:
                model_fname = os.path.join(saved_model_dir, f"{f_args.info}_{at_m}_best")
                if torch.cuda.is_available():
                    state = torch.load(model_fname)
                else:
                    state = torch.load(model_fname, map_location=torch.device('cpu'))
                model.load_state_dict(state['model']['state_dict'])
                logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")


                # ##############
                # Validation
                # ##############
                model.eval()
                logger.info("\n ### Metric on validation  ### \n")
                audio_tag_dfs, dec_predictions = get_sedt_predictions(model,
                                                                      criterion,
                                                                      postprocessors,
                                                                      validation_dataloader,
                                                                      label_encoder,
                                                                      at=True,
                                                                      fusion_strategy=[at_m])

                logger.info(f"decoder output \n {'='*50} ")
                for at_m, pred in dec_predictions.items():
                    logger.info(f"Fusion strategy: {at_m}")
                    compute_metrics(pred, validation_labels_df)
                    if not audio_tag_dfs.empty:
                        clip_metric = audio_tagging_results(validation_labels_df, audio_tag_dfs)
                        logger.info(f"AT Class-wise clip metrics \n {'='*50} \n {clip_metric}")


                ## eval
                logger.info("\n ### Metric on eval ### \n")
                audio_tag_dfs, dec_predictions = get_sedt_predictions(model,
                                                                      criterion,
                                                                      postprocessors,
                                                                      eval_dataloader,
                                                                      label_encoder,
                                                                      at=True,
                                                                      fusion_strategy=[at_m])
                logger.info(f"decoder output \n {'='*50}")
                for at_m, pred in dec_predictions.items():
                    logger.info(f"Fusion strategy: {at_m}")
                    compute_metrics(pred, eval_labels_df)
                    if not audio_tag_dfs.empty:
                        clip_metric = audio_tagging_results(eval_labels_df, audio_tag_dfs)
                        logger.info(f"AT Class-wise clip metrics\n {'='*50} \n {clip_metric}")
