#!/usr/bin/env python
# encoding: utf-8
"""
@author: yzr
@file: train_at.py
@time: 2020/12/1 14:52
"""
import torch
import torch.nn as nn
import inspect

from utilities.metrics import audio_tagging_results
from utilities.Logger import create_logger, set_logger
from data_utils.SedData import SedData
from utilities.FrameEncoder import ManyHotEncoder
from utilities.FrameTransforms import get_transforms
from data_utils.DataLoad import DataLoadDf, data_prefetcher, ConcatDataset
from utilities.Scaler import Scaler
from torch.utils.data import DataLoader
from audio_tag.backbone import build_backbone
from utilities.utils import to_cuda_if_available, SaveBest
import datetime
import pandas as pd
import argparse
import shutil
import config as cfg
from pprint import  pprint
import os


def get_dfs(desed_dataset, dataname):
    if "urban" in dataname:
        train_df = desed_dataset.initialize_and_get_df(cfg.urban_train_tsv)
        valid_df = desed_dataset.initialize_and_get_df(cfg.urban_valid_tsv)
        eval_df = desed_dataset.initialize_and_get_df(cfg.urban_eval_tsv)
        return {"train": train_df,
                "val": valid_df,
                "test": eval_df}
    else:
        synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic)
        validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir)
        weak_df = desed_dataset.initialize_and_get_df(cfg.weak)
        eval_df = desed_dataset.initialize_and_get_df(cfg.eval_desed)
        return {"weak": weak_df,
                "synthetic": synthetic_df,
                "val": validation_df,
                "test": eval_df}


def train(model, train_loader, optim, c_epoch, grad_step, max_norm=0.1):
    loss_func = nn.BCELoss()
    prefetcher = data_prefetcher(train_loader)
    input, targets = prefetcher.next()
    i = -1
    while input is not None:
        output = model(input)
        loss = loss_func(output, targets)

        loss.backward()
        if i % grad_step == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optim.step()
            optim.zero_grad()
        input, targets = prefetcher.next()
    print("Epoch:{} Loss:{} lr:{}".format(c_epoch, loss.item(), optim.param_groups[0]["lr"]))


def evaluate(model, data_loader, decoder):
    logger.info("validation")
    loss_func = nn.BCELoss()
    audio_tag_dfs = pd.DataFrame()
    prefetcher = data_prefetcher(data_loader, return_indexes=True)
    (input, targets), indexes = prefetcher.next()
    i = -1
    while input is not None:
        i += 1
        indexes = indexes.numpy()
        with torch.no_grad():
            output = model(input)
        loss = loss_func(output, targets)

        audio_tags = output
        audio_tags = (audio_tags > 0.5).long()
        for j, audio_tag in enumerate(audio_tags):
            audio_tag_res = decoder(audio_tag)
            audio_tag_res = pd.DataFrame(audio_tag_res, columns=["event_label"])
            audio_tag_res["filename"] = data_loader.dataset.filenames.iloc[indexes[j]]
            audio_tag_res["onset"] = 0
            audio_tag_res["offset"] = 0
            audio_tag_dfs = audio_tag_dfs.append(audio_tag_res)
        (input, targets), indexes = prefetcher.next()
    if "event_labels" in data_loader.dataset.df.columns:
        reformat_df = pd.DataFrame()
        filenames = data_loader.dataset.filenames
        for file in filenames:
            labels = audio_tag_dfs[audio_tag_dfs['filename']==file].event_label.drop_duplicates().to_list()
            labels = ",".join(labels)
            df = pd.DataFrame([[file, labels]], columns=['filename', 'event_labels'])
            reformat_df = reformat_df.append(df)
        return reformat_df
    else:
        return audio_tag_dfs


if __name__ == "__main__":
    torch.manual_seed(2020)

    parser = argparse.ArgumentParser(description="")
    # model param
    parser.add_argument("--pooling", choices=["max", "avg"], default="avg")
    parser.add_argument("--pretrained", action="store_false", default=True)
    ###
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_false', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # train param
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_drop", type=int, default=20)
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--back_up", action="store_true", default=False)
    parser.add_argument("--fix_backbone", action="store_true", default=False)
    # data param
    parser.add_argument('--dataname', default='urbansed', choices=['urbansed', 'dcase'])

    f_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f_args.gpu

    store_dir = os.path.join(cfg.dir_root, f_args.dataname)
    code_dir = os.path.join(store_dir, "code")
    model_dir = os.path.join(store_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"backbone_{f_args.backbone}_{f_args.pooling}"
    if f_args.pretrained:
        model_name += '_pretrained'
    model_path = os.path.join(model_dir, model_name)
    set_logger(model_name)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Audio_Tag_Module")
    logger.info(f"starting time ：{datetime.datetime.now()}")
    pprint(vars(f_args))

    ################
    # code back-up
    ################
    current_time = datetime.datetime.now().strftime('%F_%H%M')
    if f_args.back_up:
        # code file path
        cur_code_dir = os.path.join(code_dir, f'{current_time}_{model_name}')
        if os.path.exists(cur_code_dir):
            shutil.rmtree(cur_code_dir)
        os.makedirs(cur_code_dir)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(this_dir):
            if filename in ['data', 'exp', 'log']:
                continue
            old_path = os.path.join(this_dir, filename)
            new_path = os.path.join(cur_code_dir, filename)
            if os.path.isdir(old_path):
                shutil.copytree(old_path, new_path)
            else:
                shutil.copyfile(old_path, new_path)


    # model
    model = build_backbone(f_args)
    model = to_cuda_if_available(model)
    logger.info(model)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("number of parameters in the model: {}".format(param_num))

    # data preparation
    dataset = SedData(f_args.dataname, recompute_features=False, compute_log=False)
    dfs = get_dfs(dataset, f_args.dataname)
    if "urban" in f_args.dataname:
        encoder = ManyHotEncoder(cfg.urban_classes, n_frames=cfg.umax_frames)
        transformer = get_transforms(cfg.umax_frames, add_axis=0)
    else:
        encoder = ManyHotEncoder(cfg.dcase_classes, n_frames=cfg.max_frames)
        transformer = get_transforms(cfg.max_frames, add_axis=0)

    weak_data = DataLoadDf(dfs["weak"], encoder.encode_weak, transform=transformer)
    syn_data = DataLoadDf(dfs["synthetic"], encoder.encode_weak, transform=transformer)
    train_data = ConcatDataset([weak_data, syn_data])
    scaler = Scaler()
    scaler.calculate_scaler(train_data)

    transformer = get_transforms(cfg.umax_frames if "urbansed" in f_args.dataname else cfg.max_frames, scaler=scaler, add_axis=0)
    weak_data = DataLoadDf(dfs["weak"], encoder.encode_weak, transform=transformer, in_memory=cfg.in_memory)
    syn_data = DataLoadDf(dfs["synthetic"], encoder.encode_weak, transform=transformer, in_memory=cfg.in_memory)
    val_data = DataLoadDf(dfs["val"], encoder.encode_weak, transform=transformer, return_indexes=True)
    test_data = DataLoadDf(dfs["test"], encoder.encode_weak, transform=transformer, return_indexes=True)

    train_loader = DataLoader(ConcatDataset([weak_data, syn_data]), batch_size=f_args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=f_args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=f_args.batch_size, shuffle=False, drop_last=False)

    validation_labels_df = dfs["val"].drop("feature_filename", axis=1)
    test_labels_df = dfs["test"].drop("feature_filename", axis=1)


    optim = torch.optim.Adam(model.parameters(), lr=f_args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                             amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, f_args.lr_drop)
    best_saver = SaveBest("sup")
    # train
    state = {"model": model.state_dict(), "epoch": 0}
    for epoch in range(f_args.nepochs):
        model.train()
        train(model, train_loader, optim, epoch, f_args.grad_steps)
        lr_scheduler.step()
        model = model.eval()


        audio_tag_df = evaluate(model, val_loader, encoder.decode_weak)
        clip_metric = audio_tagging_results(validation_labels_df, audio_tag_df)
        clip_macro_f1 = clip_metric.loc['avg', 'f']
        print("AT Class-wise clip metrics")
        print("=" * 50)
        print(clip_metric)
        # print("clip_macro_metrics:" + f'{clip_metric.values.mean():.3f}')
        state["model"] = model.state_dict()
        state["epoch"] = epoch
        # save best model
        if best_saver.apply(clip_macro_f1):
            torch.save(state, model_path)
    state = torch.load(model_path, map_location=torch.device("cpu") if not torch.cuda.is_available() else None)
    model.load_state_dict(state['model'])
    logger.info(f"testing model of epoch {state['epoch']} at {model_path}")
    model.eval()
    audio_tag_df = evaluate(model, val_loader, encoder.decode_weak)
    clip_metric = audio_tagging_results(validation_labels_df, audio_tag_df)
    clip_macro_f1 = clip_metric.loc['avg', 'f']
    print("AT Class-wise clip metrics on validation set")
    print("=" * 50)
    print(clip_metric)

    audio_tag_df = evaluate(model, test_loader, encoder.decode_weak)
    clip_metric = audio_tagging_results(test_labels_df, audio_tag_df)
    clip_macro_f1 = clip_metric.loc['avg', 'f']
    print("AT Class-wise clip metrics on test set")
    print("=" * 50)
    print(clip_metric)







