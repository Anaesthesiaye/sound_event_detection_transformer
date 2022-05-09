#!/usr/bin/env python
# encoding: utf-8
"""
@author: yzr
@file: config.py
@time: 2020/7/30 12:06
"""
import logging
import math
import os
import pandas as pd

# save directory
dir_root = "./exp/"

# data
dcase_dir = "./data/dcase2019/"
# # DESED Paths
weak = dcase_dir + 'metadata/train/weak.tsv'
synthetic = dcase_dir + 'metadata/train/synthetic_2019/soundscapes.tsv'
unlabel = dcase_dir + 'metadata/train/unlabel_in_domain.tsv'
dcase2018_task5 = dcase_dir + "metadata/train/dcase2018_task5.tsv"
validation = dcase_dir + 'metadata/validation/validation.tsv'
eval_desed = dcase_dir + "metadata/eval/public.tsv"
# Useful because does not correspond to the tsv file path (metadata replace by audio), (due to subsets test/eval2018)
audio_validation_dir = dcase_dir + '/audio/validation'

# urbansound
urbansed_dir =  "./data/URBAN-SED_v2.0.0/" 
urban_train_tsv = urbansed_dir + "metadata/train.tsv"
urban_valid_tsv = urbansed_dir + "metadata/validate.tsv"
urban_eval_tsv = urbansed_dir + "metadata/test.tsv"


max_len_seconds = 10.
noise_snr = 30

# dcase features
sample_rate = 16000
n_window = 1024
n_fft = 1024
hop_size = 323
n_mels = 64
max_frames = math.ceil(max_len_seconds * sample_rate / hop_size)  # 496

# urbansound feature
usample_rate =  44100 
un_fft = 2048 
un_window = int(0.04 * usample_rate)
uhop_size = int(0.02 * usample_rate)
un_mels = 64
umax_frames = int(max_len_seconds * usample_rate / uhop_size)

# Training
checkpoint_epochs = None
save_best = True
early_stopping = True
es_init_wait = 50  # es for early stopping
in_memory = True

# Classes
file_path = os.path.abspath(os.path.dirname(__file__))
dcase_classes = pd.read_csv(os.path.join(file_path, validation), sep="\t").event_label.dropna().sort_values().unique()
urban_classes = pd.read_csv(os.path.join(file_path, urban_train_tsv),
                            sep="\t").event_label.dropna().sort_values().unique()

# Logger
terminal_level = logging.INFO

# focal loss related
alpha_fl = 0.5
gamma_fl = float(1)

