#!/usr/bin/env python
# encoding: utf-8
"""
@author: yzr
@file: collapse_event.py
@time: 2020/9/2 21:43
"""
import os, sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
from tqdm import tqdm
import config as cfg

def collapse(meta_df):
    df_new = pd.DataFrame()
    filenames = meta_df.filename.drop_duplicates()
    cols = ["onset", "offset", "event_label"]
    for f in tqdm(filenames):
        label = meta_df[meta_df.filename == f][cols]
        events = label.event_label.drop_duplicates()
        for e in events:
            time = label[label.event_label == e][["onset", "offset"]]
            time = time.sort_values(by='onset')
            time = time.reset_index(drop=True)
            i = 0
            while i < len(time):
                if i == 0:
                    i += 1
                    continue
                if time.loc[i, 'onset'] <= time.loc[i-1, 'offset']:
                    time.loc[i-1, 'offset'] = max(time.loc[i, 'offset'], time.loc[i-1, 'offset'])
                    time = time.drop(index=i).reset_index(drop=True)
                    i = i-1
                i += 1
            time["event_label"] = e.strip()
            time["filename"] = f
            df_new = df_new.append(time, ignore_index=True)
    return df_new

if __name__=='__main__':
    annotation_dir = os.path.join(cfg.urbansed_dir, "annotations")
    datasets = ["train", "validate", "test"]
    meta_dir = annotation_dir.replace("annotations", "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    for dataset in datasets:
        df = pd.DataFrame(columns=["filename", "event_label", "onset", "offset"])
        df_sub = pd.DataFrame()
        f_list = list(filter(lambda x: x.endswith(".txt") and not x.startswith("."), os.listdir(os.path.join(annotation_dir, dataset))))
        for f in tqdm(f_list):
            fr = open(os.path.join(annotation_dir, dataset, f),'r')
            lines = fr.readlines()
            lines = [l.split('\t') for l in lines]
            df_sub = pd.DataFrame(lines, columns=["onset", "offset", "event_label"])
            df_sub["filename"] = os.path.splitext(f)[0] + ".wav"
            df = df.append(df_sub,ignore_index=True)
        df = collapse(df)
        df = df[["filename", "event_label", "onset", "offset"]]
        df.to_csv(os.path.join(meta_dir, dataset+".tsv"), index=False, sep="\t", float_format="%.3f")







