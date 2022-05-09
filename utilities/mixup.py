#!/usr/bin/env python
# encoding: utf-8
"""
@author: yzr
@file: mixup.py
@time: 2020/8/5 12:53
"""
import numpy as np
import torch
from utilities.box_ops import box_cxcywh_to_se


def mixup_data(x, y, mask_strong, mask_weak, mix_up_ratio=0.5, max_events=20, alpha=3):
    """
    mix up data for sedt model, only mix up data with same type of labels
    :param x: feature, shape:[batch_size, channel, nframe, nfreq]
    :param y: label, [{"labels":ndarray, "boxes":ndarray, "orig_size"},{...},...]
    :param mix_up_ratio:  only mix up a part of data
    :param alpha:
    :return:
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x.tensors.shape[0]
    mix_num = int(bs*mix_up_ratio)
    index = np.asarray(list(range(bs)))
    np.random.shuffle(index)

    data_1 = x.tensors[:mix_num, :]
    data_2 = x.tensors[index][:mix_num, :]
    label_1 = y[:mix_num]
    label_2 = [y[i] for i in index[:mix_num]]

    # not collapse overlapped events of the same class
    data = lam * data_1 + (1-lam) * data_2
    strong_label, strong_data = [], []
    weak_label, weak_data = [], []
    unlabel, unlabel_data = [], []
    for i, (l_1, l_2 )in enumerate(zip(label_1,label_2)):
        if len(l_1["boxes"]) == 0 or len(l_2["boxes"]) ==0:
            if (len(l_1["boxes"]) > 0):
                strong_label.append(label_1[i])
                strong_data.append(data_1[i].unsqueeze(dim=0))
            elif (len(l_2["boxes"]) > 0):
                strong_label.append(label_2[i])
                strong_data.append(data_2[i].unsqueeze(dim=0))
            else:
                weak_label.append({
                "labels": torch.cat((l_1["labels"], l_2["labels"]), dim=0),
                "boxes": torch.tensor([], device=x.tensors.device),
                "ratio": torch.tensor([lam] * len(l_1["labels"]) + [1 - lam] * len(l_2["labels"]), device=x.tensors.device),
                "orig_size": l_1["orig_size"]})
                weak_data.append(data[i].unsqueeze(dim=0))
        else:
            # abandom data with more than num_queries events
            if len(l_1["boxes"]) + len(l_2["boxes"]) > max_events:
                if len(l_1["boxes"]):
                    strong_label.append(l_1)
                    strong_data.append(data_1[i].unsqueeze(dim=0))
                else:
                    strong_label.append(l_2)
                    strong_data.append(data_2[i].unsqueeze(dim=0))
            else:
                ds = data_1[i] # data with strong label
                if len(l_1["boxes"]) == 0:
                    tmp = l_1
                    l_1 = l_2
                    l_2 = tmp
                    lam = 1-lam
                    ds = data_2[i]

                strong_label.append({
                    "labels": torch.cat((l_1["labels"], l_2["labels"]),dim=0),
                    "boxes": torch.cat((l_1["boxes"], l_2["boxes"]), dim=0),
                    "ratio": torch.tensor([lam]*len(l_1["labels"]) + [1-lam]*len(l_2["labels"]), device=x.tensors.device),
                    "orig_size": l_1["orig_size"]
                })
                strong_data.append(data[i].unsqueeze(dim=0))

                # abandom data which mix up events with same class label
                cur_labels = strong_label[-1]["labels"]
                cur_boxes = strong_label[-1]["boxes"]
                events = set(cur_labels.tolist())
                for e in events:
                    boxes = cur_boxes[(cur_labels == e)[:len(cur_boxes)]]
                    boxes = box_cxcywh_to_se(boxes)
                    boxes = boxes[boxes.argsort(dim=0)[:, 0]]
                    boxes_e = boxes[:, 1][:-1]
                    boxes_s = boxes[:, 0][1:]
                    if not (boxes_e < boxes_s).all().item():
                        strong_label[-1] = l_1
                        strong_data[-1] = ds.unsqueeze(dim=0)
                        break
    data_final = []
    label_final = []
    # integrate with non-mix-up data
    if len(x.tensors[mask_strong][mix_num:]):
        strong_data.append(x.tensors[mask_strong][mix_num:])
        strong_label.extend(y[mask_strong][mix_num:])
    if len(strong_data):
        data_final.extend(strong_data)
        label_final.extend(strong_label)

    if mask_weak is not None:
        left_weak_index = max(0, mix_num - mask_strong.stop)
        if len(x.tensors[mask_weak][left_weak_index:]):
            weak_data.append(x.tensors[mask_weak][left_weak_index:])
            weak_label.extend(y[mask_weak][left_weak_index:])
        if len(weak_data):
            data_final.extend(weak_data)
            label_final.extend(weak_label)


        left_unlabel_index = max(0, mix_num - mask_weak.stop)
        if len(x.tensors[mask_weak.stop:][left_unlabel_index:]):
            unlabel_data.append(x.tensors[mask_weak.stop:][left_unlabel_index:])
            unlabel.extend(y[mask_weak.stop:][left_unlabel_index:])
        if len(unlabel_data):
            data_final.extend(unlabel_data)
            label_final.extend(unlabel)

    x.tensors = torch.cat(data_final,dim=0)
    y = label_final

    return x, y, slice(len(strong_label)), slice(len(strong_label), len(strong_label)+len(weak_label))

def mixup_label_unlabel(x1, x2, y1, y2, mix_up_ratio=0.5, max_events=20, alpha=3):
    """
    mix up data for sedt model
    :param x1: label data feature, shape:[batch_size, channel, nframe, nfreq]
    :param y1: label data target, [{"labels":ndarray, "boxes":ndarray, "orig_size"},{...},...]
    :param x2: unlabel data feature, shape:[batch_size, channel, nframe, nfreq]
    :param y2: unlabel data target, [{"labels":ndarray, "boxes":ndarray, "orig_size"},{...},...]
    :param mix_up_ratio:  only mix up a part of data
    :param alpha:
    :return:
    """
    assert mix_up_ratio <= 0.5
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x1.tensors.shape[0]
    mix_num = int(bs * mix_up_ratio)

    data_1 = x1.tensors[:mix_num, :]
    data_2 = x2.tensors[:mix_num, :]
    label_1 = y1[:mix_num]
    label_2 = y2[:mix_num]

    # not collapse overlapped events of the same class
    data = lam * data_1 + (1 - lam) * data_2
    strong_label, strong_data = [], []

    for i, (l_1, l_2) in enumerate(zip(label_1, label_2)):
        if len(l_1["boxes"]) + len(l_2["boxes"]) > max_events:
            if len(l_2["boxes"]):
                strong_label.append(l_2)
                strong_data.append(data_2[i].unsqueeze(dim=0))
            else:
                strong_label.append(l_1)
                strong_data.append(data_1[i].unsqueeze(dim=0))
        else:
            ds = data_1[i]
            strong_label.append({
                "labels": torch.cat((l_1["labels"], l_2["labels"]), dim=0),
                "boxes": torch.cat((l_1["boxes"], l_2["boxes"]), dim=0),
                "ratio": torch.tensor([lam] * len(l_1["labels"]) + [1 - lam] * len(l_2["labels"]),
                                      device=x1.tensors.device),
                "orig_size": l_1["orig_size"]
            })
            strong_data.append(data[i].unsqueeze(dim=0))

            # abandon data which mix up events with same class label
            cur_labels = strong_label[-1]["labels"]
            cur_boxes = strong_label[-1]["boxes"]
            events = set(cur_labels.tolist())
            for e in events:
                boxes = cur_boxes[(cur_labels == e)[:len(cur_boxes)]]
                boxes = box_cxcywh_to_se(boxes)
                boxes = boxes[boxes.argsort(dim=0)[:, 0]]
                boxes_e = boxes[:, 1][:-1]
                boxes_s = boxes[:, 0][1:]
                if not (boxes_e < boxes_s).all().item():
                    strong_label[-1] = l_1
                    strong_data[-1] = ds.unsqueeze(dim=0)
                    break

    strong_data.append(x2.tensors[mix_num:])
    strong_label.extend(y2[mix_num:])

    x2.tensors, y2 = torch.cat(strong_data, dim=0), strong_label

    return x2, y2

