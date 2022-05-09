import math
import sys
import time

import inspect
import torch
import pandas as pd
from data_utils.DataLoad import data_prefetcher
from utilities.metrics import audio_tagging_results, compute_metrics
from utilities.Logger import create_logger
from utilities.distribute import reduce_dict, get_reduced_loss
from utilities.mixup import mixup_data, mixup_label_unlabel
from utilities.utils import MetricLogger, SmoothedValue, AverageMeter, to_cuda_if_available
from collections import Counter
import config as cfg
import numpy as np


def train(train_loader, model, criterion, optimizer, c_epoch, accumrating_gradient_steps,
          mask_weak=None, mask_strong=None, fine_tune=False, normalize=False, max_norm=0.1, mix_up_ratio=0):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: (input, labels)
        model: torch.Module, model to be trained
        criterion:
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

        if mix_up_ratio:
            batch_input, target, mask_strong_c, mask_weak_c  = mixup_data(batch_input, target, mask_strong, mask_weak,  mix_up_ratio, alpha=1)
        else:
            mask_weak_c, mask_strong_c = mask_weak, mask_strong

        # Outputs
        if 'patches' in target[0]:
            patches = [t['patches'] for t in target]
            patches = torch.stack(patches, dim=0)
            outputs = model(batch_input.decompose(), patches)
        else:
            outputs = model(batch_input)

        loss_dict, _ = criterion(outputs, target, mask_weak_c, mask_strong_c, fine_tune, normalize)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # loss_value = losses_reduced_scaled.item()
        loss_value = get_reduced_loss(loss_dict, weight_dict, metric_logger)

        if not math.isfinite(loss_value):
            log.info("Loss is {}, stopping training".format(loss_value))
            log.info(loss_dict)
            sys.exit(1)

        losses.backward()
        if (i + 1) % accumrating_gradient_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_error=0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_input, target = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info("Epoch:{} data_time:{:.3f}({:.3f}) batch_time:{:.3f}({:.3f})".
          format(c_epoch, data_time.val, data_time.avg, batch_time.val, batch_time.avg))
    log.info("Train averaged stats: \n" + str(metric_logger))
    return loss_value

def semi_train(train_loader, model, ema, criterion, optimizer, c_epoch, accumrating_gradient_steps,
               accumlating_ema_steps, postprocessor,
               mask_weak=None, mask_strong=None, fine_tune=False, normalize=False, max_norm=0.1, mask_unlabel=None,
               mask_label=None, fl=False, mix_up_ratio=0, classwise_threshold=None):
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    metric_logger: MetricLogger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    log.debug("Nb batches: {}".format(len(train_loader)))
    end = time.time()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    i = -1
    pseudo_labels_counter = Counter()
    for _, data in enumerate(train_loader):
        i += 1
        ((batch_input_teacher, batch_input_student), target) = to_cuda_if_available(data)
        data_time.update(time.time() - end)
        global_step = c_epoch * len(train_loader) + i + 1

        # split labeled and unlabeld data
        batch_input_labeled = batch_input_teacher[mask_label]
        target_labeled = target[mask_label]

        batch_input_unnlabel_teacher = batch_input_teacher[mask_unlabel]
        batch_input_unnlabel_student = batch_input_student[mask_unlabel]
        target_unlabeled = target[mask_unlabel]


        # train on labeled data like sedt
        if mix_up_ratio > 0:
            batch_input_labeled, target_labeled, mask_strong_c, mask_weak_c = mixup_data(batch_input_labeled,
                                                                                         target_labeled, mask_strong,
                                                                                         mask_weak, mix_up_ratio = mix_up_ratio, alpha=1)
        else:
            mask_weak_c, mask_strong_c = mask_weak, mask_strong
        labeled_outputs = model(batch_input_labeled)
        sup_loss_dict, _ = criterion(labeled_outputs, target_labeled, mask_weak_c, mask_strong_c, fine_tune, normalize,
                                     fl)
        weight_dict = criterion.weight_dict
        sup_losses = sum(sup_loss_dict[k] * weight_dict[k] for k in sup_loss_dict.keys() if k in weight_dict)
        sup_loss_value = get_reduced_loss(sup_loss_dict, weight_dict, metric_logger, prefix="sup_")


        # train on unlabeld data
        # teacher
        ema.apply_shadow()
        with torch.no_grad():
            tea_outputs = model(batch_input_unnlabel_teacher)
            orig_unlabel_target_sizes = torch.stack([t["orig_size"] for t in target_unlabeled], dim=0)
            pseudo_labels = get_pseudo_labels(tea_outputs, postprocessor, orig_unlabel_target_sizes, target_unlabeled,
                                              pseudo_labels_counter, classwise_threshold=classwise_threshold)
            if mix_up_ratio > 0:
                batch_input_unnlabel_student, pseudo_labels = mixup_label_unlabel(batch_input_labeled,
                                                                           batch_input_unnlabel_student, target_labeled,
                                                                           pseudo_labels, alpha=1)
        ema.restore()

        # student
        st_outputs = model(batch_input_unnlabel_student)

        unsup_loss_dict, _ = criterion(st_outputs, pseudo_labels, None,
                                       slice(batch_input_unnlabel_student.tensors.size(0)), fine_tune, normalize, fl)

        unsup_losses = sum(unsup_loss_dict[k] * weight_dict[k] for k in unsup_loss_dict.keys() if k in weight_dict)
        unsup_loss_value = get_reduced_loss(unsup_loss_dict, weight_dict, metric_logger, prefix="unsup_")


        total_losses = sup_losses + unsup_losses
        if not (math.isfinite(total_losses)):
            print("Loss is infinite, stopping training")
            sys.exit(1)
        total_losses.backward()



        if (i + 1) % accumrating_gradient_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % accumlating_ema_steps == 0:
            ema.update()

        global_step += 1
        metric_logger.update(class_error=0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=sup_loss_value + unsup_loss_value)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # gather the stats from all processes
    log.info("Epoch:{} data_time:{:.3f}({:.3f}) batch_time:{:.3f}({:.3f})".
             format(c_epoch, data_time.val, data_time.avg, batch_time.val, batch_time.avg))
    log.info("Train averaged stats: \n" + str(metric_logger))
    log.info("class nums: " + str(pseudo_labels_counter))
    return sup_loss_value + unsup_loss_value, pseudo_labels_counter


def evaluate(model, criterion, postprocessors, dataloader, decoder, ref_df, fusion_strategy, at=True, cal_seg=False, cal_clip=False):
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    audio_tag_dfs, dec_prediction_dfs = get_sedt_predictions(model, criterion, postprocessors, dataloader, decoder, fusion_strategy, at)


    if not audio_tag_dfs.empty:
        clip_metric = audio_tagging_results(ref_df, audio_tag_dfs)
        logger.info(f"AT Class-wise clip metrics \n {'=' * 50} \n {clip_metric}")

    metrics = {}
    logger.info(f"decoder output \n {'=' * 50}")
    for at_m, dec_pred in dec_prediction_dfs.items():
        logger.info(f"Fusion strategy: {at_m}")
        event_macro_f1 = compute_metrics(dec_pred, ref_df, cal_seg=cal_seg, cal_clip=cal_clip)
        metrics[at_m] = event_macro_f1
    return metrics



def get_sedt_predictions(model, criterion, postprocessors, dataloader, decoder, fusion_strategy, at=True):
    """ Get the predictions of a trained model on a specific set
    Args:
        model: torch.Module, a trained pytorch model (you usually want it to be in .eval() mode).
        dataloader: torch.utils.data.DataLoader, giving ((input_data, label), indexes) but label is not used here
        decoder: function, takes a numpy.array of shape (time_steps, n_labels) as input and return a list of lists
            of ("event_label", "onset", "offset") for each label predicted.

    Returns:
        dict of the different predictions with associated fusion_strategy
    """
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    # Init a dataframe per threshold
    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    epoch_time = time.time()
    decoding_time = 0.
    dec_prediction_dfs = {}
    audio_tag_dfs = pd.DataFrame()
    for at_m in fusion_strategy:
        dec_prediction_dfs[at_m] = pd.DataFrame()

    # Get predictions
    prefetcher = data_prefetcher(dataloader, return_indexes=True)
    i = -1
    (input_data, targets), indexes = prefetcher.next()
    while input_data is not None:
        i += 1
        with torch.no_grad():
            outputs = model(input_data)
        # ##############
        # compute losses
        # ##############
        weak_mask = None
        strong_mask = slice(len(input_data.tensors))
        loss_dict, indices = criterion(outputs, targets, weak_mask, strong_mask)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_value = get_reduced_loss(loss_dict, weight_dict, metric_logger)
        # loss_dict_unscaled = {f'{k}_unscaled': v for k, v in loss_dict.items()}
        # loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        # losses_scaled = sum(loss_dict_scaled.values())
        # loss_value = losses_scaled.item()
        # metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_dict_unscaled)

        # ###################
        # get decoder results
        # ###################
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if at:
            assert "at" in outputs
            audio_tags = outputs["at"]
            audio_tags = (audio_tags > 0.5).long()
            for j, audio_tag in enumerate(audio_tags):
                audio_tag_res = decoder.decode_weak(audio_tag)
                audio_tag_res = pd.DataFrame(audio_tag_res, columns=["event_label"])
                audio_tag_res["filename"] = dataloader.dataset.filenames.iloc[indexes[j]]
                audio_tag_res["onset"] = 0
                audio_tag_res["offset"] = 0
                audio_tag_dfs = audio_tag_dfs.append(audio_tag_res)
        else:
            audio_tags = None

        decoding_start = time.time()
        for at_m in fusion_strategy:
            results = postprocessors['bbox'](outputs, orig_target_sizes, audio_tags=audio_tags, at_m=at_m)

            for j, res in enumerate(results):
                for item in res:
                    res[item] = res[item].cpu()
                pred = decoder.decode_strong(res, threshold=0.5)
                pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset", "score"])
                # Put them in seconds
                pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(0, cfg.max_len_seconds)
                pred["filename"] = dataloader.dataset.filenames.iloc[indexes[j]]
                dec_prediction_dfs[at_m] = dec_prediction_dfs[at_m].append(pred, ignore_index=True)

        decoding_time += time.time() - decoding_start
        (input_data, targets), indexes = prefetcher.next()

    logger.info("Val averaged stats:" + metric_logger.__str__())
    epoch_time = time.time() - epoch_time
    logger.info(f"val_epoch_time:{epoch_time}  decoding_time:{decoding_time}")
    return audio_tag_dfs, dec_prediction_dfs


def get_pseudo_labels(tea_outputs, postprocessor, orig_unlabel_target_sizes, target_unlabeled, pseudo_labels_counter,
                      threshold=0.5, del_overlap=True, classwise_threshold=None):
    if "at" in tea_outputs:
        audio_tags = tea_outputs["at"]
        audio_tags = (audio_tags >= classwise_threshold).long()
    else:
        audio_tags = None

    results = postprocessor['bbox'](tea_outputs, orig_unlabel_target_sizes, audio_tags=audio_tags, at_m=1, is_semi=True,
                                    threshold=None)

    for i, result in enumerate(results):
        filter_class = classwise_threshold[result['labels']]
        filtered_idx_1 = result['scores'] >= filter_class  # confidence score > threshold
        filtered_idx_2 = result['boxes'][:, 1] > 0.2 / orig_unlabel_target_sizes[0].item()  # duration > 0.02 s
        filtered_idx = filtered_idx_1 & filtered_idx_2

        if not del_overlap:
            target_unlabeled[i]['labels'] = result['labels'][filtered_idx]
            target_unlabeled[i]['boxes'] = result['boxes'][filtered_idx]
        else:
            # delete overlapped event
            tmp_labels, tmp_boxes, tmp_scores = result['labels'][filtered_idx], result['boxes'][filtered_idx], \
                                                result['scores'][filtered_idx]
            tmp_scores, indices = tmp_scores.sort(descending=True)
            x = tmp_boxes[:, 0] - tmp_boxes[:, 1] / 2
            y = tmp_boxes[:, 0] + tmp_boxes[:, 1] / 2
            keep = []
            while indices.numel() > 0:
                if indices.numel() == 1:
                    k = indices.item()
                    keep.append(k)
                    break
                else:
                    k = indices[0].item()
                    keep.append(k)
                    cur_label = tmp_labels[k]
                x_max = x[indices[1:]].clamp(min=x[k])
                y_min = y[indices[1:]].clamp(max=y[k])
                overlap = (y_min - x_max).clamp(min=0)
                idx = ((overlap == 0) + (tmp_labels[indices[1:]] != cur_label.item())).nonzero().squeeze()
                if idx.numel() == 0:
                    break
                indices = indices[idx + 1]
            target_unlabeled[i]['labels'] = tmp_labels[keep]
            target_unlabeled[i]['boxes'] = tmp_boxes[keep]
            pseudo_labels_counter.update(tmp_labels[keep].cpu().numpy().tolist())

    return target_unlabeled

def adjust_threshold(pseudo_labels_counter, origin_threshold):
    labels_num_dict = dict(sorted(dict(pseudo_labels_counter).items(), key=lambda x: x[0]))
    labels_num = np.array(list(labels_num_dict.values()))
    labels_ratio = torch.tensor(labels_num / np.sum(labels_num))
    true_distribution = torch.tensor(
        [0.09915014, 0.02266289, 0.08050047, 0.13385269, 0.13456091, 0.01534466, 0.02219075, 0.05594901, 0.41406988,
         0.0217186])
    adjust_ratio = (labels_ratio / true_distribution) ** 0.7
    adjust_ratio = to_cuda_if_available(adjust_ratio)
    class_threshold = torch.clamp(adjust_ratio * origin_threshold, min=0.45, max=0.7)
    return class_threshold