import datetime
import inspect
import os
from pprint import pprint
import numpy as np
import torch

from engine import evaluate, semi_train, adjust_threshold
from data_utils.SedData import SedData, get_dfs
from torch.utils.data import DataLoader
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
import config as cfg
from train_sedt import get_parser
from utilities.Logger import create_logger, set_logger
from utilities.Scaler import Scaler
from utilities.utils import SaveBest, collate_fn, get_cosine_schedule_with_warmup, back_up_code, EarlyStopping
from utilities.utils import to_cuda_if_available
from utilities.BoxEncoder import BoxEncoder
from utilities.BoxTransforms import get_transforms as box_transforms
from sedt import build_model
from utilities.utils import EMA



if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    parser = get_parser()
    # semi-train
    parser.add_argument('--focal_loss', action="store_true", default=False)
    parser.add_argument('--ema_m', type=float, default=0.9996, help='ema momentum for eval_model')
    parser.add_argument('--semi_batch_size', default=64, type=int)
    parser.add_argument('--accumlating_ema_steps', default=1, type=int)
    parser.add_argument('--teacher_model', default=None, help='load teacher from specific model')
    parser.add_argument('--teacher_eval', help='load teacher model for evaluation', action="store_false", default=True)

    f_args = parser.parse_args()
    assert f_args.dataname == "dcase", "only support dcase dataset now"
    if f_args.eval:
        f_args.epochs = 0
        assert f_args.info, "Don't give the model information to be evaluated"
    if f_args.info is None:
        f_args.info = f"semi_supervised_{f_args.dataname}_atloss_{f_args.weak_loss_coef}_atploss_{f_args.weak_loss_p_coef}_enc_{f_args.enc_layers}_pooling_{f_args.pooling}_{f_args.fusion_strategy}"
    if f_args.log:
        set_logger(f_args.info)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Semi-supervised Learning for Sound Event Detection Transformer")
    logger.info(f"Starting time: {datetime.datetime.now()}")

    os.environ["CUDA_VISIBLE_DEVICES"] = f_args.gpus

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
    dfs = get_dfs(dataset, f_args.dataname, unlabel_data=True)

    # Normalisation per audio
    add_axis_conv = 0
    scaler = Scaler()
    scaler_path = os.path.join(store_dir, f_args.dataname + ".json")
    num_class = cfg.dcase_classes
    label_encoder = BoxEncoder(num_class, seconds=cfg.max_len_seconds)
    transforms = box_transforms(cfg.max_frames, add_axis=add_axis_conv)
    encod_func = label_encoder.encode_strong_df
    weak_data = DataLoadDf(dfs["weak"], encod_func, transforms)
    train_synth_data = DataLoadDf(dfs["synthetic"], encod_func, transforms)
    train_labeled_data = ConcatDataset([weak_data, train_synth_data])

    if os.path.isfile(scaler_path):
        logger.info('loading scaler from {}'.format(scaler_path))
        scaler.load(scaler_path)
    else:
        scaler.calculate_scaler(train_labeled_data)
        scaler.save(scaler_path)
    logger.debug(f"scaler mean: {scaler.mean_}")

    # prepare transforms
    transforms_noise = box_transforms(cfg.max_frames, scaler, add_axis_conv,
                                      noise_dict_params={"mean": 0., "snr": cfg.noise_snr},
                                      freq_mask=f_args.freq_mask, freq_shift=f_args.freq_shift,
                                      time_mask=f_args.time_mask)
    transforms_valid = box_transforms(cfg.max_frames, scaler, add_axis_conv)


    # prepare train dataset
    semi_weak_data = DataLoadDf(dfs["weak"], encod_func, transforms_noise, in_memory=cfg.in_memory)
    semi_train_synth_data = DataLoadDf(dfs["synthetic"], encod_func, transforms_noise, in_memory=cfg.in_memory)
    unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms_noise, in_memory=cfg.in_memory)

    # prepare semi-supervised learning dataset, default: a batch contains 1/4 synthetic data, 1/4 weak data, 1/2 unlabel data
    train_semi_dataset = [semi_train_synth_data, semi_weak_data, unlabel_data]
    semi_batch_sizes = [f_args.semi_batch_size // 4, f_args.semi_batch_size // 4, 2 * f_args.semi_batch_size // 4]

    # prepare semi dataloader
    semi_concat_dataset = ConcatDataset(train_semi_dataset)
    semi_sampler = MultiStreamBatchSampler(semi_concat_dataset, batch_sizes=semi_batch_sizes)
    semi_training_loader = DataLoader(semi_concat_dataset, batch_sampler=semi_sampler, collate_fn=collate_fn,
                                      pin_memory=True, num_workers=0)

    # prepare data mask, use it to calculate loss and split labeled and unlabeled dataset
    semi_weak_mask = slice(semi_batch_sizes[0], semi_batch_sizes[0] + semi_batch_sizes[1])
    semi_strong_mask = slice(semi_batch_sizes[0])
    semi_label_mask = slice(semi_batch_sizes[0] + semi_batch_sizes[1])
    semi_unlabel_mask = slice(semi_batch_sizes[0] + semi_batch_sizes[1], f_args.semi_batch_size)

    # prepare eval dataloader
    validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)
    eval_data = DataLoadDf(dfs["eval"], encod_func, transform=transforms_valid, return_indexes=True)
    validation_dataloader = DataLoader(validation_data, batch_size=f_args.batch_size, collate_fn=collate_fn,
                                       num_workers=0)
    eval_dataloader = DataLoader(eval_data, batch_size=f_args.batch_size, collate_fn=collate_fn, num_workers=0)
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

    # load a well-trained model as teacher
    if not f_args.eval:
        assert f_args.teacher_model is not None, "please provide the teacher model"
        model_fname = os.path.join(saved_model_dir, f_args.teacher_model)
        if torch.cuda.is_available():
            state = torch.load(model_fname)
        else:
            state = torch.load(model_fname, map_location=torch.device('cpu'))
        load_dict = state['model']['state_dict']
        model.load_state_dict(load_dict)
        logger.info('Using teacher model: ' + model_fname)

    model = model.cuda()

    # ema formula
    ema = EMA(model, f_args.ema_m)
    ema.register()

    # optimizer and scheduler
    optim= torch.optim.AdamW(param_dicts, lr=f_args.lr, weight_decay=f_args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(optim, f_args.epochs, num_warmup_steps=f_args.epochs * 0)

    state = {
        'model': {"name": model.__class__.__name__,
                  'args': '',
                  "kwargs": '',
                  'state_dict': model.state_dict()},

        'ema_model': {"name": ema.model.__class__.__name__,
                      'args': '',
                      "kwargs": '',
                      'state_dict': ema.model.state_dict()},

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

    start_epoch = 0
    origin_threshold = torch.tensor([0.5] * f_args.num_classes)
    origin_threshold = to_cuda_if_available(origin_threshold)
    classwise_threshold = origin_threshold

    for epoch in range(start_epoch, f_args.epochs):
        # Train
        model.train()
        loss_value, pseudo_labels_counter = semi_train(semi_training_loader, model, ema, criterion, optim, epoch,
                                                       f_args.accumrating_gradient_steps, f_args.accumlating_ema_steps,
                                                       postprocessors,
                                                       mask_weak=semi_weak_mask, fine_tune=f_args.fine_tune,
                                                       normalize=f_args.normalize,
                                                       mask_strong=semi_strong_mask, max_norm=0.1,
                                                       mask_unlabel=semi_unlabel_mask, mask_label=semi_label_mask,
                                                       fl=f_args.focal_loss, mix_up_ratio=f_args.mix_up_ratio,
                                                       classwise_threshold=classwise_threshold)

        classwise_threshold = adjust_threshold(pseudo_labels_counter, origin_threshold)

        if f_args.adjust_lr:
            lr_scheduler.step()

        # Validation
        model = model.eval()

        # Update state
        state['model']['state_dict'] = model.state_dict()  # student
        ema.apply_shadow()
        state['ema_model']['state_dict'] = ema.model.state_dict()  # teacher
        ema.restore()

        state['optimizer']['state_dict'] = optim.state_dict()
        state['epoch'] = epoch


        # Validation with real data
        if f_args.teacher_eval:
            logger.info("Using teacher model for validation \n")
            ema.apply_shadow()
        else:
            logger.info("Using student model for validation \n")

        metrics = evaluate(model, criterion, postprocessors, validation_dataloader, label_encoder, validation_labels_df,
                           at=True, fusion_strategy=fusion_strategy)

        if f_args.teacher_eval:
            ema.restore()

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
            model_fname = os.path.join(saved_model_dir, "semi_train_{}_loss_{}".format(f_args.info, epoch))
            torch.save(state, model_fname)

    if cfg.save_best or f_args.eval:
        for at_m in fusion_strategy:
            model_fname = os.path.join(saved_model_dir, f"{f_args.info}_{at_m}_best")
            if torch.cuda.is_available():
                state = torch.load(model_fname)
            else:
                state = torch.load(model_fname, map_location=torch.device('cpu'))
            if f_args.teacher_eval:
                model.load_state_dict(state['ema_model']['state_dict'])
                logger.info(f"using teacher model for test...")
            else:
                model.load_state_dict(state['model']['state_dict'])
                logger.info(f"using student model for test...")
            logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")

            # ##############
            # Validation
            # ##############
            model.eval()
            logger.info("Metric on validation")
            evaluate(model, criterion, postprocessors, validation_dataloader, label_encoder, validation_labels_df,
                     at=True, fusion_strategy=[at_m], cal_seg=True, cal_clip=True)

            logger.info("Metric on eval")
            evaluate(model, criterion, postprocessors, eval_dataloader, label_encoder, eval_labels_df,
                     at=True, fusion_strategy=[at_m], cal_seg=True, cal_clip=True)