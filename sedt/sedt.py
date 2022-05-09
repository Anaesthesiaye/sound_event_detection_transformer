# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
SEDT model and criterion classes.
"""

import torch
import torch.nn.functional as F
from torch import nn
from utilities import box_ops
from utilities.utils import NestedTensor, nested_tensor_from_tensor_list, accuracy
import config as cfg


class SEDT(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, dec_at=False, pooling=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)  # MLP(hidden_dim, hidden_dim, 2, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_at = dec_at
        self.pooling = pooling
        if self.dec_at:
            self.query_embed = nn.Embedding(num_queries + 1, hidden_dim)
            self.weak_class_embed = nn.Linear(hidden_dim, num_classes)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if self.pooling is not None:
            if 'max' in self.pooling:
                self.pooling_func = nn.AdaptiveMaxPool2d((1, None))
            elif 'avg' in self.pooling:
                self.pooling_func = nn.AdaptiveAvgPool2d((1, None))
            elif "attn" in self.pooling:
                self.attn_dense_softmax = nn.Linear(hidden_dim, num_classes)
                self.softmax = nn.Softmax(dim=-1)
                def attn_pooling(x, y):
                    sof = self.attn_dense_softmax(x)
                    sof =  self.softmax(sof)
                    sof = torch.clamp(sof, min=1e-7, max=1)
                    res = (sof * y).sum(dim=1)/sof.sum(dim=1)
                    return res
                self.pooling_func = attn_pooling


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        if len(features) > 1:
            src, mask = features[-2].decompose()
        else:
            src, mask = features[-1].decompose()
        assert mask is not None
        out = {}
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1], enc_at_embed=None)
        if self.dec_at:
            outputs_class = self.class_embed(hs[:, :, 1:, :])
            outputs_coord = self.bbox_embed(hs[:, :, 1:, :]).sigmoid()
            at = self.weak_class_embed(hs[-1, :, 0, :]).squeeze().sigmoid()
            out['pred_logits'] = outputs_class[-1]
            out['pred_boxes'] = outputs_coord[-1]
            out['at'] = at
            if self.pooling is not None:
                class_pro = F.softmax(outputs_class[-1], -1)
                if 'weighted_sum' in self.pooling:
                    weights = out['pred_boxes'][:, :, 1]
                    at_p = (class_pro[:, :, :-1] * weights[:, :, None]).sum(1).clip(0, 1)
                elif 'attn' in self.pooling:
                    at_p = self.pooling_func(hs[-1, :, 1:, :], class_pro[:, :, :-1])
                else:
                    at_p = self.pooling_func(class_pro[:, :, :-1]).squeeze()
                # at_p = at_p.sigmoid()
                out['at_p'] = at_p
        else:
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out['pred_logits'] = outputs_class[-1]
            out['pred_boxes'] = outputs_coord[-1]
            if self.pooling is not None:
                class_pro = F.softmax(outputs_class[-1], -1)
                if 'attn' in self.pooling:
                    at_p = self.pooling_func(hs[-1, :, :, :], class_pro[:, :, :-1])
                else:
                    at_p = self.pooling_func(class_pro[:, :, :-1]).squeeze()
                # at_p = at_p.sigmoid()
                out['at_p'] = at_p

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.weak_label_criterion = nn.BCELoss()

    def loss_weak(self, outputs, targets, indices, num_boxes, strong_mask, weak_mask, coef, fine_tune=False, fl=False):
        # change the targets format
        losses = {}
        if 'at' in outputs:
            # strongly and weakly labeled data
            labeld_mask = slice(weak_mask.stop) if weak_mask is not None else slice(strong_mask.stop)
            pred_weak = outputs['at'][labeld_mask]
            gt = torch.zeros(pred_weak.shape, device=pred_weak.device)
            for i in range(pred_weak.shape[0]):
                for j, l in enumerate(targets[i]["labels"]):
                    if 'ratio' in targets[i]:
                        gt[i, l] += targets[i]['ratio'][j]
                    else:
                        gt[i, l] += 1
            gt = gt.clamp(0, 1)
            gt = gt.to(pred_weak.device)
            if fl:
                loss_weak = weak_focal_loss(pred_weak, gt)
            else:
                loss_weak = self.weak_label_criterion(pred_weak, gt)
            losses.update({'loss_weak': loss_weak})
        if 'at_p' in outputs:
            pooling_weak = outputs['at_p']
            loss_weak_p = self.weak_label_criterion(pooling_weak[weak_mask], gt[weak_mask])
            losses.update({'loss_weak_p': loss_weak_p})
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, strong_mask, weak_mask, coef, fine_tune=False, fl=False,
                    log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'][strong_mask]
        index = indices

        idx = self._get_src_permutation_idx(index)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets[strong_mask], index)])
        coef = torch.cat(coef).to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        coef_b = torch.full(src_logits.shape[:2], 1, dtype=torch.float32, device=src_logits.device)
        target_classes[idx] = target_classes_o
        coef_b[idx] = coef
        if fl:
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, self.empty_weight)
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')
        loss_ce_weighted = (loss_ce * coef_b).sum() / num_boxes
        losses = {'loss_ce': loss_ce_weighted}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, strong_mask, weak_mask, coef, fine_tune=False,
                         fl=False):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, strong_mask, weak_mask, coef, fine_tune=False, fl=False):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
           fine_tune : set the labels of unsatisfactory boxes as empty
                        to do : set the labels of satisfactory boxes as closest gt box
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes),
                              reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses = {}
        coef = torch.cat(coef).to(loss_giou.device)
        # tmp = coef.sum()
        losses['loss_bbox'] = (loss_bbox.sum(dim=1) * coef).sum() / num_boxes
        losses['loss_giou'] = (loss_giou * coef).sum() / num_boxes
        return losses

    def loss_feature(self, outputs, targets, indices, num_boxes, strong_mask, weak_mask, coef, fine_tune=False,
                     fl=False):
        """Compute the mse loss between normalized features.
        """
        target_feature = outputs['gt_feature']
        idx = self._get_src_permutation_idx(indices)
        batch_size = len(indices)
        target_feature = target_feature.view(batch_size, target_feature.shape[0] // batch_size, -1)

        src_feature = outputs['pred_feature'][idx]
        target_feature = torch.cat([t[i] for t, (_, i) in zip(target_feature, indices)], dim=0)

        # l2 normalize the feature
        src_feature = nn.functional.normalize(src_feature, dim=1)
        target_feature = nn.functional.normalize(target_feature, dim=1)

        loss_feature = F.mse_loss(src_feature, target_feature, reduction='none')
        losses = {'loss_feature': loss_feature.sum() / num_boxes}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, strong_mask, weak_mask, coef, fine_tune=False,
                 fl=False, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'weak': self.loss_weak,
            'feature': self.loss_feature
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, strong_mask, weak_mask, coef, fine_tune, fl=fl,
                              **kwargs)

    def forward(self, outputs, targets, weak_mask=None, strong_mask=None, fine_tune=False, normalize=False, fl=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if strong_mask is not None:
            outputs_without_aux = {k: v[strong_mask] for k, v in outputs.items() if k != 'aux_outputs'}

            # Retrieve the matching between the outputs of the last layer and the targets
            indices, coef = self.matcher(outputs_without_aux, targets[strong_mask], fine_tune=fine_tune,
                                               normalize=normalize, fl=fl)
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = torch.cat(coef).sum()
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_boxes'].device)
        else:
            indices, coef, num_boxes = None, None, None
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, strong_mask, weak_mask, coef,
                              fine_tune=fine_tune, fl=fl))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if strong_mask is not None:
                    aux_outputs_s = {k: v[strong_mask] for k, v in aux_outputs.items()}
                    sub_indices, coef = self.matcher(aux_outputs_s, targets[strong_mask], fl=fl)
                else:
                    sub_indices, coef = None, None
                for loss in self.losses:
                    if loss == 'weak':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, sub_indices, num_boxes, strong_mask, weak_mask, coef,
                                           fine_tune=fine_tune, fl=fl, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses, indices


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, audio_tags=None, at_m=2, is_semi=False, threshold=0.5):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        bs, num_q, nclass = out_logits.shape
        prob = F.softmax(out_logits, -1)
        if audio_tags is not None:
            _, idx = prob[..., :-1].max(1)
            if at_m == 1:
                audio_tags = audio_tags.unsqueeze(dim=1).repeat(1, num_q, 1).to(prob.device)
                prob[..., :-1] = prob[..., :-1] * audio_tags
            if at_m == 2:
                audio_tags = audio_tags.unsqueeze(dim=1).repeat(1, num_q, 1).to(prob.device)
                for i, j in enumerate(idx):
                    ind = prob[i, j, torch.arange(len(j))] < threshold
                    prob[i, j[ind], torch.arange(len(j))[ind]] = threshold
                prob[..., :-1] = prob[..., :-1] * audio_tags
            if at_m == 3:
                for i, (j, at) in enumerate(zip(idx, audio_tags)):
                    ind = prob[i, j, torch.arange(len(j))] < threshold
                    ind = ind & at.bool()
                    prob[i, j[ind], torch.arange(len(j))[ind]] = threshold
        scores, labels = prob[..., :-1].max(-1)

        if not is_semi:
            boxes = box_ops.box_cxcywh_to_se(out_bbox)
            l = target_sizes.unsqueeze(-1)
            boxes = boxes * l[:, None, :]
        else:
            boxes = out_bbox

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def sigmoid_focal_loss(inputs, targets, num_boxes, weight=None, alpha=cfg.alpha_fl, gamma: float = cfg.gamma_fl):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=weight, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum(2)


def weak_focal_loss(prob, targets, alpha=cfg.alpha_fl, gamma: float = cfg.gamma_fl):
    ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum(1).mean()