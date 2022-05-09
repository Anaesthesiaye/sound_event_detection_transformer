# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from collections import Counter
from utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import config as cfg


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, epsilon=0, alpha=100):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.epsilon = epsilon
        self.alpha = alpha
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, fine_tune=False, normalize=False, fl = False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) if not fl else outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"][:len(v["boxes"])] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if not fl:
            cost_class = -out_prob[:, tgt_ids]
        else:
            alpha_fl = cfg.alpha_fl
            gamma_fl = cfg.gamma_fl
            neg_cost_class = (1 - alpha_fl) * (out_prob ** gamma_fl) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha_fl * ((1 - out_prob) ** gamma_fl) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox), p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices1 = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # matching order by TLOSS matcher
        idx = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices1]
        Coef = []
        if fine_tune:
            # matching order by LLoss
            out_bbox = out_bbox.unsqueeze(dim=1).repeat(1, tgt_bbox.shape[0], 1)
            tgt_bbox = tgt_bbox.unsqueeze(dim=0).repeat(out_bbox.shape[0], 1, 1)
            # C_l = torch.abs(box_cxcywh_to_xyxy(out_bbox) - box_cxcywh_to_xyxy(tgt_bbox)).max(-1)[0]
            C_l = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C_l = C_l.view(bs, num_queries, -1).cpu()
            indices2 = [c[i].min(-1) for i, c in enumerate(C_l.split(sizes, -1))]
            idx1 = idx
            idx = []
            for i1, i2 in zip(idx1, indices2):
                i1 = list(i1)
                num_gt = len(i1[1])
                reserved = i2[0] < self.epsilon
                idx1_res = reserved[i1[0]] == True
                i1[0] = i1[0][idx1_res]
                i1[1] = i1[1][idx1_res]
                reserved[i1[0]] = False
                reserved_index = torch.where(reserved == True)[0]
                random_del_index = torch.where(torch.rand(len(reserved_index)) > (self.alpha * num_gt / num_queries))[0]
                reserved[reserved_index[random_del_index]] = False
                idx += [(torch.cat([i1[0], torch.arange(num_queries)[reserved]], dim=-1),
                         torch.cat([i1[1], torch.as_tensor(i2[1], dtype=torch.int64)[reserved]], dim=-1))]

        for i, (_, tgt) in enumerate(idx):
            if normalize:
                cur_list = tgt.tolist()
                num = Counter(cur_list)
                coef = torch.tensor([1 / num[i] for i in cur_list], dtype=torch.float32).cpu()
            elif "ratio" in targets[i]:
                coef = targets[i]["ratio"].cpu()
            else:
                coef = torch.tensor([1] * len(tgt), dtype=torch.float32).cpu()
            Coef.append(coef)
        return idx, Coef




def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou, epsilon=args.epsilon, alpha=args.alpha)

