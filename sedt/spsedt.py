# ------------------------------------------------------------------------
# Modified from UP-DETR (https://github.com/dddzg/up-detr)
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
SP-SEDT model
"""
import torch
from torch import nn
from utilities.utils import  NestedTensor
from .sedt import SEDT, MLP


class SPSEDT(SEDT):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, dec_at=False,feature_recon=True,
                 query_shuffle=False, mask_ratio=0.1, num_patches=10, pooling=None):
        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss, dec_at, pooling)
        hidden_dim = transformer.d_model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.patch2query = nn.Linear(backbone.num_channels, hidden_dim)
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.feature_recon = feature_recon
        if self.feature_recon:
            self.feature_align = MLP(hidden_dim, hidden_dim, backbone.num_channels, 2)
        self.query_shuffle = query_shuffle
        assert num_queries % num_patches == 0
        query_per_patch = num_queries // num_patches
        self.attention_mask = torch.ones(self.num_queries, self.num_queries) * float('-inf')
        for i in range(num_patches):
            self.attention_mask[i * query_per_patch:(i + 1) * query_per_patch,
            i * query_per_patch:(i + 1) * query_per_patch] = 0

    def forward(self, samples: list, patches: torch.Tensor):
    # def forward(self, samples: NestedTensor, patches: torch.Tensor):
        batch_num_patches = patches.shape[1]
        samples = [s.cuda() for s in samples]
        patches = patches.cuda()
        if isinstance(samples, (list, torch.Tensor)):
            # samples = nested_tensor_from_tensor_list(samples)
            samples=NestedTensor(samples[0],samples[1])
        feature, pos = self.backbone(samples)

        src, mask = feature[-1].decompose()
        assert mask is not None

        bs = patches.shape[0]
        patches = patches.flatten(0, 1)
        patches_feature = self.backbone(patches)
        patches_feature_gt = self.avgpool(patches_feature[-1]).flatten(1)

        # [num_queries, bs, hidden_dim]
        patches_feature = self.patch2query(patches_feature_gt) \
            .view(bs, batch_num_patches, 1, -1) \
            .repeat(1, 1, self.num_queries // self.num_patches, 1) \
            .flatten(1, 2).permute(1, 0, 2) \
            .contiguous()

        # only shuffle the event queries
        idx = torch.randperm(self.num_queries) if self.query_shuffle else torch.arange(self.num_queries)

        start = 1 if self.dec_at else 0
        if self.training:
            # for training, it uses fixed number of query patches.
            mask_query_patch = (torch.rand(self.num_queries, bs, 1, device=patches.device) > self.mask_ratio).float()
            decoder_input = self.query_embed.weight[start:, :].unsqueeze(1).repeat(1, bs, 1)[idx]  # don't include audio query
            decoder_input += patches_feature * mask_query_patch + decoder_input
            hs, memory = self.transformer(self.input_proj(src), mask, decoder_input, pos[-1],
                                          decoder_mask=self.attention_mask.to(patches_feature.device))
        else:
            # for test, it supports x query patches, where x<=self.num_queries.
            num_queries = batch_num_patches * self.num_queries // self.num_patches
            decoder_input = patches_feature + self.query_embed.weight[start:num_queries, :].unsqueeze(1).repeat(1, bs, 1)
            hs, memory = self.transformer(self.input_proj(src), mask, decoder_input, pos[-1],
                                          decoder_mask=self.attention_mask.to(patches_feature.device)[:num_queries, :num_queries])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        if self.feature_recon:
            outputs_feature = self.feature_align(hs)
            out = {'pred_logits': outputs_class[-1], 'pred_feature': outputs_feature[-1],
                   'gt_feature': patches_feature_gt,
                   'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_feature, patches_feature_gt)
        else:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = super()._set_aux_loss(outputs_class, outputs_coord)

        return out

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_feature, backbone_out):
        return [{'pred_logits': a, 'pred_boxes' : b, 'pred_feature': c, 'gt_feature': backbone_out}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_feature[:-1])]
