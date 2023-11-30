# -*- coding: utf-8 -*-
# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, build_bbox_coder, build_assigner, build_sampler, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils import build_transformer
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.detr_head import DETRHead, AnchorFreeHead
from mmdet3d.core import limit_period
from ..builder import build_loss

import numpy as np
from IPython import embed

@HEADS.register_module()
class DeformableDETRHead3DDebug1(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.
    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.
    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 ##### anchor_3d_head #########
                 use_direction_classifier=True,
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2),
                 bev_range=[-50, -50, -5, 50, 50, 3],
                 ##### anchor_3d_head #########
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        ##### anchor_3d_head #########
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.box_code_size = bbox_coder.code_size
        ##### anchor_3d_head #########
        
        #### BEV INFO ####
        self.bev_range = bev_range
        self.bev_x = self.bev_range[3] - self.bev_range[0]
        self.bev_y = self.bev_range[4] - self.bev_range[1]
        self.bev_z = self.bev_range[5] - self.bev_range[2]
        assert self.bev_x == 100 and self.bev_y == 100 and self.bev_z == 8
        
        self._init_detr__(*args, transformer=transformer, **kwargs)
        
        self.loss_dir = build_loss(loss_dir)
        self.bbox_coder = build_bbox_coder(bbox_coder)        
        
    def _init_detr__(self, 
                     num_classes,
                     in_channels,
                     num_query=100,
                     num_reg_fcs=2,
                     transformer=None,
                     sync_cls_avg_factor=False,
                     positional_encoding=dict(
                         type='SinePositionalEncoding',
                         num_feats=128,
                         normalize=True),
                     loss_cls=dict(
                         type='CrossEntropyLoss',
                         bg_cls_weight=0.1,
                         use_sigmoid=False,
                         loss_weight=1.0,
                         class_weight=1.0),
                     loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                     loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                     train_cfg=dict(
                         assigner=dict(
                             type='HungarianAssigner3D',
                             cls_cost=dict(type='ClassificationCost', weight=1.0),
                             reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                             dir_cost=dict(type='DirL1Cost', weight=1.0))),
                     test_cfg=dict(max_per_img=100),
                     init_cfg=None,
                     **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()
        
        
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        cls_branch = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = nn.ModuleList()
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())            
        xyz_dim, whl_dim, r_dim, vxvy_dim = 3,3,1,2
        assert xyz_dim+whl_dim+r_dim+vxvy_dim == self.box_code_size
        reg_branch.append(Linear(self.embed_dims, xyz_dim))
        reg_branch.append(Linear(self.embed_dims, whl_dim))
        reg_branch.append(Linear(self.embed_dims, r_dim))
        reg_branch.append(Linear(self.embed_dims, vxvy_dim))
        
        if self.use_direction_classifier:
            dir_cls_branch = Linear(self.embed_dims, 2)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            if self.use_direction_classifier:
                self.dir_cls_branches = _get_clones(dir_cls_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
            if self.use_direction_classifier:
                self.dir_cls_branches = nn.ModuleList([dir_cls_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
            
    def forward(self, mlvl_feats):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        
        batch_size = mlvl_feats[0].size(0)
        img_masks = mlvl_feats[0].new_ones((batch_size, self.bev_x, self.bev_y))
        
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
        
        
        query_embeds = None
        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None, 
                    cls_branches=self.cls_branches if self.as_two_stage else None)
        hs = hs.permute(0, 2, 1, 3) # [6, 1, 300, 256=C]

        # embed(header='debug ddert head forward.....')
        
        outputs_classes = []
        outputs_coords = []
        outputs_dirs = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl]) # [1,300,10]
             
            reg_base = self.reg_branches[lvl][:-4]
            reg_xyz, reg_whl, reg_r, reg_vxvy = self.reg_branches[lvl][-4:]
            tmp = hs[lvl]
            for layer in reg_base:
                tmp = layer(tmp)
            xyz = reg_xyz(tmp)
            whl = reg_whl(tmp) # zhiding:要归一化嘛？
            r = reg_r(tmp)
            vxvy = reg_vxvy(tmp)
            assert reference.shape[-1] == 2
            xyz[..., :2] += reference
            # reference xy z强行赋予0
            xyz[..., 2] += 0
            # center sigmoid归一化，其他的直接回归
            xyz = xyz.sigmoid()
            whl = whl.exp()
            outputs_coord = torch.cat([xyz, whl, r, vxvy], -1)
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            #方向dir的分类
            outputs_dir = self.dir_cls_branches[lvl](hs[lvl])
            outputs_dirs.append(outputs_dir)
            
        outputs_classes = torch.stack(outputs_classes) # [6, 1, 300, 10]
        outputs_coords = torch.stack(outputs_coords) # [6, 1, 300, 9]
        outputs_dirs = torch.stack(outputs_dirs) # [6, 1, 300, 2]
        
        # embed(header='debug ddetr head 3d .....')
                
        return outputs_classes, outputs_coords, outputs_dirs

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list', 'all_dir_preds'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             all_dir_preds, # 3d
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        ## TODO, generate all_gt_dirs_list-> Done
        all_gt_dirs_list = []
        for _ in range(num_dec_layers):
            _gt_dirs = get_direction_target(
                gt_bboxes_list[0].tensor,
                self.dir_offset,
                num_bins=2,
                one_hot=False).to(gt_labels_list[0].device)
            all_gt_dirs_list.append([_gt_dirs])
             
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # embed(header='debug detr 3d loss')
        '''loss = loss_cls + loss_bbox + loss_dir
        '''
        losses_cls, losses_bbox_l1, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_dir_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_dirs_list,
            img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox_l1[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        # embed(header='debug detr loss')
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_l1_i, losses_dir_i in zip(losses_cls[:-1],
                                                       losses_bbox_l1[:-1],
                                                       losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_l1_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = losses_dir_i
            num_dec_layer += 1
        return loss_dict
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    dir_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_dirs_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        dir_preds_list =  [dir_preds[i]  for i in range(num_imgs)]
        
        
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, dir_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_dirs_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, 
         bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        dir_targets = torch.cat(dir_targets_list, 0)
        dir_weights = torch.cat(dir_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()    
        bbox_preds = bbox_preds.reshape(-1, 9)
        # regression L1 loss
        # 让中心点按照正常大小去回归
        bbox_weights[:,:2] *= 10
        loss_bbox = self.loss_bbox(bbox_preds*bbox_weights, bbox_targets*bbox_weights, bbox_weights, avg_factor=num_total_pos)
        
        # embed(header='debug detr 3d loss')
        
        # dir loss
        dir_preds = dir_preds.reshape(-1, 2)
        loss_dir = self.loss_dir(dir_preds, dir_targets, dir_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_dir
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    dir_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_dirs_list, 
                    img_metas,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, 
            cls_scores_list, bbox_preds_list, dir_preds_list,
            gt_bboxes_list, gt_labels_list, gt_dirs_list,
            img_metas, gt_bboxes_ignore_list)
        
        # embed(header='debug detr loss')
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, 
                bbox_targets_list, bbox_weights_list,
                dir_targets_list, dir_weights_list,
                num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score, bbox_pred, dir_pred,
                           gt_bboxes, gt_labels, gt_dirs,
                           img_meta, gt_bboxes_ignore=None):
        
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        '''二分图匹配'''
        ''' TODO: 3d box的center xyz要归一化， 其他不归一化'''
        gt_bboxes = gt_bboxes.tensor.to(bbox_pred)
        factor = bbox_pred.new_tensor([self.bev_x, self.bev_y, self.bev_z]).unsqueeze(0)
        offset = bbox_pred.new_tensor(self.bev_range[:3]).unsqueeze(0)
        gt_bboxes[:, :3] -= offset
        gt_bboxes[:, :3] /= factor
        
        code_weight = self.train_cfg.get('code_weight', None)
        code_weight = bbox_pred.new_tensor(code_weight)
        
        assign_weight = self.train_cfg.get('assign_weight', None)
        assign_weight = bbox_pred.new_tensor(assign_weight)
        assign_result = self.assigner.assign(bbox_pred, cls_score, dir_pred,
                                             gt_bboxes, gt_labels, gt_dirs,
                                             img_meta, gt_bboxes_ignore, assign_weight)
        
        # embed(header='ddetr 3d assign')
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_weights *= code_weight
        
        # dir targets
        dir_targets = gt_bboxes.new_zeros((num_bboxes, ), dtype=torch.long)
        dir_weights = gt_bboxes.new_zeros((num_bboxes, ), dtype=torch.long)
        dir_targets[pos_inds] = gt_dirs[sampling_result.pos_assigned_gt_inds]
        dir_weights[pos_inds] = 1.0
        
        pos_gt_bboxes_targets = sampling_result.pos_gt_bboxes
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
                
        return (labels, label_weights, 
                bbox_targets, bbox_weights,
                dir_targets, dir_weights,
                pos_inds, neg_inds)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list', 'all_dir_preds'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   all_dir_preds,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.
        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        dir_preds = all_dir_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            dir_pred = dir_preds[img_id]
            input_meta = img_metas[img_id]
            proposals = self._get_bboxes_single(cls_score, bbox_pred, dir_pred,
                                                input_meta, rescale)
            result_list.append(proposals)
        return result_list
    
    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           dir_pred,
                           input_meta,
                           rescale=None):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.
        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.
        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.
                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
            dir_scores = dir_pred[bbox_index].max(-1)[1]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]
            dir_scores = dir_pred[bbox_index].max(-1)[1]
        
        '''center xyz是归一化到0-1的，需要映射为正常值'''
        factor = bbox_pred.new_tensor([self.bev_x, self.bev_y, self.bev_z]).unsqueeze(0)
        offset = bbox_pred.new_tensor(self.bev_range[:3]).unsqueeze(0)
        bbox_pred[:,:3] *= factor
        bbox_pred[:,:3] += offset
        det_bboxes = bbox_pred
        
        dir_rot = limit_period(det_bboxes[..., 6] - self.dir_offset, self.dir_limit_offset, np.pi)
        det_bboxes[..., 6] = (dir_rot + self.dir_offset + np.pi * dir_scores.to(det_bboxes.dtype))
        det_bboxes = input_meta['box_type_3d'](det_bboxes, box_dim=self.box_code_size)
        
        # embed(header='ddetr 3d test')
        return det_bboxes, scores, det_labels


def get_direction_target(gt_bboxes,
                         dir_offset=0.7854, #0,
                         num_bins=2,
                         one_hot=False):
    """Encode direction to 0 ~ num_bins-1.

    Args:
        anchors (torch.Tensor): Concatenated multi-level anchor.
        reg_targets (torch.Tensor): Bbox regression targets.
        dir_offset (int): Direction offset.
        num_bins (int): Number of bins to divide 2*PI.
        one_hot (bool): Whether to encode as one hot.

    Returns:
        torch.Tensor: Encoded direction targets.
    """
    rot_gt = gt_bboxes[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_targets = torch.zeros(
            *list(dir_cls_targets.shape),
            num_bins,
            dtype=anchors.dtype,
            device=dir_cls_targets.device)
        dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets
