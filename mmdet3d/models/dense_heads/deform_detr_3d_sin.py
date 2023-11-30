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
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet3d.core.bbox import xyzwhlr2xyzxyzr

import numpy as np
from IPython import embed
from mmcv.runner import get_dist_info

@HEADS.register_module()
class DeformableDETRHead3D_sin(DETRHead):
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
        ##### init anchor_3d_head #########
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.box_code_size = bbox_coder.code_size
        ##### init DETR #########        
        self._init_detr__(*args, transformer=transformer, **kwargs)
        #### init BEV INFO ####
        self.bev_range = bev_range
        self.bev_x = self.bev_range[3] - self.bev_range[0]
        self.bev_y = self.bev_range[4] - self.bev_range[1]
        self.bev_z = self.bev_range[5] - self.bev_range[2]
        assert self.bev_x == 100 and self.bev_y == 100 and self.bev_z == 8
        
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
                             iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
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
        # enze hack use softmax ce loss
        use_detr_softmax_ce_loss = True
        if class_weight is not None and use_detr_softmax_ce_loss:
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
        self.loss_iou = build_loss(loss_iou)

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
        # cls branch, 就一层FC
        cls_branch = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch, fc-fc-fc
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
        # dir branch, 就一层FC
        if self.use_direction_classifier:
            dir_cls_branch = Linear(self.embed_dims, 2)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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
        
        '''这一步是为了让初始化的box都长一样大'''
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
            constant_init(m[-2], 0, bias=0)
            constant_init(m[-3], 0, bias=0)
            constant_init(m[-4], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-3].bias.data, -4.0)
        
        
    def forward(self, mlvl_feats):    
        batch_size = mlvl_feats[0].size(0)
        # mask=0 表示不屏蔽attention, 1表示屏蔽attention
        img_masks = mlvl_feats[0].new_zeros((batch_size, self.bev_x, self.bev_y))
        
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
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
        outputs_classes = []
        outputs_coords = []
        outputs_dirs = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # 分类分支预测
            outputs_class = self.cls_branches[lvl](hs[lvl]) # [1,300,10]
            # 先过两个基础的fc
            reg_base = self.reg_branches[lvl][:-4]
            # 分别预测 xyz, whl, r, vxvy
            reg_xyz, reg_whl, reg_r, reg_vxvy = self.reg_branches[lvl][-4:]
            tmp = hs[lvl]
            for layer in reg_base:
                tmp = layer(tmp)
            xyz = reg_xyz(tmp)
            whl = reg_whl(tmp)
            r = reg_r(tmp)
            vxvy = reg_vxvy(tmp)
            assert reference.shape[-1] == 2
            # 预测xyz相对ref point的offset, z强行赋予0
            xyz[..., :2] += reference; xyz[..., 2] += 0
            # xyz, whl都归一化处理, 其他的直接回归
            xyz = xyz.sigmoid()
            whl = whl.sigmoid()
            outputs_coord = torch.cat([xyz, whl, r, vxvy], -1)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            #方向dir的分类
            outputs_dir = self.dir_cls_branches[lvl](hs[lvl])
            outputs_dirs.append(outputs_dir)
            
        outputs_classes = torch.stack(outputs_classes) # [6, 1, 300, 10]
        outputs_coords = torch.stack(outputs_coords) # [6, 1, 300, 9]
        outputs_dirs = torch.stack(outputs_dirs) # [6, 1, 300, 2]
        
        return outputs_classes, outputs_coords, outputs_dirs

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list', 'all_dir_preds'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             all_dir_preds,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        
        assert gt_bboxes_ignore is None
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        
        all_gt_dirs_list = []
        for _ in range(num_dec_layers):
            tmp = []
            for idx in range(len(gt_bboxes_list)):
                _gt_dirs = get_direction_target(
                    gt_bboxes_list[idx].tensor,
                    self.dir_offset,
                    num_bins=2,
                    one_hot=False).to(gt_labels_list[0].device)
                tmp.append(_gt_dirs)
                
            all_gt_dirs_list.append(tmp)
           
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        '''loss = loss_cls + loss_bbox + loss_iou + loss_dir'''
        losses_cls, losses_bbox_l1, losses_bbox_iou, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_dir_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_dirs_list,
            img_metas_list, all_gt_bboxes_ignore_list)
        
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox_l1[-1]
        loss_dict['loss_iou'] = losses_bbox_iou[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_l1_i, loss_bbox_iou_i, loss_dir_i in zip(losses_cls[:-1],
                                              losses_bbox_l1[:-1],
                                              losses_bbox_iou[:-1],
                                              losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_l1_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_bbox_iou_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
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
        
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        dir_preds_list  = [dir_preds[i]  for i in range(num_imgs)]
        
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

        # 计算 classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        
        #####
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()    
        bbox_preds = bbox_preds.reshape(-1, 9)
        if self.diff_rad_by_sin:
            bbox_preds, bbox_targets = self.add_sin_difference(
                bbox_preds, bbox_targets)
        # regression L1 loss
        '''
        1. 先把bbox转成真实坐标, 再转成xyzxyz
        2. 对于l1 loss，要把xyzxyz归一化后算
        3. 对于iou loss, 要基于xyzxyz真实坐标算
        '''
        def hack_box_convert(box, factor):
            box = box * factor
            box = xyzwhlr2xyzxyzr(box)
            real_xyzxyz, rvxvy = box.split(6, dim=1)
            norm_xyzxyz = real_xyzxyz / factor[:3].repeat(2)
            real_box = torch.cat([real_xyzxyz, rvxvy], 1)
            norm_box = torch.cat([norm_xyzxyz, rvxvy], 1)
            return real_box, norm_box
        
        # 暂时写死这个 factor_xyz_whl_r_vxvy
        factor_xyz_whl_r_vxvy = bbox_targets.new_tensor([100,100,8,20,20,20,1,1,1])
        real_bbox_preds, norm_bbox_preds = hack_box_convert(bbox_preds, factor_xyz_whl_r_vxvy)
        real_bbox_targets, norm_bbox_targets = hack_box_convert(bbox_targets, factor_xyz_whl_r_vxvy)
        # embed(header='debug vdasdad')
        # 计算 bbox l1 loss
        loss_bbox = self.loss_bbox(norm_bbox_preds*bbox_weights,
                                   norm_bbox_targets*bbox_weights, 
                                   bbox_weights,
                                   avg_factor=num_total_pos)
        
        # TODO 计算 bbox iou loss 
        real_bbox_preds_xyzwhl = real_bbox_preds[:, :6]
        real_bbox_targets_xyzwhl = real_bbox_targets[:, :6]
        bbox_weights = bbox_weights[:, :6].mean(-1)
        loss_iou = self.loss_iou(real_bbox_preds_xyzwhl, 
                                real_bbox_targets_xyzwhl,
                                weight=bbox_weights,
                                avg_factor=num_total_pos)


        # 计算dir loss
        dir_preds = dir_preds.reshape(-1, 2)
        loss_dir = self.loss_dir(dir_preds, dir_targets, dir_weights, avg_factor=num_total_pos)
        
        return loss_cls, loss_bbox, loss_iou, loss_dir
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    dir_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_dirs_list, 
                    img_metas,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None
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
        '''3d box的center xyz要归一化'''
        gt_bboxes = gt_bboxes.tensor.to(bbox_pred)
        factor = bbox_pred.new_tensor([self.bev_x, self.bev_y, self.bev_z]).unsqueeze(0)
        offset = bbox_pred.new_tensor(self.bev_range[:3]).unsqueeze(0)
        gt_bboxes[:, :3] -= offset
        gt_bboxes[:, :3] /= factor
        '''3d box的whl要归一化'''
        gt_bboxes[:,3:6] /= 20
        
        code_weight = self.train_cfg.get('code_weight', None)
        code_weight = bbox_pred.new_tensor(code_weight)
        #
        assign_weight = self.train_cfg.get('assign_weight', None)
        assign_weight = bbox_pred.new_tensor(assign_weight)
        
        '''匹配检查一下  gt和gt匹配'''
        # todo 修改成3d assign
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
                   test_dec_layer=-1,
                   ref_points=None,
                   rescale=False):
        
        # embed(header='layers')
        cls_scores = all_cls_scores[test_dec_layer]
        bbox_preds = all_bbox_preds[test_dec_layer]
        dir_preds = all_dir_preds[test_dec_layer]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            dir_pred = dir_preds[img_id]
            input_meta = img_metas[img_id]
            proposals = self._get_bboxes_single(cls_score, bbox_pred, dir_pred,
                                                input_meta, ref_points, rescale)
            result_list.append(proposals)
        return result_list
    
    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           dir_pred,
                           input_meta,
                           ref_points=None,
                           rescale=None):
        
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
            if ref_points is not None:
                ref_points = ref_points[bbox_index]
            dir_scores = dir_pred[bbox_index].max(-1)[1]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]
            if ref_points is not None:
                ref_points = ref_points[bbox_index]
            dir_scores = dir_pred[bbox_index].max(-1)[1]
        
        '''center xyz是归一化到0-1的，需要映射为正常值'''
        factor = bbox_pred.new_tensor([self.bev_x, self.bev_y, self.bev_z]).unsqueeze(0)
        offset = bbox_pred.new_tensor(self.bev_range[:3]).unsqueeze(0)
        bbox_pred[:,:3] *= factor
        bbox_pred[:,:3] += offset
        '''whl 是归一化到0-1的，需要映射为正常值'''
        bbox_pred[:,3:6] *= 20
        det_bboxes = bbox_pred
        
        dir_rot = limit_period(det_bboxes[..., 6] - self.dir_offset,
                               self.dir_limit_offset, np.pi)
        det_bboxes[..., 6] = (dir_rot + self.dir_offset + np.pi * dir_scores.to(det_bboxes.dtype))
        det_bboxes = input_meta['box_type_3d'](det_bboxes, box_dim=self.box_code_size)
        
        return det_bboxes, scores, det_labels
    
    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2


def get_direction_target(gt_bboxes,
                         dir_offset=0.7854,
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
