# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# -*- coding: utf-8 -*-
model = dict(
    type='PDBEVNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=256, out_channels=128),
    neck_depth=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=16,
        num_outs=4),
    neck_depth_fuse=dict(in_channels=64, out_channels=2),
    feature_lifting='PD_OCC_biased',
    neck_3d=dict(
        type='PDBEVNeck',
        in_channels=128,
        out_channels=256,
        num_layers=6,
        stride=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    seg_head=dict(
        type='PDBEV_FCNHead',
        use_centerness=False,
        is_transpose=False,
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=4,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_ce=dict(type='CrossEntropyLoss_PDBEV',use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss_PDBEV', loss_weight=1.0),
        ignore_index=255,
        split_vis_occ_seg_loss=False,
        seg_vis_region_weight=1.0,
        seg_occ_region_weight=1.0
    ),
    bbox_head=dict(
        type='FreeAnchor3DHead',
        is_transpose=False,
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        num_convs=2,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            # scales=[1, 2, 4],
            sizes=[
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    
    n_voxels=(400, 400, 12),
    voxel_size=[0.25, 0.25, 0.5],
    pd_loss=True, # Supervision for parametric depth
    pd_loss_weight_nll=0.1,
    pd_loss_weight_reg=0.0,
    pd_loss_weight_l1=0.0,
    # model training and testing settings
    train_cfg = dict(
        assigner=dict(
            type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False),
    test_cfg = dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500))

# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'NuScenesMultiView_Map_Dataset_PD'
data_root = 'data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True, # For training only.
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=1,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bev_seg=True),
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='ProjectLidarPtsToDepthMap',max_pts=50000),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bev_seg', 'lidar_depth_maps'], 
                meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'img_info', 'lidar2cam', 'img_filename'
                            ])]
test_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'], 
                meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'img_info', 'lidar2cam', 'img_filename'
                            ])]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            with_box2d=True,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_box2d=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False
    )

total_epochs = 30
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(name='2022-06-30_pdbev_joint_sup_r50_biased',project='3d-joint')), 
    ])
evaluation = dict(interval=32)
dist_params = dict(backend='nccl', nccl_timeout_s=36000) # timeout in seconds. Timeout is extended for the time consuming evaluation per epoch.
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
# load_from = None
load_from = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
resume_from = None
workflow = [('train', 1)]

# fp16 settings, the loss scale is specifically tuned to avoid Nan
fp16 = dict(loss_scale='dynamic')
