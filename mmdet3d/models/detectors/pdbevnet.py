# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
import mmcv
from mmseg.ops import resize
from IPython import embed
from mmcv.runner import get_dist_info

from mmdet3d.models.losses.parametric_depth_loss import Parametric_Depth_NLL_Loss

import sys
sys.setrecursionlimit(10000)

@DETECTORS.register_module()
class PDBEVNet(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 neck_fuse,
                 neck_depth,
                 neck_depth_fuse,
                 feature_lifting,
                 neck_3d,
                 bbox_head,
                 seg_head,
                 n_voxels,
                 voxel_size,
                 head_2d=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 extrinsic_noise=0,
                 pd_dist='gaussian',
                 pd_loss=False,
                 pd_loss_weight_nll=1.0,
                 pd_loss_weight_reg=1.0,
                 pd_loss_weight_l1=1.0,
                 save_occ=False,
                 fix_sigma=0):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_fuse = nn.Conv2d(neck_fuse['in_channels'], neck_fuse['out_channels'],
                                   kernel_size=3, stride=1, padding=1)
        self.neck_depth = build_neck(neck_depth)
        self.neck_depth_fuse = nn.Conv2d(neck_depth_fuse['in_channels'], neck_depth_fuse['out_channels'],
                            kernel_size=3, stride=1, padding=1)
        self.neck_3d = build_neck(neck_3d)
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        self.seg_head = build_seg_head(seg_head) if seg_head is not None else None
        
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise

        # Parametric depth parameters
        self.feature_lifting = feature_lifting
        self.pd_dist = pd_dist
        self.pd_loss = pd_loss
        self.pd_loss_weight_nll = pd_loss_weight_nll
        self.pd_loss_weight_reg = pd_loss_weight_reg
        self.pd_loss_weight_l1 = pd_loss_weight_l1

        # DEBUG settings
        self.save_occ = save_occ
        self.fix_sigma = fix_sigma
        self.tmp_idx = 0
        
    def extract_feat(self, img, img_metas, mode, lidar_depth_maps=None, fix_sigma=0, input_depth=None):

        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)
        
        # use for vovnet
        if isinstance(x, dict):
            tmp = []
            for k in x.keys():
                tmp.append(x[k])
            x = tmp

        # fuse pdepth features
        d1,d2,d3,d4 = self.neck_depth(x)
        d2 = resize(d2, size=d1.size()[2:], mode='bilinear',align_corners=False)
        d3 = resize(d3, size=d1.size()[2:], mode='bilinear',align_corners=False)
        d4 = resize(d4, size=d1.size()[2:], mode='bilinear',align_corners=False)
        
        pdepths_raw = self.neck_depth_fuse(torch.cat([d1,d2,d3,d4], dim=1))
        pdepths_raw = pdepths_raw.reshape([batch_size, -1] + list(pdepths_raw.shape[1:]))

        pdepths_raw = torch.nn.functional.relu(pdepths_raw, inplace=False) # Parametric depth (mu,sigma) [1, 6, 2, 232, 400]
        mu = pdepths_raw[:,:,0]
        sigma = pdepths_raw[:,:,1]+0.1 # Make sure sigma is not zero.
        pdepths_est = torch.stack([mu,sigma],dim=2) 

        # fuse bev features
        c1,c2,c3,c4 = self.neck(x)
        c2 = resize(c2, size=c1.size()[2:], mode='bilinear',align_corners=False)
        c3 = resize(c3, size=c1.size()[2:], mode='bilinear',align_corners=False)
        c4 = resize(c4, size=c1.size()[2:], mode='bilinear',align_corners=False)

        x = self.neck_fuse(torch.cat([c1,c2,c3,c4], dim=1))
        x = x.reshape([batch_size, -1] + list(x.shape[1:]))

        pdepths = pdepths_est.type_as(x)

        stride = img.shape[-1] / x.shape[-1]
        assert stride == 4  
        stride = int(stride)
        
        # reconstruct 3d voxels using parametric depth
        if self.feature_lifting == 'PD_OCC_biased':
            volumes, valids = [], []
            for feature, img_meta, pdepth in zip(x, img_metas, pdepths):            
                projection = self._compute_projection_fixed(img_meta, stride, noise=self.extrinsic_noise).to(x.device)
                points = get_points(
                    n_voxels=torch.tensor(self.n_voxels),
                    voxel_size=torch.tensor(self.voxel_size),
                    origin=torch.tensor(img_meta['lidar2img']['origin'])).to(x.device)
                
                height = img_meta['img_shape'][0] // stride
                width = img_meta['img_shape'][1] // stride

                # Use parametric depth likelihood and occupancy weight to lift and compress feature
                volume, valid = backproject_pdepth_occ_weight_biased(feature[:, :, :height, :width], points, projection, pdepth[:, :, :height, :width],img_meta)
                volume = volume.sum(dim=0)
                valid = valid.sum(dim=0)
                volume[:,valid[0]>0] = volume[:,valid[0]>0]/valid[:,valid[0]>0]
                valid = valid > 0
                volume[:, ~valid[0]] = .0
                volumes.append(volume)
                valids.append(valid)
                
            x = torch.stack(volumes)
            valids = torch.stack(valids)
            x = self.neck_3d(x)

        return x, valids, pdepths_est

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg, lidar_depth_maps, **kwargs):
        feature_bev, valids, est_pdepths = self.extract_feat(img, img_metas, 'train', lidar_depth_maps, self.fix_sigma)
        assert self.bbox_head is not None or self.seg_head is not None
        
        losses = dict()
        ###### Parametric depth loss
        if self.pd_loss: # Enable depth supervision

            losses_nll = []
            losses_depth_reg = []
            losses_depth_l1 = []
            nll_valid_pts = 0.0
            mean_depth_error = 0.0
            mean_depth_sigma = 0.0

            for view_idx in range(img.shape[1]):
                est_pdepth = est_pdepths[:,view_idx]
                lidar_depth = lidar_depth_maps[view_idx]
                valid_lidar_depth_mask = lidar_depth>0
                valid_est_mu = est_pdepth[:,0,:,:][valid_lidar_depth_mask]
                valid_est_sigma = est_pdepth[:,1,:,:][valid_lidar_depth_mask]
                valid_lidar_depth = lidar_depth[valid_lidar_depth_mask]

                loss_pd, valid_pts = parametric_depth_NLL_loss(valid_est_mu,valid_est_sigma,valid_lidar_depth,dist='laplacian')
                losses_nll.append(loss_pd)

                loss_depth_sigma_reg = torch.mean(valid_est_sigma)
                losses_depth_reg.append(loss_depth_sigma_reg)

                nll_valid_pts += valid_pts

                l1 = torch.abs(valid_lidar_depth-valid_est_mu)
                losses_depth_l1.append(l1.mean())
                mean_depth_error += l1.mean()/img.shape[1]

                mean_sigma = torch.mean(valid_est_sigma)
                mean_depth_sigma += mean_sigma/img.shape[1]

            loss_depth_NLL = self.pd_loss_weight_nll*torch.sum(torch.stack(losses_nll))/img.shape[1]
            loss_depth_var_reg = self.pd_loss_weight_reg*torch.sum(torch.stack(losses_depth_reg))/img.shape[1]
            loss_depth_l1 = self.pd_loss_weight_l1*torch.sum(torch.stack(losses_depth_l1))/img.shape[1]
            
            loss_depth = dict()

            if self.pd_loss_weight_nll > 0:
                loss_depth['loss_depth_NLL']=loss_depth_NLL
                loss_depth['nll_valid_pts']=nll_valid_pts.detach()
            if self.pd_loss_weight_l1 > 0:
                loss_depth['loss_depth_l1']=loss_depth_l1

            loss_depth['mean_depth_error']=mean_depth_error.detach()
            loss_depth['mean_depth_sigma']=mean_depth_sigma.detach()

            losses.update(loss_depth)

        ###### Detection loss
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)
        ###### Segmentation loss
        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            if self.seg_head.split_vis_occ_seg_loss:
                gt_bev = gt_bev_seg[0][None, ...].long()
                occ_mask = gt_bev[:,:,:,2].bool()
                gt_bev = gt_bev[:,:,:,:2]
                # Visible region
                gt_bev_vis = gt_bev.clone()
                gt_bev_vis[:,:,:,0][~occ_mask] = 255 # Set occluded regions to ignore_index
                gt_bev_vis[:,:,:,1][~occ_mask] = 255 # Only use labels from visible regions
                loss_seg_vis_tmp = self.seg_head.losses(x_bev, gt_bev_vis)
                loss_seg_vis = {
                    'loss_seg_dice_vis':loss_seg_vis_tmp['loss_seg_dice']*self.seg_head.seg_vis_region_weight,
                    'loss_seg_ce_vis':loss_seg_vis_tmp['loss_seg_ce']*self.seg_head.seg_vis_region_weight,
                    'iou_road_vis':loss_seg_vis_tmp['iou_road'],
                    'iou_lane_vis':loss_seg_vis_tmp['iou_lane']
                }
                losses.update(loss_seg_vis)
                # Occluded region
                gt_bev_occ = gt_bev.clone()
                gt_bev_occ[:,:,:,0][occ_mask] = 255 # Set visible regions to ignore_index
                gt_bev_occ[:,:,:,1][occ_mask] = 255 # Only use labels from occluded regions
                loss_seg_occ_tmp = self.seg_head.losses(x_bev, gt_bev_occ)
                loss_seg_occ = {
                    'loss_seg_dice_occ':loss_seg_occ_tmp['loss_seg_dice']*self.seg_head.seg_occ_region_weight,
                    'loss_seg_ce_occ':loss_seg_occ_tmp['loss_seg_ce']*self.seg_head.seg_occ_region_weight,
                    'iou_road_occ':loss_seg_occ_tmp['iou_road'],
                    'iou_lane_occ':loss_seg_occ_tmp['iou_lane']
                }
                losses.update(loss_seg_occ)
            else:
                gt_bev = gt_bev_seg[0][None, ...].long()
                # Apply seg loss on all regions.
                loss_seg = self.seg_head.losses(x_bev, gt_bev)
                losses.update(loss_seg)
        
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        # not supporting aug_test for now
        return self.simple_test(img, img_metas, kwargs)

    def simple_test(self, img, img_metas, kwargs):
        feature_bev, valids, est_pdepths = self.extract_feat(img, img_metas, 'test')

        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=valids.float())
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list]
        else:
            bbox_results = [dict()]
                
        # BEV semantic seg
        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            bbox_results[0]['bev_seg'] = x_bev
        
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass

    def show_results(self, *args, **kwargs):
        pass

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3]+noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    @staticmethod
    def _compute_projection_fixed(img_meta, stride, noise=0):
        projection = []
        intrinsics = torch.tensor(img_meta['cam2img'])[:,:3,:3] # [6,3,3]
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsics[:,:2,:] /= ratio
        extrinsics = torch.tensor(img_meta['lidar2cam']) # [6,4,4]
        
        for cam_idx in range(intrinsics.shape[0]):
            if noise > 0:
                projection.append(intrinsics[cam_idx] @ extrinsics[cam_idx,:,:3].T+noise)
            else:
                projection.append(intrinsics[cam_idx] @ extrinsics[cam_idx,:,:3].T)
        return torch.stack(projection)

@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points

# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device).type_as(features)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid

# Parametric depth weighted back projection
def backproject_pdepth_occ_weight_biased(features, points, projection, pdepth, img_meta):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection.type_as(points), points)
    x = (points_2d_3[:, 0] / points_2d_3[:, 2])
    y = (points_2d_3[:, 1] / points_2d_3[:, 2])
    z = points_2d_3[:, 2]

    # Valid mask 
    x_round = x.round().long()
    y_round = y.round().long()
    valid = (x_round >= 0) & (y_round >= 0) & (x_round < width) & (y_round < height) & (z > 0)
    valid_volume = valid.view(n_images, n_x_voxels, n_y_voxels, n_z_voxels)

    # Normalize x y for gridsampling
    x_normalized = x / ((width - 1) / 2) - 1
    y_normalized = y / ((height - 1) / 2) - 1
    grid = torch.stack((x_normalized, y_normalized), dim=2)  # [n_images, n_points, 2]

    # Init volumes
    feature_volume_flattened = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device).type_as(features)
    feature_volume = torch.zeros((n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels), device=features.device).type_as(features)
    prob_volume_flattened = torch.zeros((n_images, 1, points.shape[-1]), device=features.device).type_as(features)
    occ_volume_flattened = torch.zeros((n_images, 1, points.shape[-1]), device=features.device).type_as(features)

    occlusion_volume_flattened = torch.zeros((n_images, 1, points.shape[-1]), device=features.device).type_as(features)
    soft_occlusion_volume_flattened = torch.ones((n_images, 1, points.shape[-1]), device=features.device).type_as(features)

    # Compress Z axis using occupancy weight
    for i in range(n_images):

        valid_grid = grid[i,valid[i],:].unsqueeze(0)
        fetched_features = F.grid_sample(features[i].unsqueeze(0), valid_grid[None,:,:], mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(2).squeeze(0).type_as(features)
        mu = F.grid_sample(pdepth[i,0].unsqueeze(0).unsqueeze(1), valid_grid[None,:,:], mode='bilinear', padding_mode='border', align_corners=False).squeeze(2).squeeze(1).squeeze(0)
        sigma = F.grid_sample(pdepth[i,1].unsqueeze(0).unsqueeze(1), valid_grid[None,:,:], mode='bilinear', padding_mode='border', align_corners=False).squeeze(2).squeeze(1).squeeze(0)

        # 1. Calculate depth probability likelihood
        projected_depth = z[i,valid[i]]
        prob = distribution(mu,sigma,projected_depth,dist="laplacian")
        prob = prob+0.001 # Add bias
        prob_volume_flattened[i, :, valid[i]] = prob[None,:].type_as(prob_volume_flattened)

        # 2. Calculate occupancy distribution
        occ_raw = prob_volume_flattened[i].view(1,n_x_voxels, n_y_voxels, n_z_voxels)
        occ_dist = occ_raw.clone()/(occ_raw.sum(dim=3,keepdim=True)+0.01) 
        occ_flattened = occ_dist.view(1,-1)[:,valid[i]].type_as(features)
        occ_volume_flattened[i, :, valid[i]] = occ_flattened.type_as(occ_volume_flattened)

        # 3. Compress feature using occupancy weight
        feature_volume_flattened[i, :, valid[i]] = occ_flattened*fetched_features
        feature_volume[i] = feature_volume_flattened[i].view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)

        # Extra outputs
        ## Hard visibility
        occlusion_mask = projected_depth>mu
        occlusion_volume_flattened[i,:,valid[i]] = occlusion_mask[None,:].type_as(occlusion_volume_flattened)

        ## Soft visibility using CDF of parametric depth distribution
        occ_soft_prob = occlusion_CDF(mu,sigma,projected_depth)
        soft_occlusion_volume_flattened[i, :, valid[i]] = occ_soft_prob[None,:].type_as(soft_occlusion_volume_flattened)

    # 4. Compress Z axis
    feature_volume_compressed = feature_volume.sum(dim=4)
    valid_weight = valid_volume.any(dim=3).unsqueeze(1)

    ##### Additional outputs #####
    write_additional_outputs = False
    if write_additional_outputs:
        from datetime import datetime
        import numpy as np
        now = datetime.now().time()
        sample_idx = img_meta['sample_idx']

        occlusion_volume = occlusion_volume_flattened.view(6, 1, n_x_voxels, n_y_voxels, n_z_voxels).bool()
        occlusion_volume = occlusion_volume.all(dim=4)
        occlusion_volume = occlusion_volume.any(dim=0)
        occlusion_volume = -occlusion_volume.float()+1

        # Generate soft occlusion volume
        soft_occlusion_volume = soft_occlusion_volume_flattened.view(6, 1, n_x_voxels, n_y_voxels, n_z_voxels)
        soft_occlusion_volume, _ = soft_occlusion_volume.min(dim=0) # max visibility on all views ==> [1, X, Y, Z]
        soft_occlusion_volume, _ = soft_occlusion_volume.min(dim=3) # max visibility on all heights ==> [1, X, Y]
        soft_occlusion_volume = (soft_occlusion_volume-soft_occlusion_volume.min())/(soft_occlusion_volume.max()-soft_occlusion_volume.min())
        soft_occlusion_volume = -soft_occlusion_volume + 1

        # Generate occupancy volume 
        occupancy_volume = occ_volume_flattened.view(6, 1, n_x_voxels, n_y_voxels, n_z_voxels)
        occupancy_volume, _ = occupancy_volume.max(dim=0) # [1, X, Y, Z]
        occupancy_volume = occupancy_volume/(occupancy_volume.sum(dim=-1,keepdim=True)+0.001)

        # 1. Hard occlusion map
        print('trash/final_additional_outputs/vis_hard_'+str(now)+'_'+sample_idx+'.png')
        mmcv.imwrite(255*occlusion_volume[0].data.cpu().numpy(), 'trash/final_additional_outputs/vis_hard_'+str(now)+'_'+sample_idx+'.png')
        np.save('trash/final_additional_outputs/vis_hard_'+str(now)+'_'+sample_idx,occlusion_volume.data.cpu().numpy())

        # 2. Soft occlusion map
        print("soft_occlusion_volume.min()",soft_occlusion_volume.min())
        print("soft_occlusion_volume.max()",soft_occlusion_volume.max())
        print('trash/final_additional_outputs/vis_soft_'+str(now)+'_'+sample_idx+'.png')
        mmcv.imwrite(255*soft_occlusion_volume[0].float().data.cpu().numpy(), 'trash/final_additional_outputs/vis_soft_'+str(now)+'_'+sample_idx+'.png')
        np.save('trash/final_additional_outputs/vis_soft_'+str(now)+'_'+sample_idx,soft_occlusion_volume.data.cpu().numpy())

    return feature_volume_compressed, valid_weight


# Parametric depth distribution functions
def gaussian(mu, sigma, labels):
    return torch.exp(-0.5*(mu-labels)** 2/ sigma** 2)/sigma

def laplacian(mu, b, labels):
    return 0.5 * torch.exp(-(torch.abs(mu-labels)/b))/b

def distribution(mu, sigma, labels, dist="laplacian"):
    return gaussian(mu, sigma, labels) if dist=="gaussian" else \
           laplacian(mu, sigma, labels)

def parametric_depth_NLL_loss(mu, sigma, labels, dist='laplacian'):
    neg_log_likelihood = torch.abs(mu - labels)/sigma + torch.log(sigma)
    valid_neg_log_likelihood = neg_log_likelihood[~torch.isinf(neg_log_likelihood)]
    valid_pts = (~torch.isinf(neg_log_likelihood)).sum()
    return valid_neg_log_likelihood.mean(), valid_pts

def occlusion_CDF(mu, b, labels):
    larger_mask = labels >= mu
    smaller_mask = labels < mu
    occlusion_cdf = torch.zeros(mu.shape,device=mu.device)
    occlusion_cdf[smaller_mask] = 0.5 * torch.exp((labels[smaller_mask]-mu[smaller_mask])/b[smaller_mask])
    occlusion_cdf[larger_mask] = 1 - 0.5 * torch.exp(-(labels[larger_mask]-mu[larger_mask])/b[larger_mask])
    return occlusion_cdf


