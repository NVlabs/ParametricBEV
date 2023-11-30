# -*- coding: utf-8 -*-
import torch
import numpy as np
import os 
from mmdet.datasets import DATASETS
from .nuscenes_monocular_dataset_pd import NuScenesMultiViewDataset_PD
from .dataset_wrappers import MultiViewMixin
from IPython import embed
import mmcv
import skimage.io
import matplotlib.pyplot as plt
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import cv2
import imageio
from nuscenes.nuscenes import NuScenes
import tqdm
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from pyquaternion import Quaternion
import cv2
from tools.data_converter.nuscenes_converter import get_2d_boxes

from PIL import Image


@DATASETS.register_module()
class NuScenesMultiView_Map_Dataset_PD(NuScenesMultiViewDataset_PD):
    
    def __init__(self,
                 with_box2d=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)
        self.scene2map = get_scene2map(self.nusc)
        self.maps = get_nusc_maps(map_folder=self.data_root)
        # box 2d
        self.with_box2d = with_box2d
        
        xbound = [-50, 50, 0.5]
        ybound = [-50, 50, 0.5]
        zbound = [-10, 10, 20.0]
        dbound = [4.0, 45.0, 1.0]
        self.nx = np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]],dtype='int64') 
        self.dx = np.array([row[2] for row in [xbound, ybound, zbound]]) 
        self.bx = np.array([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        
        self.lane_thickness = 2
        for i in range(5):
            print('lane thickness: {}'.format(self.lane_thickness))
        
        self.debug = False
        
    def coord_transform(self, pts, pose):
        pts = convert_points_to_local(pts, pose)
        pts = np.round(
                (pts - self.bx[:2]) / self.dx[:2]
            ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        return pts

    def get_data_info(self, index):
        data_info = super().get_data_info(index)
        sample_token = data_info['sample_idx']
        
        if 'ann_info' in data_info:
            # get bev segm map
            bev_seg_gt = self._get_map_by_sample_token(sample_token)
            data_info['ann_info']['gt_bev_seg'] = bev_seg_gt
        
            # get bbox2d for camera
            if self.with_box2d:
                camera_types = [
                    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                info = self.data_infos[index]
                gt_bboxes_mv, gt_labels_mv, gt_bboxes_ignore_mv = [], [], []
                for cam in camera_types:
                    gt_bboxes, gt_labels, gt_bboxes_ignore = [], [], []
                    coco_infos = get_2d_boxes(self.nusc,
                                              info['cams'][cam]['sample_data_token'],
                                              visibilities=['', '1', '2', '3', '4'],
                                              mono3d=False)
                    for coco_info in coco_infos:
                        if coco_info is None:
                            continue
                        elif coco_info.get('ignore', False):
                            continue
                        x1, y1, w, h = coco_info['bbox']
                        inter_w = max(0, min(x1 + w, 1600) - max(x1, 0))
                        inter_h = max(0, min(y1 + h, 900) - max(y1, 0))
                        if inter_w * inter_h == 0:
                            continue
                        if coco_info['area'] <= 0 or w < 1 or h < 1:
                            continue
                        if coco_info['category_id']<0 or coco_info['category_id']>9:
                            continue
                        bbox = [x1, y1, x1 + w, y1 + h]
                        if coco_info.get('iscrowd', False):
                            gt_bboxes_ignore.append(bbox)
                        else:
                            gt_bboxes.append(bbox)
                            gt_labels.append(coco_info['category_id'])

                    if gt_bboxes:
                        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                        gt_labels = np.array(gt_labels, dtype=np.int64)
                    else:
                        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                        gt_labels = np.array([], dtype=np.int64)

                    if gt_bboxes_ignore:
                        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
                    else:
                        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

                    gt_bboxes_mv.append(gt_bboxes)
                    gt_labels_mv.append(gt_labels)
                    gt_bboxes_ignore_mv.append(gt_bboxes_ignore)

                data_info['ann_info']['bboxes'] = gt_bboxes_mv 
                data_info['ann_info']['labels'] = gt_labels_mv 
                data_info['ann_info']['bboxes_ignore'] = gt_bboxes_ignore_mv 
             
        return data_info
        

    def _get_map_by_sample_token(self, sample_token):
        egopose = self.nusc.get(
            'ego_pose', 
            self.nusc.get(
                'sample_data', 
                 self.nusc.get('sample', sample_token)['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        pose = [egopose['translation'][0], egopose['translation'][1],
                        np.cos(rot), np.sin(rot)]
        
        bev_seg_gt_road = np.zeros((self.nx[0], self.nx[1]))
        bev_seg_gt_lane = np.zeros((self.nx[0], self.nx[1]))
        
        tgt_type = ['road', 'lane']
        
        scene_name = self.nusc.get('scene', self.nusc.get('sample', sample_token)['scene_token'])['name']
        map_name = self.scene2map[scene_name]
        nmap = self.maps[map_name]
        
        if 'road' in tgt_type :
            records = getattr(nmap, 'drivable_area')
            for record in records:
                polygons = [nmap.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
                for poly in polygons:
                    # plot exterior
                    ext = self.coord_transform(np.array(poly.exterior.coords), pose)
                    bev_seg_gt_road = cv2.fillPoly(bev_seg_gt_road, [ext], 1)
                    
                    # plot interior
                    intes = [self.coord_transform(np.array(pi.coords), pose) for pi in poly.interiors]
                    bev_seg_gt_road = cv2.fillPoly(bev_seg_gt_road, intes, 0)
                    
        if 'lane' in tgt_type:
            for layer_name in ['road_divider', 'lane_divider']:
                records = getattr(nmap, layer_name)
                for record in records:
                    line = nmap.extract_line(record['line_token'])
                    if line.is_empty:
                        continue
                    line = self.coord_transform(np.array(line.xy).T, pose)
                    bev_seg_gt_lane = cv2.polylines(bev_seg_gt_lane, [line], isClosed=False,
                                                    color=1, thickness=self.lane_thickness)
        
        # need flip
        bev_seg_gt_road = np.flip(bev_seg_gt_road, axis=1).copy()
        bev_seg_gt_lane = np.flip(bev_seg_gt_lane, axis=1).copy()
        bev_seg_gt = np.stack([bev_seg_gt_road, bev_seg_gt_lane],axis=-1)
        
        return bev_seg_gt
    
    def evaluate(self, results, vis_mode=False, *args, **kwargs):
        
        eval_seg = 'bev_seg' in results[0]
        eval_det = 'boxes_3d' in results[0]
        assert eval_seg == True or eval_det == True
        
        new_bevseg_results = None
        new_det_results = None
        if eval_seg:
            new_bevseg_results = []
            new_bevseg_gts_road, new_bevseg_gts_lane = [], []
        if eval_det:
            new_det_results = []
        
        for i in range(len(results)):
            if eval_det:
                box_type = type(results[i]['boxes_3d'])
                boxes_3d = results[i]['boxes_3d'].tensor
                boxes_3d = box_type(boxes_3d, box_dim=9, origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
                new_det_results.append(dict(
                    boxes_3d=boxes_3d,
                    scores_3d=results[i]['scores_3d'],
                    labels_3d=results[i]['labels_3d']))

            if eval_seg:
                assert results[i]['bev_seg'].shape[0] == 1
                seg_pred = results[i]['bev_seg'][0]
                seg_pred_road, seg_pred_lane = seg_pred[0], seg_pred[1]
                seg_pred_road = (seg_pred_road.sigmoid()>0.5).int().data.cpu().numpy()
                seg_pred_lane = (seg_pred_lane.sigmoid()>0.5).int().data.cpu().numpy()
                
                new_bevseg_results.append(dict(seg_pred_road=seg_pred_road,
                                               seg_pred_lane=seg_pred_lane))
                
                # bev seg gt path
                seg_gt_path = 'data/nuscenes/maps_bev_seg_gt_2class/'
                if not mmcv.is_filepath(seg_gt_path):
                    # online generate map, too slow
                    if i == 0:
                        print('### first time need generate bev seg map online ###')
                    sample_token = self.get_data_info(i)['sample_idx']
                    seg_gt = self._get_map_by_sample_token(sample_token)
                    seg_gt_road, seg_gt_lane = seg_gt[..., 0], seg_gt[..., 1]
                    mmcv.imwrite(seg_gt_road, seg_gt_path+'road/{}.png'.format(i))
                    mmcv.imwrite(seg_gt_lane, seg_gt_path+'lane/{}.png'.format(i))   

                # load gt from local machine
                seg_gt_road = mmcv.imread(seg_gt_path+'road/{}.png'.format(i), flag='grayscale').astype('float64')
                seg_gt_lane = mmcv.imread(seg_gt_path+'lane/{}.png'.format(i), flag='grayscale').astype('float64')

                new_bevseg_gts_road.append(seg_gt_road)
                new_bevseg_gts_lane.append(seg_gt_lane)

        
        if vis_mode:
            print('### vis nus test data ###')
            self.show(new_det_results, 'trash/final_add_vis_mini', bev_seg_results=new_bevseg_results, thr=0.3)
            embed(header='### vis nus test data ###')
            print('### finish vis ###')
            exit()
        
        result_dict = dict()            
        if eval_det:
            # eval detection
            result_dict = super().evaluate(new_det_results, *args, **kwargs)
        if eval_seg:
            # eval segmentation
            bev_res_dict = self.evaluate_bev(new_bevseg_results,
                                             new_bevseg_gts_road,
                                             new_bevseg_gts_lane)
            for k in bev_res_dict.keys():
                result_dict[k] = bev_res_dict[k]
        return result_dict
    
    
    def evaluate_bev(self,
                     new_bevseg_results,
                     new_bevseg_gts_road,
                     new_bevseg_gts_lane):
        from mmseg.core import eval_metrics
        assert len(new_bevseg_results) == len(new_bevseg_gts_road) == len(new_bevseg_gts_lane)
        print('### evaluate BEV segmentation start ###')
        categories = ['road', 'lane']
        
        results_road, results_lane = [], []
        for i in range(len(new_bevseg_results)):
            seg_pred_road = new_bevseg_results[i]['seg_pred_road']
            seg_pred_lane = new_bevseg_results[i]['seg_pred_lane']
            results_road.append(seg_pred_road)
            results_lane.append(seg_pred_lane)
        
        ret_metrics_road = eval_metrics(results_road,
                                        new_bevseg_gts_road,
                                        num_classes=2,
                                        ignore_index=255)
        
        ret_metrics_lane = eval_metrics(results_lane,
                                        new_bevseg_gts_lane,
                                        num_classes=2,
                                        ignore_index=255)
        
        IoU_road = ret_metrics_road['IoU'][-1]
        IoU_lane = ret_metrics_lane['IoU'][-1]
        IoU = [IoU_road, IoU_lane]
        res_dict = dict()

        for idx, c in enumerate(categories):
            print("{} IoU: {:.3f}".format(c, IoU[idx]))
            res_dict[c] = IoU[idx]
        print('### evaluate BEV segmentation finish ###')
        return res_dict

def get_scene2map(nusc):
    scene2map = {}
    for rec in nusc.scene:
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    return scene2map

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def convert_points_to_local(points, pose):
    points -= pose[:2]
    rot = get_rot(np.arctan2(pose[3], pose[2])).T
    points = np.dot(points, rot)
    return points

def get_nusc_maps(map_folder='./nuScenes_v1.0-mini'):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                 map_name=map_name) for map_name in [
                    "singapore-hollandvillage",
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

