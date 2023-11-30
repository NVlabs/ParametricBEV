import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose, RandomFlip, LoadImageFromFile
from IPython import embed

from pdb import set_trace as st

@PIPELINES.register_module()
class MultiViewPipeline:
    def __init__(self, transforms, n_images):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        
    def __sort_list(self, old_list, order):
        new_list = []
        for i in order:
            new_list.append(old_list[i])
        return new_list
        

    def __call__(self, results):
        imgs = []
        extrinsics = []
        # lidar2cam = []
        # cam2img = []
        ids = np.arange(len(results['img_info']))
        replace = True if self.n_images > len(ids) else False
        # ids = np.random.choice(ids, self.n_images, replace=replace) # Disable shuffle of 6 images
        ids_list = ids.tolist()
        for i in ids_list:
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])

        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        # resort 2d box by random ids
        if 'gt_bboxes' in results.keys():
            gt_bboxes = self.__sort_list(results['gt_bboxes'], ids_list)
            gt_labels = self.__sort_list(results['gt_labels'], ids_list)
            gt_bboxes_ignore = self.__sort_list(results['gt_bboxes_ignore'], ids_list)
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
        
        results['lidar2img']['extrinsic'] = extrinsics
        return results

@PIPELINES.register_module()
class MultiViewPipeline_UB:
    def __init__(self, transforms, n_images):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        
    def __sort_list(self, old_list, order):
        new_list = []
        for i in order:
            new_list.append(old_list[i])
        return new_list
        

    def __call__(self, results):
        imgs = []
        extrinsics = []
        nlspn_depth = []
        # lidar2cam = []
        # cam2img = []
        ids = np.arange(len(results['img_info']))
        replace = True if self.n_images > len(ids) else False
        # ids = np.random.choice(ids, self.n_images, replace=replace) # Disable shuffle of 6 images
        ids_list = ids.tolist()
        for i in ids_list:
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])
            nlspn_depth.append(_results['nlspn_depth'])

        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        results['nlspn_depth'] = nlspn_depth
        # resort 2d box by random ids
        if 'gt_bboxes' in results.keys():
            gt_bboxes = self.__sort_list(results['gt_bboxes'], ids_list)
            gt_labels = self.__sort_list(results['gt_labels'], ids_list)
            gt_bboxes_ignore = self.__sort_list(results['gt_bboxes_ignore'], ids_list)
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
        
        results['lidar2img']['extrinsic'] = extrinsics
        return results

@PIPELINES.register_module()
class RandomShiftOrigin:
    def __init__(self, std):
        self.std = std

    def __call__(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results


@PIPELINES.register_module()
class KittiSetOrigin:
    def __init__(self, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def __call__(self, results):
        results['lidar2img']['origin'] = self.origin.copy()
        return results


@PIPELINES.register_module()
class KittiRandomFlip:
    def __call__(self, results):
        if results['flip']:
            results['lidar2img']['intrinsic'][0, 2] = -results['lidar2img']['intrinsic'][0, 2] + \
                                                      results['ori_shape'][1]
            flip_matrix_0 = np.eye(4, dtype=np.float32)
            flip_matrix_0[0, 0] *= -1
            flip_matrix_1 = np.eye(4, dtype=np.float32)
            flip_matrix_1[1, 1] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0]
            extrinsic = flip_matrix_0 @ extrinsic @ flip_matrix_1.T
            results['lidar2img']['extrinsic'][0] = extrinsic
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 0], -center[:, 1]) - alpha
            center_flip = center
            center_flip[:, 1] *= -1
            alpha_flip = np.arctan2(center_flip[:, 0], -center_flip[:, 1]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results


@PIPELINES.register_module()
class SunRgbdSetOrigin:
    def __call__(self, results):
        intrinsic = results['lidar2img']['intrinsic'][:3, :3]
        extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
        projection = intrinsic @ extrinsic
        h, w, _ = results['ori_shape']
        center_2d_3 = np.array([w / 2, h / 2, 1], dtype=np.float32)
        center_2d_3 *= 3
        origin = np.linalg.inv(projection) @ center_2d_3
        results['lidar2img']['origin'] = origin
        return results


@PIPELINES.register_module()
class SunRgbdTotalLoadImageFromFile(LoadImageFromFile):
    def __call__(self, results):
        file_name = results['img_info']['filename']
        flip = file_name.endswith('_flip.jpg')
        if flip:
            results['img_info']['filename'] = file_name.replace('_flip.jpg', '.jpg')
        results = super().__call__(results)
        if flip:
            results['img'] = results['img'][:, ::-1]
        return results


@PIPELINES.register_module()
class SunRgbdRandomFlip:
    def __call__(self, results):
        if results['flip']:
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
            results['lidar2img']['extrinsic'][0][:3, :3] = flip_matrix @ extrinsic @ flip_matrix.T
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 1], center[:, 0]) - alpha
            center_flip = center @ flip_matrix
            alpha_flip = np.arctan2(center_flip[:, 1], center_flip[:, 0]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results



@PIPELINES.register_module()
class ProjectLidarPtsToImage:
    """
    Project Lidar points to each image frame and save in a list.
    For each image, pts exceed max_pts will be droped.

    This module require lidar points pre-loaded. results['points'] should exist.

    """
    def __init__(self, max_pts):
        self.max_pts = max_pts

    def project_pts_to_cam_space(self, points, img, lidar2img_rt, max_distance=100):
        """Project the 3D points cloud into 2D image coordinate.
        This function is modified from mmdet3d/core/visualizer/image_vis.py#L9
        Args:
            points (numpy.array): 3D points cloud (x, y, z) to visualize.
            img (numpy.array): The numpy array of image.
            lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
                according to the camera intrinsic parameters.
            max_distance (float): the max distance of the points cloud.
                Default: 100.
        """
        
        num_points = points.shape[0]
        pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img_rt.T

        # cam_points is Tensor of Nx4 whose last column is 1
        # transform camera coordinate to image coordinate
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        fov_inds = ((pts_2d[:, 0] < img.shape[2])
                    & (pts_2d[:, 0] >= 0)
                    & (pts_2d[:, 1] < img.shape[1])
                    & (pts_2d[:, 1] >= 0))

        imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

        if imgfov_pts_2d.shape[0] > self.max_pts: # Only return at most max_pts number of points.
            imgfov_pts_2d = imgfov_pts_2d[:max_pts] 
        
        # for i in range(10):
        #     print(len(imgfov_pts_2d))

        return imgfov_pts_2d

    def __call__(self, results):
        assert 'points' in results, "ProjectLidarPtsToImage module require Lidar points to be loaded first."
        projectd_pts = []
        extrinsics = results['lidar2img']['extrinsic']
        lidar_pts = results['points'].data
        imgs = results['img'].data
        for i in range(len(results['img_info'])):
            extrinsic = extrinsics[i]
            imgfov_pts_2d = self.project_pts_to_cam_space(lidar_pts,imgs[i],extrinsic)
            projectd_pts.append(imgfov_pts_2d)

        results['projected_2d_lidar_points'] = projectd_pts

        return results

@PIPELINES.register_module()
class ProjectLidarPtsToDepthMap:
    """
    Project Lidar points to each image frame and save as a depth map.
    For each image, pts exceed max_pts will be droped.

    This module require lidar points pre-loaded. results['points'] should exist.

    """
    def __init__(self, max_pts):
        self.max_pts = max_pts

    def project_pts_to_cam_space(self, points, img, lidar2img_rt, max_distance=100):
        """Project the 3D points cloud into 2D image coordinate.
        This function is modified from mmdet3d/core/visualizer/image_vis.py#L9
        Args:
            points (numpy.array): 3D points cloud (x, y, z) to visualize.
            img (numpy.array): The numpy array of image.
            lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
                according to the camera intrinsic parameters.
            max_distance (float): the max distance of the points cloud.
                Default: 100.
        """
        
        num_points = points.shape[0]
        pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img_rt.T

        # cam_points is Tensor of Nx4 whose last column is 1
        # transform camera coordinate to image coordinate
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        fov_inds = ((pts_2d[:, 0] < img.shape[2])
                    & (pts_2d[:, 0] >= 0)
                    & (pts_2d[:, 1] < img.shape[1])
                    & (pts_2d[:, 1] >= 0))

        imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

        if imgfov_pts_2d.shape[0] > self.max_pts: # Only return at most max_pts number of points.
            imgfov_pts_2d = imgfov_pts_2d[:self.max_pts] 
        

        frame = np.zeros([int(img.shape[1]/4),int(img.shape[2]/4)]) # TODO: Hardcode to 1/4 image size
        for pts_idx in range(imgfov_pts_2d.shape[0]):
            depth = imgfov_pts_2d[pts_idx,2]
            uu = int(imgfov_pts_2d[pts_idx,0]/4)
            vv = int(imgfov_pts_2d[pts_idx,1]/4)
            # print(uu,vv,depth)
            frame[vv,uu] = depth

        return frame

    def __call__(self, results):
        assert 'points' in results, "ProjectLidarPtsToImage module require Lidar points to be loaded first."
        lidar_depth_maps = []
        extrinsics = results['lidar2img']['extrinsic']
        lidar_pts = results['points'].data
        imgs = results['img'].data
        for i in range(len(results['img_info'])):
            extrinsic = extrinsics[i]
            lidar_depth_map = self.project_pts_to_cam_space(lidar_pts,imgs[i],extrinsic)
            lidar_depth_maps.append(lidar_depth_map)

        results['lidar_depth_maps'] = lidar_depth_maps

        return results