import json
import os
import pickle

import cv2
import math
import random

import mmcv
import numpy as np
import _pickle as cPickle
from config.config import *
from datasets.data_augmentation import defor_2D, get_rotation
FLAGS = flags.FLAGS

import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tools.eval_utils import load_depth, get_bbox
from tools.dataset_utils import *
import matplotlib.pyplot as plt
import open3d as o3d
from absl import app
class CADMaskDataset(data.Dataset):
    def __init__(self, source=None, mode='train', data_dir=None,
                 n_pts=1024, img_size=256, per_obj=''):
        '''

        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        # self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size

        self.models = {}
        with open(os.path.join(self.data_dir, 'id2point.pkl'), 'rb') as f:
            self.models = pickle.load(f)

        # assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['train.txt','test.txt']
        # model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
        #                    'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']

        if mode == 'train':
            del img_list_path[1]
        else:
            del img_list_path[0]

        self.data_dir = os.path.join(self.data_dir, 'CAD_Mask_Expend')
        img_list = [os.path.join('Masks',line.strip())
                    for line in open(os.path.join(self.data_dir, img_list_path[0]))]
        # subset_len = [len(img_list)]

        self.cat_names = ["bathtub", "bed", "bin", "bookcase", "cabinet", "chair", "display", "sofa", "table"]
        self.cat_name2id = {
            "bathtub": 1,
            "bed": 2,
            "bin": 3,
            "bookcase": 4,
            "cabinet": 5,
            "chair": 6,
            "display": 7,
            "sofa": 8,
            "table": 9
        }
        self.id2cat_name = {
            '1': 'bathtub',
            '2': 'bed',
            '3': 'bin',
            '4': 'bookcase',
            '5': 'cabinet',
            '6': 'chair',
            '7': 'display',
            '8': 'sofa',
            '9': 'table'
        }
        self.catid2_catname = {
            '03337140': 'cabinet',
            '02818832': 'bed',
            '04256520': 'sofa',
            '03001627': 'chair',
            '02747177': 'bin',
            '02933112': 'cabinet',
            '03211117': 'display',
            '04379243': 'table',
            '02871439': 'bookcase',
            '02808440': 'bathtub'
        }

        self.id2cat_name_CAMERA = {
            '1': '02808440',
            '2': '02818832',
            '3': '02747177',
            '4': '02871439',
            '5': '02933112',
            '6': '03001627',
            '7': '03211117',
            '8': '04256520',
            '9': '04379243'
        }

        self.shapenetid2cat_name_CAMERA = {
             '02808440':1,
             '02818832':2,
             '02747177':3,
             '02871439':4,
             '02933112':5,
             '03001627':6,
             '03211117':7,
             '04256520':8,
             '04379243':9
        }


        self.id2cat_name = self.id2cat_name_CAMERA
        self.per_obj = per_obj
        self.per_obj_id = self.cat_name2id[self.per_obj]
        # only train one object
        if self.per_obj in self.cat_names:
            cat_name_id = self.cat_name2id[self.per_obj]
            catid_cad = self.id2cat_name_CAMERA[str(cat_name_id)]
            img_list_obj = [name for name in img_list if catid_cad in name]

            # img_list_obj = 0


            #  if use only one dataset
            #  directly load all data
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)

        print('{} images found.'.format(self.length))
        # print('{} models loaded.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #   load ground truth
        #  if per_obj is specified, then we only select the target object
        # index = index % self.length  # here something wrong
        img_path = os.path.join(self.data_dir, self.img_list[index])

        # mask_path = img_path + '_mask.png'
        mask = cv2.imread(img_path)
        if mask is not None:
            mask = mask[:,:,2]
        else:
            return self.__getitem__((index + 1) % self.__len__())
        mask[mask>100]=255
        mask[mask<=100] = 0
        im_H,im_W = mask.shape[0], mask.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
        # aggragate information about the selected object
        # inst_id = gts['instance_ids'][idx]
        plt.figure()
        plt.imshow(mask)
        plt.show()
        instance_id = np.unique(mask)[1]
        instance = np.argwhere(mask == instance_id)
        min_y,min_x = np.min(instance,axis=0)
        max_y,max_x = np.max(instance,axis=0)
        bboxes = [max_y, min_x, min_y, max_x]
        rmin, rmax, cmin, cmax = get_bbox(bboxes)
        # here resize and crop to a fixed size 256 x 256
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_DZI(FLAGS, bbox_xyxy, im_H, im_W)
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)

        ## roi_image ------------------------------------
        # roi_img = crop_resize_by_warp_affine(
        #     rgb, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        # ).transpose(2, 0, 1)

        # show roi_image
        # plt.figure()
        # plt.imshow(roi_img.transpose(1,2,0))
        # plt.show()

        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        mask_target = mask.copy().astype(float)
        mask_target[mask != instance_id] = 0.0
        mask_target[mask == instance_id] = 1.0

        # plt.figure()
        # plt.imshow(mask_target)
        # plt.show()

        # depth[mask_target == 0.0] = 0.0
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        )
        # plt.figure()
        # plt.imshow(roi_mask)
        # plt.show()

        roi_mask = np.expand_dims(roi_mask, axis=0)
        catid_cad = img_path.split('/')[-2].split('-')[0]
        cat_id = self.shapenetid2cat_name_CAMERA[catid_cad]  # convert to 0-indexed
        # note that this is nocs model, normalized along diagonal axis
        model_name = img_path.split('/')[-2]

        model = self.models[model_name]
        # model = self.models[model_name].astype(float)  # 1024 points
        nocs_scale = [1,1,1]  # nocs_scale = image file / model file
        # transfer nocs_scales(3) ---> nocs_scales(1), np.mean()
        nocs_scale = np.mean(nocs_scale)
        
        # fsnet scale (from model) scale residual
        fsnet_scale, mean_shape = self.get_fs_net_scale(self.id2cat_name[str(cat_id + 1)], model, nocs_scale)
        fsnet_scale = fsnet_scale / 1000.0
        mean_shape = mean_shape / 1000.0

        filename = img_path.split('/')[-1]
        rotation = self.get_rotation(float(filename.split('-')[0]), float(filename.split('-')[1]), float(filename.split('-')[2].split('.')[0]))
        translation = [0,0,1.5]

        # dense depth map
        # dense_depth = depth_normalize

        # sym
        sym_info = np.array([0, 0, 0, 0], dtype=int)
        # sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=mug_handle)

        # add nnoise to roi_mask
        roi_mask_def = defor_2D(roi_mask, rand_r=FLAGS.roi_mask_r, rand_pro=FLAGS.roi_mask_pro)

        pcl_in = self._sample_points(model, FLAGS.random_points)

        # generate augmentation parameters
        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()

        out_camK = np.asarray([
            [435.19,    0.,  239.9,   0.],
            [  0.,  435.19,  179.91, 0.  ],
            [  0.,    0.,    1.,    0.  ],
            [  0.,    0.,    0.,    1.  ]]
        ) # 720 * 960

        data_dict = {}
        data_dict['pcl_in'] = torch.as_tensor(pcl_in.astype(float)).contiguous()
        # data_dict['roi_img'] = torch.as_tensor(roi_img.astype(float)).contiguous()
        # data_dict['roi_depth'] = torch.as_tensor(roi_depth.astype(float)).contiguous()
        # data_dict['dense_depth'] = torch.as_tensor(dense_depth.astype(float)).contiguous()
        # data_dict['depth_normalize'] = torch.as_tensor(depth_normalize.astype(float)).contiguous()
        data_dict['cam_K'] = torch.as_tensor(out_camK.astype(float)).contiguous()
        data_dict['roi_mask'] = torch.as_tensor(roi_mask.astype(float)).contiguous()
        data_dict['cat_id'] = torch.as_tensor(cat_id, dtype=torch.float32).contiguous()
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
        data_dict['translation'] = torch.as_tensor(translation, dtype=torch.float32).contiguous()
        data_dict['fsnet_scale'] = torch.as_tensor(fsnet_scale, dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_info.astype(float)).contiguous()
        data_dict['roi_coord_2d'] = torch.as_tensor(roi_coord_2d, dtype=torch.float32).contiguous()
        data_dict['mean_shape'] = torch.as_tensor(mean_shape, dtype=torch.float32).contiguous()
        data_dict['aug_bb'] = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
        data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
        data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        data_dict['roi_mask_deform'] = torch.as_tensor(roi_mask_def, dtype=torch.float32).contiguous()
        data_dict['model_point'] = torch.as_tensor(model, dtype=torch.float32).contiguous()
        data_dict['nocs_scale'] = torch.as_tensor(nocs_scale, dtype=torch.float32).contiguous()

        return data_dict

    def get_rotation(self, theta_x: float, theta_y: float, theta_z: float):

        # 将角度转换为弧度
        roll_rad = np.radians(theta_x)
        pitch_rad = np.radians(theta_y)
        yaw_rad = np.radians(theta_z)

        # 计算旋转矩阵
        Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                       [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                       [0, 0, 1]])

        Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                       [0, 1, 0],
                       [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll_rad), -np.sin(roll_rad)],
                       [0, np.sin(roll_rad), np.cos(roll_rad)]])

        # 旋转矩阵相乘得到最终的旋转矩阵
        R = np.matmul(np.matmul(Rz, Ry), Rx)

        return R

    def _sample_points(self, pcl, n_pts):
        """ Down sample the point cloud using farthest point sampling.

        Args:
            pcl (torch tensor or numpy array):  NumPoints x 3
            num (int): target point number
        """
        total_pts_num = pcl.shape[0]
        if total_pts_num < n_pts:
            pcl = np.concatenate([np.tile(pcl, (n_pts // total_pts_num, 1)), pcl[:n_pts % total_pts_num]], axis=0)
        elif total_pts_num > n_pts:
            ids = np.random.permutation(total_pts_num)[:n_pts]
            pcl = pcl[ids]
        return pcl

    def _depth_to_pcl(self, depth, K, xymap, mask):
        K = K.reshape(-1)
        cx, cy, fx, fy = K[2], K[5], K[0], K[4]
        depth = depth.reshape(-1).astype(float)
        valid = ((depth > 0) * mask.reshape(-1)) > 0
        depth = depth[valid]
        x_map = xymap[0].reshape(-1)[valid]
        y_map = xymap[1].reshape(-1)[valid]
        real_x = (x_map - cx) * depth / fx
        real_y = (y_map - cy) * depth / fy
        pcl = np.stack((real_x, real_y, depth), axis=-1)
        return pcl.astype(float)

    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=float), np.array([dx, dy, dz], dtype=float) / 1000.0, Rm


    def get_fs_net_scale(self, c, model, nocs_scale):
        # model pc x 3
        lx = max(model[:, 0]) - min(model[:, 0])
        ly = max(model[:, 1]) - min(model[:, 1])
        lz = max(model[:, 2]) - min(model[:, 2])

        # real scale
        lx_t = lx * nocs_scale * 1000
        ly_t = ly * nocs_scale * 1000
        lz_t = lz * nocs_scale * 1000

        if c == 'bottle':
            unitx = 87
            unity = 220
            unitz = 89
        elif c == 'bowl':
            unitx = 165
            unity = 80
            unitz = 165
        elif c == 'camera':
            unitx = 88
            unity = 128
            unitz = 156
        elif c == 'can':
            unitx = 68
            unity = 146
            unitz = 72
        elif c == 'laptop':
            unitx = 346
            unity = 200
            unitz = 335
        elif c == 'mug':
            unitx = 146
            unity = 83
            unitz = 114
        elif c == '02876657':
            unitx = 324 / 4
            unity = 874 / 4
            unitz = 321 / 4
        elif c == '02880940':
            unitx = 675 / 4
            unity = 271 / 4
            unitz = 675 / 4
        elif c == '02942699':
            unitx = 464 / 4
            unity = 487 / 4
            unitz = 702 / 4
        elif c == '02946921':
            unitx = 450 / 4
            unity = 753 / 4
            unitz = 460 / 4
        elif c == '03642806':
            unitx = 581 / 4
            unity = 445 / 4
            unitz = 672 / 4
        elif c == '03797390':
            unitx = 670 / 4
            unity = 540 / 4
            unitz = 497 / 4
        else:
            unitx = 0
            unity = 0
            unitz = 0
            # print('This category is not recorded in my little brain.')
            # raise NotImplementedError
        # scale residual
        return np.array([lx_t - unitx, ly_t - unity, lz_t - unitz]), np.array([unitx, unity, unitz])

    def get_sym_info(self, c, mug_handle=1):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=int)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=int)
        elif c == 'camera':
            sym = np.array([0, 0, 0, 0], dtype=int)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=int)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=int)
        elif c == 'mug' and mug_handle == 1:
            sym = np.array([0, 1, 0, 0], dtype=int)  # for mug, we currently mark it as no symmetry
        elif c == 'mug' and mug_handle == 0:
            sym = np.array([1, 0, 0, 0], dtype=int)
        else:
            sym = np.array([0, 0, 0, 0], dtype=int)
        return sym

import open3d as o3d
def main(argv):
    test_data = CADMaskDataset(source=None, mode='train', data_dir='/home/aston/Desktop/Datasets/pose_data', per_obj='bin')
    loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=2,
        num_workers=1,
        shuffle=True
    )
    for sample in loader:
        pc = sample['pcl_in']
        for index in range(pc.shape[0]):
            pc_temp = pc[index].cpu().numpy()
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(pc_temp)
            o3d.visualization.draw_geometries([pc_o3d])
        break
    print('hello world !')

if __name__ == '__main__':
    app.run(main)
