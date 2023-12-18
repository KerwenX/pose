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
class MaskDataset(data.Dataset):
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

        # assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['train.txt','test.txt']
        # model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
        #                    'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']

        if mode == 'train':
            del img_list_path[1]
        else:
            del img_list_path[0]

        img_list = [os.path.join('ScanNOCS',line.strip())
                    for line in open(os.path.join(self.data_dir, img_list_path[0]))]
        subset_len = [len(img_list)]

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

        self.id2cat_name = self.id2cat_name_CAMERA
        self.per_obj = per_obj
        self.per_obj_id = self.cat_name2id[self.per_obj]
        # only train one object
        # TODO: modify this code
        if self.per_obj in self.cat_names:
            cat_name_id = self.cat_name2id[self.per_obj]
            catid_cad = self.id2cat_name_CAMERA[str(cat_name_id)]
            scan2cad_image_alignment_filepath = os.path.join(self.data_dir,'scan2cad_image_alignments.json')
            img_list_obj = []
            with open(scan2cad_image_alignment_filepath,'r') as f:
                image_alignment = json.load(f)['alignments']
                for key in image_alignment:
                    for model_dict in image_alignment[key]:
                        if model_dict['catid_cad'] == catid_cad:
                            img_list_obj.append(os.path.join('ScanNOCS',key))

            # img_list_obj = 0


            #  if use only one dataset
            #  directly load all data
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)

        # models = {}
        # for path in model_file_path:
        #     with open(os.path.join(data_dir, path), 'rb') as f:
        #         models.update(cPickle.load(f))
        # self.models = models

        # move the center to the body of the mug
        # meta info for re-label mug category
        # with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        #     self.mug_meta = cPickle.load(f)

        self.camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=float)  # [fx, fy, cx, cy]
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=float)

        # full path : /home/aston/Desktop/Datasets/pose_data/intrinsics/scene0000_00/intrinscics_color.txt
        self.intrinsics_prefix = '/home/aston/Desktop/Datasets/pose_data/intrinsics'

        self.invaild_list = []
        # self.mug_sym = mmcv.load(os.path.join(self.data_dir, 'Real/train/mug_handle.pkl'))

        self.models = {}
        with open(os.path.join(self.data_dir,'id2point.pkl'), 'rb') as f:
            self.models = pickle.load(f)

        print('{} images found.'.format(self.length))
        # print('{} models loaded.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #   load ground truth
        #  if per_obj is specified, then we only select the target object
        # index = index % self.length  # here something wrong
        img_path = os.path.join(self.data_dir, self.img_list[index])
        if img_path in self.invaild_list:
            return self.__getitem__((index + 1) % self.__len__())
        try:
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
        except:
            return self.__getitem__((index + 1) % self.__len__())

        # intrinsics
        scene = self.img_list[index].split('/')[1]
        intrinsics_file = os.path.join(self.intrinsics_prefix, scene, 'intrinsics_color.txt')
        out_camK = np.loadtxt(intrinsics_file)[:3,:3]
        # img_type = 'real'

        # select one foreground object,
        # if specified, then select the object
        if self.per_obj != '':
            if self.per_obj_id not in gts['class_ids']:
                return self.__getitem__((index+1)%self.__len__())
            idx = gts['class_ids'].index(self.per_obj_id)
        else:
            idx = random.randint(0, len(gts['instance_ids']) - 1)

        # rgb = cv2.imread(img_path + '_color.jpg')
        # cv2.imshow(f'image {idx}', rgb)
        # cv2.waitKey(0)
        # cv2.destoryAllWindows()

        # if rgb is not None:
        #     rgb = rgb[:, :, :3]
        # else:
        #     return self.__getitem__((index + 1) % self.__len__())

        # im_H, im_W = rgb.shape[0], rgb.shape[1]


        # depth_path = img_path + '_depth.png'
        # if os.path.exists(depth_path):
        #     depth = load_depth(depth_path)
        # else:
        #     return self.__getitem__((index + 1) % self.__len__())

        mask_path = img_path + '_mask.png'
        mask = cv2.imread(mask_path)
        if mask is not None:
            mask = mask[:, :, 2]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        im_H,im_W = mask.shape[0], mask.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
        # aggragate information about the selected object
        inst_id = gts['instance_ids'][idx]
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
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
        mask_target[mask != inst_id] = 0.0
        mask_target[mask == inst_id] = 1.0

        # depth[mask_target == 0.0] = 0.0
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        )
        # plt.figure()
        # plt.imshow(roi_mask)
        # plt.show()

        roi_mask = np.expand_dims(roi_mask, axis=0)
        # roi_depth = crop_resize_by_warp_affine(
        #     depth, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        # )

        # roi_depth = np.expand_dims(roi_depth, axis=0)
        # normalize depth
        # depth_valid = roi_depth > 0
        # if np.sum(depth_valid) <= 1.0:
        #     return self.__getitem__((index + 1) % self.__len__())
        # roi_m_d_valid = roi_mask.astype(bool) * depth_valid
        # if np.sum(roi_m_d_valid) <= 1.0:
        #     return self.__getitem__((index + 1) % self.__len__())
        #
        # depth_v_value = roi_depth[roi_m_d_valid]
        # if (np.max(depth_v_value) - np.min(depth_v_value))==0:
        #     depth_normalize = (roi_depth/np.min(depth_v_value))/2
        # else:
        #     depth_normalize = (roi_depth - np.min(depth_v_value)) / (np.max(depth_v_value) - np.min(depth_v_value))
        # depth_normalize[~roi_m_d_valid] = 0.0
        # cat_id, rotation translation and scale
        cat_id = gts['class_ids'][idx] - 1  # convert to 0-indexed
        # note that this is nocs model, normalized along diagonal axis
        model_name = gts['shapenet_instance_id'][idx]

        model = self.models[model_name]
        # model = self.models[model_name].astype(float)  # 1024 points
        nocs_scale = gts['scales'][idx]  # nocs_scale = image file / model file
        # transfer nocs_scales(3) ---> nocs_scales(1), np.mean()
        nocs_scale = np.mean(nocs_scale)
        
        # fsnet scale (from model) scale residual
        fsnet_scale, mean_shape = self.get_fs_net_scale(self.id2cat_name[str(cat_id + 1)], model, nocs_scale)
        fsnet_scale = fsnet_scale / 1000.0
        mean_shape = mean_shape / 1000.0
        rotation = gts['rotations'][idx]
        translation = gts['translations'][idx]

        # dense depth map
        # dense_depth = depth_normalize

        # sym
        sym_info = np.array([0, 0, 0, 0], dtype=int)
        # sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=mug_handle)

        # add nnoise to roi_mask
        roi_mask_def = defor_2D(roi_mask, rand_r=FLAGS.roi_mask_r, rand_pro=FLAGS.roi_mask_pro)

        # pcl_in = self._depth_to_pcl(roi_depth, out_camK, roi_coord_2d, roi_mask_def) / 1000.0
        # if len(pcl_in) < 50:
        #     return self.__getitem__((index + 1) % self.__len__())
        pcl_in = self._sample_points(model, FLAGS.random_points)

        # generate augmentation parameters
        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(roi_img.transpose((1,2,0)))
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(roi_mask.transpose((1,2,0)))
        # plt.show()


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
    test_data = MaskDataset(source=None, mode='train', data_dir='/home/aston/Desktop/Datasets/pose_data', per_obj='bed')
    loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=2,
        num_workers=8,
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
