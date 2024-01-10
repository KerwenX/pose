import json
import os

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from tqdm import tqdm
import time
import numpy as np
import trimesh
from trimesh import load as load_model, viewer
import open3d as o3d

import sys
sys.path.append('/home/aston/Desktop/python/pose')
from scan2cad_rasterizer import Rasterizer
from absl import app
from tools.geom_utils import generate_RT
from config.config import *
from network.HSPose import HSPose
FLAGS = flags.FLAGS
def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def aug_bbox_DZI(bbox_xyxy, im_H, im_W):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
        center, scale
    """
    DZI_TYPE = "uniform"
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    if DZI_TYPE.lower() == "uniform":
        scale_ratio = 1 + 0.25 * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = 0.25 * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * 1.5
    elif DZI_TYPE.lower() == "roi10d":
        # shift (x1,y1), (x2,y2) by 15% in each direction
        _a = -0.15
        _b = 0.15
        x1 += bw * (np.random.rand() * (_b - _a) + _a)
        x2 += bw * (np.random.rand() * (_b - _a) + _a)
        y1 += bh * (np.random.rand() * (_b - _a) + _a)
        y2 += bh * (np.random.rand() * (_b - _a) + _a)
        x1 = min(max(x1, 0), im_W)
        x2 = min(max(x1, 0), im_W)
        y1 = min(max(y1, 0), im_H)
        y2 = min(max(y2, 0), im_H)
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        scale = max(y2 - y1, x2 - x1) * 1.5
    elif DZI_TYPE.lower() == "truncnorm":
        raise NotImplementedError("DZI truncnorm not implemented yet.")
    else:
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = max(y2 - y1, x2 - x1)
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def _sample_points(pcl, n_pts):
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

def as_mesh(scene_or_mesh):
    """
    解决Trimesh.load_mesh()方法读取不到mesh对象的问题：
    from: https://github.com/mikedh/trimesh/issues/507#issuecomment-514973337

    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mody_mesh = None  # empty scene
        else:
            # we lose texture information here
            mody_mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mody_mesh = scene_or_mesh
    return mody_mesh


def trimesh_to_open3d(trimesh_mesh):
    # 将trimesh mesh的顶点和面转换为numpy数组
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    # 创建open3d mesh对象
    open3d_mesh = o3d.geometry.TriangleMesh()

    # 为open3d mesh设置顶点和面
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # 计算法向量
    open3d_mesh.compute_vertex_normals()

    return open3d_mesh

def compute_RT_errors(sRT_1, sRT_2):
    """
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation
    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    if sRT_1 is None or sRT_2 is None:
        return -1
    try:
        assert np.array_equal(sRT_1[3, :], sRT_2[3, :])
        assert np.array_equal(sRT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT_1[3, :], sRT_2[3, :])
        exit()

    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]

    R = R1 @ R2.transpose()
    cos_theta = (np.trace(R) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


catname2id = {
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
def predict(argv):

    #  GT
    anno_file = '/home/aston/Desktop/Datasets/pose_data/scan2cad_image_alignments.json'
    with open(anno_file,'r') as f:
        anno = json.load(f)
    alignments = anno['alignments']

    FLAGS.obj_c = 9
    FLAGS.feat_c_R = 1289
    FLAGS.feat_c_ts = 1292

    device = 'cuda'
    model_dir = '/home/aston/Desktop/Datasets/ShapeNet/ShapeNetCore.v2'
    datadir = '/home/aston/Desktop/Datasets/pose_data'
    ckpt_dir = '/home/aston/Desktop/python/pose/engine/output/models'
    ckpt_epoch = 149
    with open(os.path.join(datadir,'id2point.pkl'),'rb') as f:
        id2points = pickle.load(f)

    segment_file = '/home/aston/Desktop/python/Test/2023-08-31-reconstruction/recons_res.pkl'
    with open(segment_file,'rb') as f:
        segment_res = pickle.load(f)

    error = []
    reconstruct_result = []
    total_scene = len(list(segment_res.keys()))
    correct_scene = 0
    for key in segment_res:
        scene = segment_res[key]
        mask = scene[0]
        im_H, im_W = mask.shape[0], mask.shape[1]

        filename = key.split('/')[-2]+"/"+key.split('/')[-1].split('_')[0]

        # gt_cad_catname_list = []
        # gt_cad_instance_list = []
        # gt_RT_list = []
        # gt_scale_list = []
        gt_list = []
        with open(os.path.join(datadir,"ScanNOCS",filename+"_label.pkl"), 'rb') as f:
            label = pickle.load(f)
        # gt_cad_instance_list = label['shapenet_instance_id']
        # gt_cad_cat_list = [shapeid.split('-')[0] for shapeid in label['shapenet_instance_id']]
        for index in range(len(label['class_ids'])):
            gt_dict = {}
            model_name = label['model_list'][index]
            rotaiton = label['rotations'][index]
            translation = label['translations'][index]
            scale = label['scales'][index]
            shapeid = label['shapenet_instance_id'][index]
            # gt_cad_catname_list.append(model_name)
            gt_dict['model_name'] = model_name
            gt_dict['shape_id'] = shapeid
            # gt_cad_instance_list.append(shapeid)
            gt_rt = np.eye(4)
            gt_rt[:3,:3] = rotaiton
            gt_rt[:3,3] = translation
            gt_dict['gt_rt'] = gt_rt
            gt_dict['gt_scale'] = scale
            # gt_RT_list.append(gt_rt)
            # gt_scale_list.append(scale)
            gt_list.append(gt_dict)

        # intrinsics
        intrinsics_dir = os.path.join(datadir, 'intrinsics')
        intrinsics_file = os.path.join(
            intrinsics_dir,
            filename.split('/')[0],
            'intrinsics_color.txt'
        )
        intrinsics = np.loadtxt(intrinsics_file)
        raster = Rasterizer(
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
            False,
            True
        )
        raster_model_id = 1

        for index in range(1,len(scene)):
            bbox = scene[index]['bbox']
            inst_id = scene[index]['id']
            model_name = scene[index]['name']
            cat_id = catname2id[model_name]-1
            shapenet_id = scene[index]['match_cad_name_list'][0].replace('.npy','')
            pc = id2points[shapenet_id]
            pc = _sample_points(pc,FLAGS.random_points)
            bbox = [bbox[2],bbox[1],bbox[0],bbox[3]]

            rmin, rmax, cmin, cmax = get_bbox(bbox)
            bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
            bbox_center, scale = aug_bbox_DZI(bbox_xyxy, im_H, im_W)
            bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
            bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
            mask_target = mask.copy().astype(float)
            mask_target[mask != inst_id] = 0.0
            mask_target[mask == inst_id] = 1.0
            roi_mask = crop_resize_by_warp_affine(
                mask_target, bbox_center, scale, 256, interpolation=cv2.INTER_NEAREST
            )
            roi_mask = np.expand_dims(roi_mask, axis=0)

            network = HSPose('PoseNet_only')
            network = network.to(device)
            ckpt_path = os.path.join(ckpt_dir, f'{model_name}_model_{ckpt_epoch}.pth')
            state_dict = torch.load(ckpt_path, map_location='cuda:0')['posenet_state_dict']
            unnecessary_nets = [
                'posenet.face_recon.conv1d_block',
                'posenet.face_recon.face_head',
                'posenet.face_recon.recon_head'
            ]
            for key in list(state_dict.keys()):
                origin_key = key
                key = key.replace('module.', '')
                state_dict[key] = state_dict.pop(origin_key)
                for net_to_delete in unnecessary_nets:
                    if key.startswith(net_to_delete):
                        state_dict.pop(key)
                # Adapt weight name to match old code version.
                # Not necessary for weights trained using newest code.
                # Dose not change any function.
                if 'resconv' in key:
                    state_dict[key.replace("resconv", "STE_layer")] = state_dict.pop(key)
            network.load_state_dict(state_dict, strict=True)
            network.eval()
            torch.cuda.empty_cache()
            # transform data to batch
            output_dict = network(
                obj_id=torch.tensor(cat_id).unsqueeze(0).to(device).float(),
                PC=torch.tensor(pc).unsqueeze(0).to(device).float(),
                sym=torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device).float(),
                roi_mask=torch.tensor(roi_mask).unsqueeze(0).to(device).float()
            )

            p_green_R_vec = output_dict['p_green_R'].detach()
            p_red_R_vec = output_dict['p_red_R'].detach()
            p_T = output_dict['Pred_T'].detach()
            p_s = output_dict['Pred_s'].detach()
            f_green_R = output_dict['f_green_R'].detach()
            f_red_R = output_dict['f_red_R'].detach()
            pred_s = p_s
            pred_RT = generate_RT(
                [p_green_R_vec, p_red_R_vec],
                [f_green_R, f_red_R],
                p_T,
                mode='vec',
                sym=torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device).float()
            )
            pred_s = pred_s.cpu().numpy().squeeze()
            pred_RT = pred_RT.cpu().numpy().squeeze()
            lx = max(pc[:, 0]) - min(pc[:, 0])
            ly = max(pc[:, 1]) - min(pc[:, 1])
            lz = max(pc[:, 2]) - min(pc[:, 2])
            pred_s = np.array([pred_s[0] / lx, pred_s[1] / ly, pred_s[2] / lz])
            s_matrix = np.eye(4)
            s_matrix[0, 0] = pred_s[0]
            s_matrix[1, 1] = pred_s[1]
            s_matrix[2, 2] = pred_s[2]
            pose = pred_RT.dot(s_matrix)

            model_path = os.path.join(
                model_dir,
                shapenet_id.split('-')[0],
                shapenet_id.split('-')[1],
                'models',
                'model_normalized.obj'
            )

            mesh = load_model(model_path)
            mesh = as_mesh(mesh)
            mesh.apply_transform(pose)

            raster.add_model(
                np.asarray(mesh.faces, dtype=raster.index_dtype),
                np.asarray(mesh.vertices, dtype=raster.scalar_dtype),
                raster_model_id,
                np.asarray(mesh.face_normals, raster.scalar_dtype)
            )
            raster_model_id += 1

            o3d_mesh = trimesh_to_open3d(mesh)
            # o3d.visualization.draw_geometries([o3d_mesh])
            o3d_mesh.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
            reconstruct_result.append(o3d_mesh)

            # difference
            for index in range(len(gt_list)):
                temp_dict = gt_list[index]
                if shapenet_id == temp_dict['shape_id']:
                    gt_rt = temp_dict['gt_rt']
                    scale = temp_dict['gt_scale']
                    r_diff,t_diff= compute_RT_errors(gt_rt,pred_RT)
                    s_diff = abs(np.mean((pred_s/scale)-1))
                    if r_diff <= 20 and t_diff <= 20 and s_diff <= 0.2:
                        print(f"shape_id:{shapenet_id}, r_diff:{r_diff}, t_diff:{t_diff}, s_diff:{s_diff}")
                        # print('correct!')
                        gt_list.pop(index)
                        break


        if len(gt_list) == 0:
            print(f"complete correct file : {filename} !!")
            correct_scene+=1
        # 光栅化渲染
        raster.rasterize()
        # 获取渲染后的实例instance map
        instances = np.uint8(raster.read_idx())
        # 获取实例个数，instance map中的每个值都表示一个实例
        colors = np.unique(instances)
        if len(colors) == 1:
            raster.clear_models()
            # print(f' ================= {key} ===================== ')
            print("visual None ! ")
            # continue
        mask[mask == 255] = 0
        # plt.figure()
        # plt.imshow(instances)
        # plt.title(filename)
        # plt.show()
        # plt.cla()
        # plt.close()

        combined_mesh = o3d.geometry.TriangleMesh()
        for mesh in reconstruct_result:
            combined_mesh += mesh
        combined_mesh.compute_vertex_normals()
        # visual
        # o3d.visualization.draw_geometries([combined_mesh])
        # break
    print(f"accurate: {correct_scene/total_scene * 100}%")
    print('hello world !')


if __name__ == '__main__':
    app.run(predict)