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

def predict(argv):
    # config
    FLAGS.obj_c = 9
    FLAGS.feat_c_R = 1289
    FLAGS.feat_c_ts = 1292
    model_dir = '/home/aston/Desktop/Datasets/ShapeNet/ShapeNetCore.v2'
    datadir = '/home/aston/Desktop/Datasets/pose_data'
    ckpt_dir = '/home/aston/Desktop/python/pose/engine/output/models'
    ckpt_epoch = 149
    with open(os.path.join(datadir,'id2point.pkl'),'rb') as f:
        id2points = pickle.load(f)

    # output
    output_dir = './output'
    # os.makedirs(output_dir,exist_ok=True)

    filename = FLAGS.filename
    assert filename != "", "Filename is None"
    output = os.path.join(output_dir,filename.split('/')[0])
    os.makedirs(output,exist_ok=True)
    output = os.path.join(output,filename.split('/')[1])
    # intrinsics
    intrinsics_dir = os.path.join(datadir,'intrinsics')
    intrinsics_file = os.path.join(
        intrinsics_dir,
        filename.split('/')[0],
        'intrinsics_color.txt'
    )
    intrinsics = np.loadtxt(intrinsics_file)


    # mask
    mask = cv2.imread(os.path.join(datadir,f'ScanNOCS/{filename}_mask.png'))
    mask = mask[:,:,2]
    # instances = np.unique(mask)

    with open(os.path.join(datadir,f'ScanNOCS/{filename}_label.pkl'),'rb') as f:
        label = pickle.load(f)

    img = cv2.imread(os.path.join(datadir,f'ScanNOCS/{filename}_color.jpg'))
    img = np.asarray(img)
    fig,ax = plt.subplots(2,2)
    ax[0,0].imshow(img)
    ax[0,0].set_title('Image')
    # ax[0,0].axis('off')
    ax[0,1].imshow(img)
    for index in range(len(label['class_ids'])):
        # y_max,x_min,y_min,x_max
        bbox = label['bboxes'][index]
        retangle = patches.Rectangle(
            (bbox[1],bbox[2]),
            bbox[3]-bbox[1],
            bbox[0]-bbox[2],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax[0,1].add_patch(retangle)
        ax[0,1].text((bbox[1]+bbox[3])/2,(bbox[0]+bbox[2])/2,label['model_list'][index],verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))

    ax[0,1].set_title('BBox')
    # ax[0,1].axis('off')
    # plt.show()

    device = 'cuda'

    im_H,im_W = mask.shape[0], mask.shape[1]
    # bbox
    instance_num = len(label['class_ids'])

    # raster
    raster = Rasterizer(intrinsics[0][0],intrinsics[1][1],intrinsics[0][2],intrinsics[1][2],False,True)
    raster_model_id = 1

    error = []
    reconstruct_result = []
    for index in range(instance_num):
        inst_id = label['instance_ids'][index]
        class_id = label['class_ids'][index]
        bbox = label['bboxes'][index]
        shapenet_id = label['shapenet_instance_id'][index]
        model_name = label['model_list'][index]
        cat_id = label['class_ids'][index] - 1
        pc = id2points[shapenet_id]
        pc = _sample_points(pc,FLAGS.random_points)
        rotaiton = label['rotations'][index]
        translation = label['translations'][index]
        scale = label['scales'][index]

        rmin, rmax, cmin, cmax = get_bbox(bbox)
        # here resize and crop to a fixed size 256 x 256
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_DZI(bbox_xyxy, im_H, im_W)
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
        mask_target = mask.copy().astype(float)
        mask_target[mask != inst_id] = 0.0
        mask_target[mask == inst_id] = 1.0
        # plt.figure()
        # plt.imshow(mask_target)
        # plt.title('mask all')
        # plt.show()

        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, 256, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)

        # plt.figure()
        # plt.title('mask crop')
        # plt.imshow(roi_mask)
        # plt.show()

        # load checkpoint according to class name
        network = HSPose('PoseNet_only')
        network = network.to(device)
        ckpt_path = os.path.join(ckpt_dir,f'{model_name}_model_{ckpt_epoch}.pth')
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
            obj_id = torch.tensor(cat_id).unsqueeze(0).to(device).float(),
            PC = torch.tensor(pc).unsqueeze(0).to(device).float(),
            sym = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device).float(),
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
        lx = max(pc[:,0]) - min(pc[:,0])
        ly = max(pc[:,1]) - min(pc[:,1])
        lz = max(pc[:,2]) - min(pc[:,2])
        pred_s = np.array([pred_s[0]/lx,pred_s[1]/ly,pred_s[2]/lz])
        s_matrix = np.eye(4)
        s_matrix[0,0] = pred_s[0]
        s_matrix[1,1] = pred_s[1]
        s_matrix[2,2] = pred_s[2]
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
        raster_model_id+=1

        o3d_mesh = trimesh_to_open3d(mesh)
        # o3d.visualization.draw_geometries([o3d_mesh])
        o3d_mesh.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        reconstruct_result.append(o3d_mesh)

        # difference
        gt_rt = np.eye(4)
        gt_rt[:3,:3] = rotaiton
        gt_rt[:3,3] = translation

        res = compute_RT_errors(gt_rt,pred_RT)

        s_error = abs(np.mean((pred_s/scale)-1))
        res = np.append(res,s_error)
        error.append(res)
        # break
    np.savetxt(output+"_error.txt", np.asarray(error), delimiter=' ', fmt="%.4f")
    # print(np.asarray(error))

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
    mask[mask==255] = 0

    ax[1,0].imshow(instances)
    ax[1,0].set_title('pred_mask')
    ax[1,0].axis('off')
    # plt.subplot(1,2,2)
    ax[1,1].imshow(mask)
    ax[1,1].set_title('GT_mask')
    ax[1,1].axis('off')

    plt.tight_layout()
    plt.savefig(output+".png", bbox_inches='tight',dpi=600)
    if FLAGS.visual==1:
        # visual
        plt.show(dpi=600)

    # clear
    plt.clf()
    plt.close()
    raster.clear_models()

    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in reconstruct_result:
        combined_mesh+=mesh
    combined_mesh.compute_vertex_normals()
    if FLAGS.visual == 1:
        # visual
        o3d.visualization.draw_geometries([combined_mesh])
    o3d.io.write_triangle_mesh(output+".ply", combined_mesh)

    # print('hello world !')

if __name__ == '__main__':
    app.run(predict)