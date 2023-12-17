import os
import torch
import random
from network.HSPose import HSPose
from tools.geom_utils import generate_RT
from config.config import *
from absl import app

FLAGS = flags.FLAGS
from evaluation.load_data_eval import PoseDataset
import numpy as np
import time

# from creating log
import tensorflow as tf
from evaluation.eval_utils import setup_logger
from evaluation.eval_utils_v1 import compute_degree_cm_mAP
from tqdm import tqdm
import cv2

def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

device = 'cuda'


def get_fs_net_scale(self, c, model, nocs_scale):
    # model pc x 3
    lx = max(model[:, 0]) - min(model[:, 0])
    ly = max(model[:, 1]) - min(model[:, 1])
    lz = max(model[:, 2]) - min(model[:, 2])

    # real scale
    lx_t = lx * nocs_scale * 1000
    ly_t = ly * nocs_scale * 1000
    lz_t = lz * nocs_scale * 1000

    unitx, unity, unitz = 0,0,0

    return np.array([lx_t - unitx, ly_t - unity, lz_t - unitz]), np.array([unitx, unity, unitz])

def _sample_points(pcl, n_pts=1024):
    # n_pts = 1024
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

def eval(argv):
    FLAGS.obj_c = 9
    FLAGS.feat_c_R = 1289
    FLAGS.feat_c_ts = 1292
    # seed
    if FLAGS.eval_seed == -1:
        seed = int(time.time())
    else:
        seed = FLAGS.eval_seed

    # 初始化随机种子
    seed_init_fn(seed)
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    # tensorflow禁用某些内容
    tf.compat.v1.disable_eager_execution()
    # 设置logger
    logger = setup_logger('eval_log', os.path.join(FLAGS.model_save, 'log_eval.txt'))
    Train_stage = 'PoseNet_only'
    FLAGS.train = False

    # 模型名称
    model_name = os.path.basename(FLAGS.resume_model).split('.')[0]
    # 输出路径
    output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    import pickle
    t_inference = 0.0
    img_count = 0
    pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    if os.path.exists(pred_result_save_path):
        with open(pred_result_save_path, 'rb') as file:
            pred_results = pickle.load(file)
        img_count = 1
    else:
        network = HSPose(Train_stage)
        network = network.to(device)

        if FLAGS.resume:
            state_dict = torch.load(FLAGS.resume_model)['posenet_state_dict']
            unnecessary_nets = ['posenet.face_recon.conv1d_block', 'posenet.face_recon.face_head', 'posenet.face_recon.recon_head']
            for key in list(state_dict.keys()):
                for net_to_delete in unnecessary_nets:
                    if key.startswith(net_to_delete):
                        state_dict.pop(key)

                if 'resconv' in key:
                    state_dict[key.replace('resconv','STE_layer')] = state_dict.pop(key)
            network.load_state_dict(state_dict=state_dict,strict=True)

        else:
            raise NotImplementedError

        network = network.eval()

        pred_results = []

        # load data

        data_root = '/home/aston/Desktop/Datasets/pose_data'
        val_list_file = os.path.join(data_root,'val.txt')
        with open(val_list_file,'r') as f:
            val_list = [name.strip() for name in f.readlines()]
        scene_name = random.choice(val_list)
        prefix = os.path.join(data_root, scene_name)
        with open(prefix+'_label.pkl','rb') as f:
            label = pickle.load(f)
        mask_path = prefix+'_mask.png'
        img_path = prefix+'_color.jpg'
        img = cv2.imread(img_path)[:,:,:3]
        mask = cv2.imread(mask_path)[:,:,2]


        class_ids = np.asarray(label['class_ids'])
        class_ids_0base = np.asarray([i-1 for i in label['class_ids']])
        bboxes = np.asarray(label['bboxes'])
        rotations = np.asarray(label['rotations'])
        translations = np.asarray(label['translations'])
        scales = np.asarray(label['scales'])
        shapenet_instance_ids = label['shapenet_instance_id']
        with open('/home/aston/Desktop/Datasets/pose_data/id2point.pkl','rb') as f:
            models = pickle.load(f)



        mean_shapes = []
        sym_info = []
        pcl_in = []
        for i in range(len(class_ids)):
            sym = np.array([0,0,0,0],dtype=int)
            sym_info.append(sym)
            mean_shape = np.array([0,0,0])
            mean_shapes.append(mean_shape)
            pcl_in.append(models[shapenet_instance_ids[i]])
        sym_info = np.asarray(sym_info)
        mean_shapes = np.asarray(mean_shapes)
        pcl_in = np.asarray(pcl_in)

        data = {}
        data['cat_id'] = torch.as_tensor(class_ids)
        data['cat_id_0base'] = torch.as_tensor(class_ids_0base)
        data['sym_info'] = torch.as_tensor(sym_info.astype(float)).contiguous()
        data['meah_shape'] = torch.as_tensor(mean_shapes,dtype=torch.float32).contiguous()
        data['pcl_in'] = torch.as_tensor(
            pcl_in.astype(float)
        ).contiguous()

        # data = data, detection_dict = None, gts = label

        mean_shape = data['mean_shape'].to(device)
        sym = data['sym_info'].to(device)

        output_dict = network(
            PC=data['pcl_in'].to(device),
            obj_id = data['cat_id_0base'].to(device),
            mean_shape = mean_shape,
            sym = sym
        )

        p_green_R_vec = output_dict['p_green_R'].detach()
        p_red_R_vec = output_dict['p_red_R'].detach()
        p_T = output_dict['Pred_T'].detach()
        p_s = output_dict['Pred_s'].detach()
        f_green_R = output_dict['f_green_R'].detach()
        f_red_R = output_dict['f_red_R'].detach()
        pred_s = p_s + mean_shape
        pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)




if __name__ == '__main__':
    app.run(eval)