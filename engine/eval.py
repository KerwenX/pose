import os
import pickle
import random

import math

import numpy
import torch
from absl import app
import sys
sys.path.insert(0,'/home/aston/Desktop/python/pose')

from config.config import *
from tools.training_utils import build_lr_rate, build_optimizer
from tools.geom_utils import generate_RT
from evaluation.eval_utils import setup_logger
from evaluation.eval_utils_v1 import compute_degree_cm_mAP
from network.HSPose import HSPose

FLAGS = flags.FLAGS
from datasets.load_data import PoseDataset
from datasets.load_data_roca import ScanNetDataset
from datasets.load_data_new import MaskDataset

from tqdm import tqdm
import time
import numpy as np

# from creating log
import tensorflow as tf
from tools.eval_utils import setup_logger
from tensorflow.compat.v1 import Summary
from scipy.spatial.transform import Rotation as Rotate
torch.autograd.set_detect_anomaly(True)

device = 'cuda'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train(argv):
    # mode = train / test
    mode = 'test'

    FLAGS.resume = True
    FLAGS.per_obj = 'chair'
    print("===============================",FLAGS.per_obj,"==============================")
    FLAGS.resume_model = f'/home/aston/Desktop/python/pose/engine/output/models/{FLAGS.per_obj}_model_149.pth'
    # FLAGS.resume_model = f'/home/aston/Desktop/python/pose/engine/output/models/chair_model_59.pth'
    FLAGS.model_save = 'eval'
    FLAGS.train = False

    # build dataset annd dataloader
    # train_config_list = ['GPVPose','mytrain']
    FLAGS.batch_size = 3
    FLAGS.num_workers = 8
    index = 1

    if index == 0:
        # GPV pose train
        train_dataset = PoseDataset(source=FLAGS.dataset, mode=mode,
                                    data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)

    elif index == 1:
        # mytrain
        FLAGS.dataset_dir = '/home/aston/Desktop/Datasets/pose_data'
        FLAGS.obj_c = 9
        FLAGS.feat_c_R = 1289
        FLAGS.feat_c_ts = 1292
        train_dataset = MaskDataset(source=FLAGS.dataset, mode=mode,
                                       data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)

    Train_stage = 'PoseNet_only'
    network = HSPose(Train_stage)
    network = network.to(device)
    train_steps = FLAGS.train_steps
    #  build optimizer
    # param_list = network.build_params(training_stage_freeze=[])
    # optimizer = build_optimizer(param_list)
    # optimizer.zero_grad()   # first clear the grad
    # scheduler = build_lr_rate(optimizer, total_iters=train_steps * FLAGS.total_epoch // FLAGS.accumulate)
    # resume or not
    s_epoch = 0
    if FLAGS.resume:
        state_dict = torch.load(FLAGS.resume_model,map_location='cuda:0')['posenet_state_dict']
        unnecessary_nets = [
            'posenet.face_recon.conv1d_block',
            'posenet.face_recon.face_head',
            'posenet.face_recon.recon_head'
        ]
        for key in list(state_dict.keys()):
            origin_key = key
            key = key.replace('module.','')
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
    else:
        raise NotImplementedError


    # build dataset annd dataloader
    test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                   num_workers=FLAGS.num_workers, pin_memory=True,
                                                   prefetch_factor = 4,
                                                   worker_init_fn =seed_worker,
                                                   shuffle=False )
    network.eval()
    global_step = train_steps * s_epoch  # record the number iteration
    # count = 0
    result = []
    torch.cuda.empty_cache()
    output = []
    for epoch in range(s_epoch, FLAGS.total_epoch):
        i = 0

        for data in tqdm(test_dataloader, desc=f'Testing {epoch}/{FLAGS.total_epoch}', dynamic_ncols=True):
            output_dict \
                = network(
                      obj_id=data['cat_id'].to(device).float(),
                      PC=data['pcl_in'].to(device).float(),
                      # gt_R=data['rotation'].to(device).float(),
                      # gt_t=data['translation'].to(device).float(),
                      # gt_s=data['fsnet_scale'].to(device).float(),
                      # mean_shape=data['mean_shape'].to(device).float(),
                      sym=data['sym_info'].to(device).float(),
                      # aug_bb=data['aug_bb'].to(device).float(),
                      # aug_rt_t=data['aug_rt_t'].to(device).float(),
                      # aug_rt_r=data['aug_rt_R'].to(device).float(),
                      # model_point=data['model_point'].to(device).float(),
                      # nocs_scale=data['nocs_scale'].to(device).float(),
                      # do_loss=True,
                # roi_img=data['roi_img'].to(device).float(),
                roi_mask=data['roi_mask'].to(device).float()
            )

            sym = data['sym_info'].to(device).float()
            p_green_R_vec = output_dict['p_green_R'].detach()
            p_red_R_vec = output_dict['p_red_R'].detach()
            p_T = output_dict['Pred_T'].detach()
            p_s = output_dict['Pred_s'].detach()
            f_green_R = output_dict['f_green_R'].detach()
            f_red_R = output_dict['f_red_R'].detach()
            pred_s = p_s
            pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)

            gt_s = data['nocs_scale'].numpy()

            new_s = []
            s_differ = []
            for index in range(pred_RT.shape[0]):
                pc_temp = data['pcl_in'][index]
                s_temp = pred_s[index].cpu().numpy()
                lx = max(pc_temp[:,0]) - min(pc_temp[:,0])
                ly = max(pc_temp[:,1]) - min(pc_temp[:,1])
                lz = max(pc_temp[:,2]) - min(pc_temp[:,2])
                s_temp = np.asarray([s_temp[0]/lx,s_temp[1]/ly,s_temp[2]/lz])

                s_differ.append(s_temp/gt_s[index])

                new_s.append(s_temp)


            s_differ = np.asarray(s_differ)
            new_s = np.asarray(new_s)

            gt_r = data['rotation']
            gt_t = data['translation']

            Transformations = []
            for index in range(gt_r.shape[0]):
                R = gt_r[index].view(3,3).cpu().numpy()
                T = gt_t[index].view(3,1).cpu().numpy()
                tmp = np.array([0,0,0,1]).reshape(1,4)
                a = np.concatenate((R,T),axis=1)
                a = np.concatenate((a,tmp),axis=0)
                Transformations.append(a)
            Transformations = np.asarray(Transformations)
            Transformations = torch.as_tensor(Transformations).to(device).float()

            pred_RT = pred_RT.cpu().numpy()
            gt_r = gt_r.cpu().numpy()

            for index in range(pred_RT.shape[0]):
                temp_dict = {
                    'pred_rt':pred_RT[index],
                    'pred_s':new_s[index],
                    'gt_rt':Transformations[index].cpu(),
                    'gt_s':gt_s[index]
                }
                output.append(temp_dict)


            for index in range(pred_RT.shape[0]):
                rotation_vector_R = Rotate.from_matrix(pred_RT[index][:3,:3]).as_rotvec()
                rotation_vector_R_gt = Rotate.from_matrix(gt_r[index]).as_rotvec()
                angle_difference_x = torch.norm(torch.tensor(rotation_vector_R[0] - rotation_vector_R_gt[0]))
                angle_difference_y = torch.norm(torch.tensor(rotation_vector_R[1] - rotation_vector_R_gt[1]))
                angle_difference_z = torch.norm(torch.tensor(rotation_vector_R[2] - rotation_vector_R_gt[2]))

                # 将弧度转为角度
                angle_difference_x_degrees = torch.rad2deg(angle_difference_x)
                angle_difference_y_degrees = torch.rad2deg(angle_difference_y)
                angle_difference_z_degrees = torch.rad2deg(angle_difference_z)

                # print(f"绕 x 轴的角度差异：{angle_difference_x_degrees.item()} 度")
                # print(f"绕 y 轴的角度差异：{angle_difference_y_degrees.item()} 度")
                # print(f"绕 z 轴的角度差异：{angle_difference_z_degrees.item()} 度")
                average_R = (angle_difference_x_degrees.item()+angle_difference_y_degrees.item()+angle_difference_z_degrees.item())/3

                Translation_pred = p_T[index].cpu().numpy()
                Translation_gt = gt_t[index].cpu().numpy()
                distance = np.linalg.norm(Translation_gt-Translation_pred)
                # print(f"平均旋转误差： {average_R}度, 两个平移的距离：{distance}cm")



                result.append([average_R,distance,s_differ])

            torch.cuda.empty_cache()

        #     # print('net_process', time.time()-begin)
        #     fsnet_loss = loss_dict['fsnet_loss']
        #     recon_loss = loss_dict['recon_loss']
        #     geo_loss = loss_dict['geo_loss']
        #     prop_loss = loss_dict['prop_loss']
        #
        #     total_loss = sum(fsnet_loss.values()) + sum(recon_loss.values()) \
        #                     + sum(geo_loss.values()) + sum(prop_loss.values()) \
        #
        #     if math.isnan(total_loss):
        #         print('Found nan in total loss')
        #         i += 1
        #         global_step += 1
        #         continue
        #     # backward
        #     if global_step % FLAGS.accumulate == 0:
        #         total_loss.backward()
        #         torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        #         optimizer.step()
        #         scheduler.step()
        #         optimizer.zero_grad()
        #     else:
        #         total_loss.backward()
        #         torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        #     global_step += 1
        #     if i % FLAGS.log_every == 0:
        #         write_to_summary(tb_writter, optimizer, total_loss, fsnet_loss, prop_loss, recon_loss, global_step)
        #     i += 1
        #
        # # save model
        # if (epoch + 1) % FLAGS.save_every == 0 or (epoch + 1) == FLAGS.total_epoch:
        #     torch.save(
        #         {
        #         'seed': seed,
        #         'epoch': epoch,
        #         'posenet_state_dict': network.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         },
        #         '{0}/model_{1:02d}.pth'.format(FLAGS.model_save, epoch))
        torch.cuda.empty_cache()

        with open(f'{FLAGS.per_obj}_{mode}_result.pkl', 'wb') as f:
            pickle.dump(result, f)

        with open(f'{FLAGS.per_obj}_{mode}_pose_output.pkl', 'wb') as f:
            pickle.dump(output, f)

        break



def write_to_summary(writter, optimizer, total_loss, fsnet_loss, prop_loss, recon_loss, global_step):
    summary = Summary(
        value=[
            Summary.Value(tag='lr', simple_value=optimizer.param_groups[0]["lr"]),
            Summary.Value(tag='train_loss', simple_value=total_loss),
            Summary.Value(tag='rot_loss_1', simple_value=fsnet_loss['Rot1']),
            Summary.Value(tag='rot_loss_2', simple_value=fsnet_loss['Rot2']),
            Summary.Value(tag='T_loss', simple_value=fsnet_loss['Tran']),
            Summary.Value(tag='Prop_sym_recon', simple_value=prop_loss['Prop_sym_recon']),
            Summary.Value(tag='Prop_sym_rt', simple_value=prop_loss['Prop_sym_rt']),
            Summary.Value(tag='Size_loss', simple_value=fsnet_loss['Size']),
            Summary.Value(tag='Face_loss', simple_value=recon_loss['recon_per_p']),
            Summary.Value(tag='Recon_loss_r', simple_value=recon_loss['recon_point_r']),
            Summary.Value(tag='Recon_loss_t', simple_value=recon_loss['recon_point_t']),
            Summary.Value(tag='Recon_loss_s', simple_value=recon_loss['recon_point_s']),
            Summary.Value(tag='Recon_p_f', simple_value=recon_loss['recon_p_f']),
            Summary.Value(tag='Recon_loss_se', simple_value=recon_loss['recon_point_self']),
            Summary.Value(tag='Face_loss_vote', simple_value=recon_loss['recon_point_vote']), ])
    writter.add_summary(summary, global_step)
    return

def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    app.run(train)
