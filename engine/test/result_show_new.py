import pickle
import os
import numpy as np

import sys
sys.path.append('/home/aston/Desktop/python/pose')
from evaluation.eval_utils import compute_RT_errors
from prettytable import PrettyTable
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

cats = ['cabinet','bookcase','bed','bathtub','sofa','bin','table','chair','display']
result_dict = {}
for catname in cats:
    # catname = 'cabinet'
    result_filename = f'../{catname}_test_pose_output.pkl'
    with open(result_filename,'rb') as f:
        test_result = pickle.load(f)

    total = len(test_result)
    r20_t20_s20 = 0
    r20 = 0
    t20 = 0
    s20 = 0

    r10_t10 = 0
    r10_t5 = 0
    r10_t2 = 0
    r5_t5 = 0
    r5_t2 = 0
    r5,r10,t10,t5,t2 = 0,0,0,0,0


    for item in test_result:
        pred_rt = item['pred_rt']
        pred_s = item['pred_s']
        gt_rt = item['gt_rt'].cpu().numpy()
        gt_s = item['gt_s']

        s_error = abs(np.mean(pred_s/gt_s-1))

        res = compute_RT_errors(pred_rt,gt_rt)
        r_error,t_error = res
        res = np.append(res,s_error)
        np.set_printoptions(precision=3, suppress=True)
        # print(res)
        if r_error < 20 and t_error < 20 and s_error < 0.2:
            r20_t20_s20+=1
        if r_error < 10 and t_error < 10:
            r10_t10 +=1
        if r_error<10 and t_error<5:
            r10_t5+=1
        if r_error<10 and t_error<2:
            r10_t2+=1
        if r_error<5 and t_error<5:
            r5_t5+=1
        if r_error<5 and t_error<2:
            r5_t2+=1
        if r_error<10:
            r10+=1
        if r_error<5:
            r5+=1
        if t_error < 10:
            t10+=1
        if t_error<5:
            t5+=1
        if t_error<2:
            t2+=1
        if r_error<20:
            r20+=1
        if t_error<20:
            t20+=1
        if s_error<0.2:
            s20+=1

    # print(f"======={catname}========")
    # print("5 degree, 2cm: {:.3f}'.".format(r5_t2/total))
    # print("5 degree, 5cm: {:.3f}'.".format(r5_t5/total))
    # print("10 degree, 2cm: {:.3f}'".format(r10_t2/total))
    # print("10 degree, 5cm: {:.3f}'".format(r10_t5/total))
    # print("10 degree, 10cm: {:.3f}".format(r10_t10/total))
    # print("5 degree: {:.3f}".format(r5/total))
    # print("10 degree: {:.3f}".format(r10/total))
    # print("2cm: {:.3f}".format(t2/total))
    # print("5cm: {:.3f}".format(t5/total))
    # print("10cm: {:.3f}".format(t10/total))
    # print("20 degree, 20cm, 20%: {:.3f}".format(r20_t20_s20/total))
    # print("20 degree: {:.3f}".format(r20/total))
    # print("20cm: {:.3f}".format(t20/total))
    # print("20%: {:.3f}".format(s20/total))

    result_dict[catname] = [
        catname,
        r5_t2 / total,
        r5_t5 / total,
        r10_t2 / total,
        r10_t5 / total,
        r10_t10 / total,
        r5 / total,
        r10 / total,
        t2/total,
        t5/total,
        t10/total,
        r20/total,
        t20/total,
        s20/total,
        r20_t20_s20/total
    ]

# print(result_dict)

table = PrettyTable()
table.field_names = ['Category',"5\u00b02cm","5\u00b05cm","10\u00b02cm","10\u00b05cm","10\u00b010cm",
                     "5\u00b0","10\u00b0","2cm","5cm","10cm","20\u00b0","20cm","20%(scale)","20\u00b0 20cm 20%"]
for cat in result_dict:
    table.add_row(result_dict[cat])
table.float_format = ".4"
print(table)

# print('hello world !')