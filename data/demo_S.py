# Usage Examples:
#
#   python demo_S.py B1Counting 0
#   python demo_S.py B1Counting 0 100
#


import argparse
import glob
import os
import re


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_dir', nargs='?', default='S')
arg_parser.add_argument('scene')
arg_parser.add_argument('idxes', nargs='+', type=int)
args = arg_parser.parse_args()

data_dir = args.data_dir

img_dir = os.path.join(data_dir, 'images', args.scene)
assert os.path.exists(img_dir)

anno_file = os.path.join(data_dir, 'labels', '%s_BB.mat' % args.scene)
assert os.path.exists(anno_file)

idxes = args.idxes

pattern_img_files = re.compile(r'.*_([0-9]+)\.png$')


def sort_func_img_files(s):
    res = pattern_img_files.search(s)
    return int(res.group(1))


img_left_files = sorted(glob.glob(os.path.join(img_dir, 'BB_left_*.png')), key=sort_func_img_files)
img_right_files = sorted(glob.glob(os.path.join(img_dir, 'BB_right_*.png')), key=sort_func_img_files)


import numpy as np
import pickle
import cv2
import scipy.io as sio


fx = 822.79041
fy = 822.79041
tx = 318.47345
ty = 250.31296
baseline = 120.054

R_r = np.array([
    [1, 0, 0, -baseline],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
], dtype=np.float64)

K = np.array([
    [fx, 0, tx],
    [0, fy, ty],
    [0, 0, 1]
])

anno = sio.loadmat(anno_file)

cv2.namedWindow('key_points_left')
cv2.namedWindow('key_points_right')

i = 0
idx = idxes[i]
try:
    while True:
        print([os.path.basename(x) for x in (img_left_files[idx], img_right_files[idx])])

        img_left = cv2.imread(img_left_files[idx], cv2.IMREAD_UNCHANGED)
        img_right = cv2.imread(img_right_files[idx], cv2.IMREAD_UNCHANGED)

        xyz_left = anno['handPara'][:, :, idx]
        uv_left_homo = K.dot(xyz_left)
        uv_left = uv_left_homo[:2, :] / uv_left_homo[2:, :]

        xyz_left_homo = np.concatenate([xyz_left, np.ones([1, 21])])

        xyz_right = R_r.dot(xyz_left_homo)
        uv_right_homo = K.dot(xyz_right)
        uv_right = uv_right_homo[:2, :] / uv_right_homo[2:, :]

        uv_left = uv_left.T
        xyz_left = xyz_left.T
        uv_right = uv_right.T
        xyz_right = xyz_right.T

        for img, uv, xyz in zip((img_left, img_right), (uv_left, uv_right), (xyz_left, xyz_right)):
            u1v1_homo = np.dot(K, xyz.T).T
            u1v1 = u1v1_homo[:, :2] / u1v1_homo[:, 2:]

            for this_uv, this_u1v1 in zip(uv, u1v1):
                this_uv = this_uv.astype(np.int64)
                this_u1v1 = this_u1v1.astype(np.int64)

                cv2.rectangle(img,
                              (this_uv[0] - 1, this_uv[1] - 1),
                              (this_uv[0] + 1, this_uv[1] + 1),
                              (0, 255, 0), 1)
                cv2.rectangle(img,
                              (this_u1v1[0] - 1, this_u1v1[1] - 1),
                              (this_u1v1[0] + 1, this_u1v1[1] + 1),
                              (255, 255, 255), 1)

        cv2.imshow('key_points_left', img_left)
        cv2.imshow('key_points_right', img_right)

        while True:
            key = cv2.waitKey(0)
            if key in [ord(i) for i in 'pnjq']:
                break

        key = chr(key)
        if key == 'p':
            idx -= 1
        elif key == 'n':
            idx += 1
        elif key == 'j':
            i += 1
            idx = idxes[i]
        else:
            break
except IndexError:
    pass

cv2.destroyAllWindows()
