# Usage Examples:
#
#   python demo_R.py R/evaluation 0
#   python demo_R.py R/evaluation 0 100
#


import argparse
import glob
import os


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('data_dir')
arg_parser.add_argument('idxes', nargs='+', type=int)
args = arg_parser.parse_args()

img_color_files = sorted(glob.glob(os.path.join(args.data_dir, 'color', '*.png')))
img_mask_files = sorted(glob.glob(os.path.join(args.data_dir, 'mask', '*.png')))

anno_file = glob.glob(os.path.join(args.data_dir, 'anno_*.pickle'))[0]

idxes = args.idxes


import numpy as np
import pickle
import cv2


def get_bounding_boxes(img_mask):
    img_mask_left = np.logical_and(2 <= img_mask, img_mask <= 17)
    img_mask_right = np.logical_and(18 <= img_mask, img_mask <= 33)

    res = []
    for mask in [img_mask_left, img_mask_right]:
        if np.any(mask):
            mask = np.logical_not(mask)

            mg_x, mg_y = np.meshgrid(range(mask.shape[0]), range(mask.shape[1]))
            mg_x_min = mg_x.astype(np.int64); mg_x_max = np.copy(mg_x_min)
            mg_y_min = mg_y.astype(np.int64); mg_y_max = np.copy(mg_y_min)

            mg_x_min[mask] = 1e16; x_min = np.min(mg_x_min)
            mg_y_min[mask] = 1e16; y_min= np.min(mg_y_min)
            mg_x_max[mask] = -1; x_max = np.max(mg_x_max)
            mg_y_max[mask] = -1; y_max = np.max(mg_y_max)

            res.append((x_min, y_min, x_max, y_max))
        else:
            res.append(None)
    return res

with open(anno_file, 'rb') as f:
    anno = pickle.load(f)

cv2.namedWindow('boxes')
cv2.namedWindow('key_points')

i = 0
idx = idxes[i]
try:
    while True:
        print(os.path.basename(img_color_files[idx]))

        img_color = cv2.imread(img_color_files[idx], cv2.IMREAD_UNCHANGED)
        img_mask = cv2.imread(img_mask_files[idx], cv2.IMREAD_UNCHANGED)


        img_boxes = np.copy(img_color)
        for box in get_bounding_boxes(img_mask):
            if box is None:
                continue
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.imshow('boxes', img_boxes)

        uv = anno[idx]['uv_vis']
        uv, vis = uv[:, :2], uv[:, 2]

        xyz = anno[idx]['xyz']
        K = anno[idx]['K']

        u1v1_homo = np.dot(K, xyz.T).T
        u1v1 = u1v1_homo[:, :2] / u1v1_homo[:, 2:]

        img_uv = np.copy(img_color)
        for this_uv, this_u1v1, this_vis in zip(uv, u1v1, vis):
            if this_vis == 0:
                continue

            this_uv = this_uv.astype(np.int64)
            this_u1v1 = this_u1v1.astype(np.int64)

            cv2.rectangle(img_uv, (this_uv[0] - 1, this_uv[1] - 1),
                        (this_uv[0] + 1, this_uv[1] + 1), (0, 255, 0), 1)
            cv2.rectangle(img_uv, (this_u1v1[0] - 1, this_u1v1[1] - 1),
                        (this_u1v1[0] + 1, this_u1v1[1] + 1), (255, 255, 255), 1)

        cv2.imshow('key_points', img_uv)

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
