# Usage Examples:
#
#   python demo_E.py E/_LABELLED_SAMPLES/CARDS_COURTYARD_B_T 0
#   python demo_E.py E/_LABELLED_SAMPLES/CARDS_COURTYARD_B_T 0 60
#


import argparse
import glob
import os


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('img_dir')
arg_parser.add_argument('idxes', nargs='+', type=int)
args = arg_parser.parse_args()

img_dir = args.img_dir
idxes = args.idxes

img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
anno_file = os.path.join(img_dir, 'polygons.mat')


import numpy as np
import scipy.io as sio
import cv2


anno = sio.loadmat(anno_file)
polygons = anno['polygons'][0]

cv2.namedWindow('boxes')

i = 0
idx = idxes[i]
try:
    while True:
        print(os.path.basename(img_files[idx]))

        img = cv2.imread(img_files[idx])
        masks = polygons[idx]

        for mask in masks:
            if mask.size > 0:
                x_min = np.min(mask[:, 0]) - 1
                x_max = np.max(mask[:, 0]) - 1
                y_min = np.min(mask[:, 1]) - 1
                y_max = np.max(mask[:, 1]) - 1

                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)

        cv2.imshow('boxes', img)

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
