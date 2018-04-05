import os

import cv2
import PIL
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from key_points_estimation.gen_tfrecord import decompose_xyz
from key_points_estimation.cook_tfrecord import _get_crop_box, _gen_score_maps
from key_points_estimation.config import CROP_SIZE

from utils.DatasetReader import RReader, SReader

from test.run import get_save_func, draw_hand, plot_hand_3d, get_fig_img, cvt_float_img
from test.config import config


DIR = '../__backup/test/rs'


def cook(img, uv, vis, *, g=None):
    def _build_graph():
        g = tf.Graph()
        with g.as_default():
            img = tf.placeholder(tf.float32, shape=(None, None, 3), name='img')
            img_shape = tf.shape(img)

            uv = tf.placeholder(tf.float32, shape=(21, 2), name='uv')

            vis = tf.placeholder(tf.float32, shape=(21,), name='vis')
            vis = tf.greater(vis, 0.5)

            box, box_size, box_lt, box_rel = _get_crop_box(uv, vis, img_shape, False, False, False)

            img_cr = tf.image.crop_and_resize(tf.expand_dims(tf.cast(img, tf.float32), 0),
                                              tf.expand_dims(box_rel, 0), [0],
                                              [CROP_SIZE, CROP_SIZE])
            img_cr = tf.squeeze(img_cr)
            img_cr.set_shape([CROP_SIZE, CROP_SIZE, 3])
            img_cr = tf.cast(img_cr, tf.uint8, name='img_cr')

            uv_cr = (uv - box_lt) * (tf.cast(CROP_SIZE, tf.float32) / box_size)
            uv_cr = tf.identity(uv_cr, name='uv_cr')

            _ = tf.logical_and(tf.greater_equal(uv_cr[:, 0], 0),
                               tf.logical_and(tf.less(uv_cr[:, 0], CROP_SIZE),
                                              tf.logical_and(tf.greater_equal(uv_cr[:, 1], 0),
                                                             tf.less(uv_cr[:, 1], CROP_SIZE))))
            vis = tf.logical_and(vis, _)

            score_maps = _gen_score_maps(uv_cr, vis, [CROP_SIZE, CROP_SIZE])
            score_maps = tf.identity(score_maps, name='score_maps')

            return g

    if g is None:
        g = _build_graph()

    with g.as_default():
        with tf.Session(graph=g) as sess:
            img_t = g.get_tensor_by_name('img:0')
            uv_t = g.get_tensor_by_name('uv:0')
            vis_t = g.get_tensor_by_name('vis:0')
            img_cr_t = g.get_tensor_by_name('img_cr:0')
            uv_cr_t = g.get_tensor_by_name('uv_cr:0')
            score_maps_t = g.get_tensor_by_name('score_maps:0')

            img_cr, uv_cr, score_maps = sess.run(
                [img_cr_t, uv_cr_t, score_maps_t],
                feed_dict={
                    img_t: img,
                    uv_t: uv,
                    vis_t: vis
                })
            return img_cr, uv_cr, score_maps


def main(*, fig=None, axes=None):
    if fig is None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')

    cv2.namedWindow('boxes')
    cv2.namedWindow('uv')
    cv2.namedWindow('uv_cr')
    cv2.namedWindow('score_map')
    cv2.namedWindow('xyz_rel')

    for reader in (RReader(data_dir='../data/R', verbose=True), SReader(data_dir='../data/S', verbose=True)):
        if type(reader) == SReader:
            cv2.destroyWindow('boxes')

        for example in reader:
            img_right = None
            img_boxes = None

            if type(example) == list:
                example, example_right = example
                img_right = cv2.imread(example_right['img_file'])

            img_file, boxes, uv, xyz, vis, is_right_hand, K = example[
                'img_file', 'boxes', 'uv', 'xyz', 'vis', 'is_right_hand', 'K'
            ]
            img = cv2.imread(img_file)
            img_H, img_W, _ = img.shape

            img_file = os.path.join(DIR, os.path.basename(img_file))
            _save = get_save_func(img_file)

            if boxes is not None:
                img_boxes = img.copy()
                for box in boxes:
                    if box is None:
                        continue
                    x_min, x_max, y_min, y_max = [
                        int(box[x]) for x in ('x_min', 'x_max', 'y_min', 'y_max')
                    ]
                    cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                cv2.imshow('boxes', img_boxes)

            img_uv = img.copy()
            draw_hand(img_uv, uv)
            cv2.imshow('uv', img_uv)

            trans, scale, xyz_rel, rot, rot_inv, xyz_cano = decompose_xyz(xyz)

            axes.cla()
            plot_hand_3d(axes, xyz_rel)
            img_xyz_rel = get_fig_img(fig)[:, :, ::-1].copy()
            cv2.imshow('xyz_rel', img_xyz_rel)

            img_cr, uv_cr, score_maps = cook(img, uv, vis)
            img_uv_cr = img_cr.copy()
            draw_hand(img_uv_cr, uv_cr)
            cv2.imshow('uv_cr', img_uv_cr)

            img_score_map = cvt_float_img(score_maps[:, :, 0])
            cv2.imshow('score_map', img_score_map)

            plt_showed = False
            saved = False
            while True:
                key = cv2.waitKey(0)
                if chr(key) == 'p':
                    if not plt_showed:
                        plt.show()
                        fig = plt.figure()
                        axes = fig.add_subplot(111, projection='3d')
                        plt_showed = True

                if chr(key) == 's':
                    if not saved:
                        if img_right is not None:
                            _save(img_right)
                        if img_boxes is not None:
                            _save(img_boxes)
                        _save(img_uv)
                        _save(img_xyz_rel)
                        _save(img_uv_cr)
                        _save(img_score_map)
                        print('\n(%s) saved' % img_file)
                        saved = True

                if key in [ord(i) for i in 'nq']:
                    break

            if chr(key) == 'q':
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
