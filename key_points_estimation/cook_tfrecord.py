import sys
import copy

import tensorflow as tf

from key_points_estimation.config import *
from key_points_estimation.gen_tfrecord_test import features_dict

from utils.TFSession import TFSession


def _get_crop_box(uv, vis, img_shape, crop_center_noise, crop_scale_noise, crop_offset_noise):
    with tf.variable_scope('get_crop_box'):
        height, width = [tf.cast(_, tf.float32) for _ in (img_shape[0], img_shape[1])]

        # crop with middle mcp as the center
        center = uv[12, :]
        if crop_center_noise:
            center += tf.truncated_normal([2], mean=0.0, stddev=20.0)

        uv_visible = tf.boolean_mask(uv, vis)

        # calculate bounding box of visible key points
        half_size = tf.reduce_max(tf.abs(uv_visible - center))
        if crop_scale_noise:
            half_size *= tf.random_uniform([], minval=1.0, maxval=1.2)

        if crop_offset_noise:
            center += tf.truncated_normal([2], mean=0.0, stddev=10.0)

        x_min = center[0] - half_size
        x_max = center[0] + half_size
        y_min = center[1] - half_size
        y_max = center[1] + half_size

        box = tf.stack([y_min, x_min, y_max, x_max], axis=0)
        box_size = 2 * half_size
        box_lt = tf.stack([x_min, y_min], axis=0)
        box_rel = tf.stack([y_min / height, x_min / width,
                            y_max / height, x_max / width])

        return box, box_size, box_lt, box_rel


def _gen_score_maps(uv, vis, img_shape):
    with tf.variable_scope('gen_score_maps'):
        height, width = img_shape[0], img_shape[1]

        X, Y = tf.meshgrid(tf.range(width), tf.range(height))
        X, Y = [tf.cast(_, tf.float32) for _ in (X, Y)]
        X = tf.tile(tf.expand_dims(X, 0), [NUM_KEY_POINTS, 1, 1])
        Y = tf.tile(tf.expand_dims(Y, 0), [NUM_KEY_POINTS, 1, 1])

        u, v = uv[:, 0], uv[:, 1]
        u, v = [tf.expand_dims(tf.expand_dims(_, 1), 2) for _ in (u, v)]

        zero_maps = tf.zeros([NUM_KEY_POINTS, height, width], dtype=tf.float32)
        gaussian_maps = tf.exp(-(tf.square(X - u) + tf.square(Y - v)) / (25.0 * 25.0))
        score_maps = tf.where(vis, gaussian_maps, zero_maps)
        score_maps = tf.transpose(score_maps, [1, 2, 0])

        return score_maps


def cook_tfrecord(tfrecord_paths_list, *, batch_size=BATCH_SIZE, num_epochs=None, uv_noise=False,
                  crop_center_noise=False, crop_scale_noise=False, crop_offset_noise=False):
    with tf.variable_scope('cook_tfrecord'):
        filename_queue = tf.train.string_input_producer(tfrecord_paths_list, num_epochs=num_epochs)

        tfrecord_reader = tf.TFRecordReader()
        _, serialized_record = tfrecord_reader.read(filename_queue)

        record = tf.parse_single_example(serialized_record, features=features_dict)

        img = tf.image.decode_jpeg(record['img'])
        img_shape = tf.shape(img)

        uv = record['uv']
        if uv_noise:
            uv += tf.truncated_normal(uv.shape, mean=0.0, stddev=2.5)

        vis = record['vis']
        vis = tf.greater(vis, 0.5)

        box, box_size, box_lt, box_rel = _get_crop_box(uv, vis, img_shape,
                                                       crop_center_noise,
                                                       crop_scale_noise,
                                                       crop_offset_noise)

        img_cr = tf.image.crop_and_resize(tf.expand_dims(tf.cast(img, tf.float32), 0),
                                          tf.expand_dims(box_rel, 0), [0],
                                          [CROP_SIZE, CROP_SIZE])
        img_cr = tf.squeeze(img_cr)
        img_cr = img_cr / 255.0 - 0.5
        img_cr.set_shape([CROP_SIZE, CROP_SIZE, 3])

        uv_cr = (uv - box_lt) * (tf.cast(CROP_SIZE, tf.float32) / box_size)

        _ = tf.logical_and(tf.greater_equal(uv_cr[:, 0], 0),
                           tf.logical_and(tf.less(uv_cr[:, 0], CROP_SIZE),
                                          tf.logical_and(tf.greater_equal(uv_cr[:, 1], 0),
                                                         tf.less(uv_cr[:, 1], CROP_SIZE))))
        vis = tf.logical_and(vis, _)

        score_maps = _gen_score_maps(uv_cr, vis, [CROP_SIZE, CROP_SIZE])

        if batch_size < 1:
            record_cooked = copy.copy(record)
            record_cooked['img'] = img
            record_cooked['uv'] = uv
            record_cooked['vis'] = vis
            record_cooked['box'] = box
            record_cooked['img_cr'] = img_cr
            record_cooked['uv_cr'] = uv_cr
            record_cooked['score_maps'] = score_maps
        else:
            _ = tf.train.shuffle_batch_join([[img_cr, uv_cr, score_maps, vis,
                                              record['xyz_rel'],
                                              record['rot_inv'],
                                              record['xyz_cano'],
                                              record['is_right_hand_vec']]],
                                            batch_size=batch_size,
                                            capacity=100,
                                            min_after_dequeue=50,
                                            enqueue_many=False)
            record_cooked = dict(zip(['img_cr', 'uv_cr', 'score_maps', 'vis',
                                      'xyz_rel', 'rot_inv', 'xyz_cano',
                                      'is_right_hand_vec'], _))

        return record_cooked


def join_cooked_records_ab(a, b, is_a):
    assert a.keys() == b.keys()

    record = {}
    for key in a.keys():
        record[key] = tf.cond(is_a, lambda: a[key], lambda: b[key])

    return record


def _test_1(tfrecord_path):
    cv2.namedWindow('uv')
    cv2.namedWindow('uv_cr')
    cv2.namedWindow('score_maps')

    record = cook_tfrecord([tfrecord_path], batch_size=0,
                           uv_noise=False,
                           crop_center_noise=False)

    with TFSession() as sess:
        res = sess.run(record)

        # is shapes all right ?
        for key, value in res.items():
            if type(value) == np.ndarray:
                print(key, value.shape)

        while True:
            res = sess.run(record)

            img = res['img']
            uv = res['uv']
            vis = res['vis']
            box = res['box']
            img_cr = res['img_cr']
            uv_cr = res['uv_cr']
            score_maps = res['score_maps']

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_cr = (img_cr - np.min(img_cr)) / (np.max(img_cr) - np.min(img_cr))
            img_cr = (img_cr * 255.0).astype(np.uint8)
            img_cr = cv2.cvtColor(img_cr, cv2.COLOR_RGB2BGR)

            # draw `uv`
            for this_uv, this_vis in zip(uv, vis):
                if not this_vis:
                    continue

                this_uv = this_uv.astype(np.int64)

                cv2.rectangle(img,
                              (this_uv[0] - 1, this_uv[1] - 1),
                              (this_uv[0] + 1, this_uv[1] + 1),
                              (0, 255, 0), 1)

            # draw `box`
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 255, 255), 1)

            cv2.imshow('uv', img)

            # draw `uv_cr`
            for this_uv, this_vis in zip(uv_cr, vis):
                if not this_vis:
                    continue

                this_uv = this_uv.astype(np.int64)

                cv2.rectangle(img_cr,
                              (this_uv[0] - 1, this_uv[1] - 1),
                              (this_uv[0] + 1, this_uv[1] + 1),
                              (0, 255, 0), 1)

            cv2.imshow('uv_cr', img_cr)

            # show `score_maps`
            for i in range(NUM_KEY_POINTS):
                cv2.imshow('score_maps', score_maps[:, :, i])
                if cv2.waitKey(0) == ord('q'):
                    break

            if cv2.waitKey(0) == ord('q'):
                break

    cv2.destroyAllWindows()


def _test_2(tfrecord_path):
    record = cook_tfrecord([tfrecord_path], uv_noise=True, crop_center_noise=True)

    with TFSession() as sess:
        res = sess.run(record)

        # is shapes all right ?
        for key, value in res.items():
            if type(value) == np.ndarray:
                print(key, value.shape)


def _test_3():
    cv2.namedWindow('img_cr')

    record_r = cook_tfrecord([R_VAL_RECORD_PATH])
    record_s = cook_tfrecord([S_VAL_RECORD_PATH])

    is_r = tf.placeholder(tf.bool, shape=[])
    record = join_cooked_records_ab(record_r, record_s, is_r)

    with TFSession() as sess:
        for _ in range(100):
            res = sess.run(record, feed_dict={is_r: _ % 2 == 0})

            img_cr = res['img_cr'][0]
            img_cr = (img_cr - np.min(img_cr)) / (np.max(img_cr) - np.min(img_cr))
            img_cr = (img_cr * 255.0).astype(np.uint8)
            img_cr = cv2.cvtColor(img_cr, cv2.COLOR_RGB2BGR)

            cv2.imshow('img_cr', img_cr)

            if cv2.waitKey(0) == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import numpy as np
    import cv2

    print('\n------ test_1/R ------\n')
    _test_1(R_VAL_RECORD_PATH)
    print('\n------ test_1/S ------\n')
    _test_1(S_VAL_RECORD_PATH)

    print('\n------ test_2/R ------\n')
    _test_2(R_VAL_RECORD_PATH)
    print('\n------ test_2/S ------\n')
    _test_2(S_VAL_RECORD_PATH)

    print('\n------ test_3 ------\n')
    _test_3()
