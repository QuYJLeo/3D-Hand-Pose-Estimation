import io
import sys
import traceback

import PIL
import numpy as np
import tensorflow as tf

from key_points_estimation.config import *

from utils.RecordReader import RecordReader


features_dict = {
    'img_file': tf.FixedLenFeature([], dtype=tf.string),
    'img': tf.FixedLenFeature([], dtype=tf.string),
    'uv': tf.FixedLenFeature(shape=[NUM_KEY_POINTS, 2], dtype=tf.float32),
    'xyz': tf.FixedLenFeature(shape=[NUM_KEY_POINTS, 3], dtype=tf.float32),
    'vis': tf.FixedLenFeature(shape=[NUM_KEY_POINTS], dtype=tf.float32),
    'is_right_hand_vec': tf.FixedLenFeature(shape=[2], dtype=tf.float32),
    'K': tf.FixedLenFeature(shape=[3, 3], dtype=tf.float32),
    'trans': tf.FixedLenFeature(shape=[3], dtype=tf.float32),
    'scale': tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    'xyz_rel': tf.FixedLenFeature(shape=[NUM_KEY_POINTS, 3], dtype=tf.float32),
    'rot': tf.FixedLenFeature(shape=[3, 3], dtype=tf.float32),
    'rot_inv': tf.FixedLenFeature(shape=[3, 3], dtype=tf.float32),
    'xyz_cano': tf.FixedLenFeature(shape=[NUM_KEY_POINTS, 3], dtype=tf.float32)
}


def get_record_reader(tfrecord_path_list):
    return RecordReader(features_dict, tfrecord_path_list)


def test(tfrecord_path):
    import cv2

    assert os.path.exists(tfrecord_path)

    cv2.namedWindow('uv')

    with get_record_reader([tfrecord_path]) as record_reader:
        for record in record_reader:
            img_file = record['img_file'].decode('utf-8')

            print(img_file)

            with io.BytesIO(record['img']) as ibs:
                img = np.array(PIL.Image.open(ibs))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            uv = record['uv']
            xyz = record['xyz']
            vis = record['vis']
            K = record['K']

            u1v1_homo = np.dot(K, xyz.T).T
            u1v1 = u1v1_homo[:, :2] / u1v1_homo[:, 2:]

            for this_uv, this_u1v1, this_vis in zip(uv, u1v1, vis):
                if this_vis == 0:
                    continue

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

            cv2.imshow('uv', img)

            while True:
                key = cv2.waitKey(0)
                if key in [ord(i) for i in 'nq']:
                    break

            if chr(key) == 'q':
                break

    cv2.destroyAllWindows()


def test_xyz(tfrecord_path):
    assert os.path.exists(tfrecord_path)

    with get_record_reader([tfrecord_path]) as record_reader:
        for record in record_reader:
            img_file = record['img_file'].decode('utf-8')

            xyz = record['xyz']

            trans = record['trans']
            scale = record['scale']
            xyz_rel = record['xyz_rel']
            rot = record['rot']
            rot_inv = record['rot_inv']
            xyz_cano = record['xyz_cano']

            try:
                for xyz_tmp in [xyz_rel, xyz_cano]:
                    assert np.allclose([0, 0, 0], xyz_tmp[0, :])

                    assert np.allclose(1, np.sqrt(EPSILON + np.sum(np.square(xyz_tmp[12, :] - xyz_tmp[11, :]))))

                def is_equal_xyz(a, b):
                    return np.max(np.abs(a - b)) < 1.0e-5

                assert is_equal_xyz(xyz_rel, (xyz + trans) * scale)
                assert is_equal_xyz(xyz_rel, rot_inv.dot(xyz_cano.T).T)
                assert is_equal_xyz(xyz_cano, rot.dot(xyz_rel.T).T)

                assert is_equal_xyz(0, xyz_cano[12, 0])
                assert is_equal_xyz(0, xyz_cano[12, 2])
                assert 1.0e-5 < xyz_cano[12, 1]

                assert is_equal_xyz(0, xyz_cano[20, 2])
                assert 1.0e-5 < xyz_cano[20, 0]
            except AssertionError as err:
                print('------ `%s`' % img_file)
                traceback.print_tb(err.__traceback__)


if __name__ == '__main__':
    test(R_VAL_RECORD_PATH)
    test(S_VAL_RECORD_PATH)
