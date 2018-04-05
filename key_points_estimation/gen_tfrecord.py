# Usage Examples:
#
#   python gen_tfrecord.py --percentage_train 0.938 R
#   python gen_tfrecord.py --resize 320 240 --percentage_train 0.833 S
#


import io
import sys
import argparse

import PIL
import numpy as np
import tensorflow as tf

from key_points_estimation.config import *

from utils.ProgressMsgDisplayer import ProgressMsgDisplayer
from utils.DatasetReader import get_examples


def decompose_xyz(xyz):
    xyz = np.copy(xyz)

    # translate s.t. palm is the origin
    trans = -np.copy(xyz[0, :])
    xyz += trans

    # scale s.t. distance between middle mcp and middle pip is 1.0
    scale = 1.0 / np.sqrt(EPSILON + np.sum(np.square(xyz[12, :] - xyz[11, :])))
    xyz *= scale

    xyz_rel = np.copy(xyz)

    # calculate the rotation matrix.
    # Refer to "Mathematics for 3D Game Programming and Computer Graphics"
    # (Third Edition, Eric Lengyel) Ch 4.3

    # rotate along axis Z s.t. middle mcp is at (0, ?y > 0, ?z)
    t = np.arctan2(xyz[12, 0], xyz[12, 1])
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    rot_z = np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])
    xyz = np.dot(rot_z, xyz.T).T

    # rotate along axis X s.t. middle mcp is at (0, ?y > 0, 0)
    t = -np.arctan2(xyz[12, 2], xyz[12, 1])
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])
    xyz = np.dot(rot_x, xyz.T).T

    # rotate along axis Y s.t. little mcp is at (?x > 0, ?y, 0)
    t = np.arctan2(xyz[20, 2], xyz[20, 0])
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    rot_y = np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])
    xyz = np.dot(rot_y, xyz.T).T

    xyz_cano = xyz

    rot = rot_y.dot(rot_x).dot(rot_z)
    rot_inv = np.linalg.inv(rot)

    return trans, scale, xyz_rel, rot, rot_inv, xyz_cano


def create_tf_example(example, size):
    img_file, uv, xyz, vis, is_right_hand, K = example['img_file', 'uv', 'xyz', 'vis', 'is_right_hand', 'K']

    img = PIL.Image.open(img_file)
    width, height = img.size

    if size is not None:
        assert size[0] > 0 and size[1] > 0

        img = img.resize((size[0], size[1]))

        scale_x = float(size[0]) / width
        scale_y = float(size[1]) / height

        # scale `uv`s
        uv[:, 0] *= scale_x
        uv[:, 1] *= scale_y

        # modify camera matrix correspondingly
        M = np.eye(3)
        M[0, 0] = scale_x
        M[1, 1] = scale_y
        K = np.dot(M, K)

    with io.BufferedRandom(io.BytesIO()) as br:
        img.save(br, "JPEG")
        br.seek(0)
        img = br.read()

    assert uv.shape == (NUM_KEY_POINTS, 2)
    assert xyz.shape == (NUM_KEY_POINTS, 3)
    assert vis.shape == (NUM_KEY_POINTS, )
    assert K.shape == (3, 3)

    trans, scale, xyz_rel, rot, rot_inv, xyz_cano = decompose_xyz(xyz)

    for _ in (trans, scale, xyz_rel, rot, rot_inv, xyz_cano):
        if not np.all(np.isfinite(_)):
            print('')
            print('Invalid data encountered, `%s` skiped' % img_file)
            print(_)

            return None

    (uv, xyz, K, trans, scale, xyz_rel, rot, rot_inv, xyz_cano) = [
        np.reshape(_, [-1]) for _ in (
            uv, xyz, K, trans, scale, xyz_rel, rot, rot_inv, xyz_cano
        )
    ]

    is_right_hand_vec = np.array([0, 1.0]) if is_right_hand else np.array([1.0, 0])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'img_file': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_file.encode('utf-8')])),
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        'uv': tf.train.Feature(float_list=tf.train.FloatList(value=uv)),
        'xyz': tf.train.Feature(float_list=tf.train.FloatList(value=xyz)),
        'vis': tf.train.Feature(float_list=tf.train.FloatList(value=vis)),
        'is_right_hand_vec': tf.train.Feature(
            float_list=tf.train.FloatList(value=is_right_hand_vec)),
        'K': tf.train.Feature(float_list=tf.train.FloatList(value=K)),
        'trans': tf.train.Feature(float_list=tf.train.FloatList(value=trans)),
        'scale': tf.train.Feature(float_list=tf.train.FloatList(value=scale)),
        'xyz_rel': tf.train.Feature(float_list=tf.train.FloatList(value=xyz_rel)),
        'rot': tf.train.Feature(float_list=tf.train.FloatList(value=rot)),
        'rot_inv': tf.train.Feature(float_list=tf.train.FloatList(value=rot_inv)),
        'xyz_cano': tf.train.Feature(float_list=tf.train.FloatList(value=xyz_cano))
    }))

    return tf_example


def gen_tfrecord(tfrecord_path, examples, size, verbose=False):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    if verbose:
        print('')
        print('Generating %s ......' % tfrecord_path)

    with ProgressMsgDisplayer(not verbose) as progress_msg_displayer:
        idx = 0
        for example in examples:
            progress_msg_displayer.update('%06d/%06d' % (idx + 1, len(examples)))

            tf_example = create_tf_example(example, size)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                idx += 1

    writer.close()


def main():
    cmdarg_parser = argparse.ArgumentParser(description='''Generate two tfrecords
                        for training and evaluating respectively''')

    cmdarg_parser.add_argument('--percentage_train',
                               type=float,
                               help='the percentage of examples for training')
    cmdarg_parser.add_argument('--resize',
                               metavar=('WIDTH', 'HEIGHT'),
                               dest='size', type=int, nargs=2,
                               help='resize images before writing them into the tfrecords')
    cmdarg_parser.add_argument('which_dataset', choices='RS',
                               help='the dataset from which to get examples')

    cmdargs = cmdarg_parser.parse_args()

    assert 0 <= cmdargs.percentage_train <= 1

    which_dataset = cmdargs.which_dataset

    if which_dataset == 'R':
        train_record_path = R_TRAIN_RECORD_PATH
        val_record_path = R_VAL_RECORD_PATH
    else:
        train_record_path = S_TRAIN_RECORD_PATH
        val_record_path = S_VAL_RECORD_PATH

    examples = get_examples(which_dataset)
    if which_dataset == 'S':
        tmp = []
        for example_left, example_right in examples:
            tmp.append(example_left)
            tmp.append(example_right)
        examples = tmp

    np.random.shuffle(examples)

    num_examples = len(examples)
    num_examples_train = int(num_examples * cmdargs.percentage_train)
    examples_train, examples_val = examples[:num_examples_train], examples[num_examples_train:]

    list(map(lambda args: gen_tfrecord(*args, verbose=True), [
        (train_record_path, examples_train, cmdargs.size),
        (val_record_path, examples_val, cmdargs.size)
    ]))


if __name__ == '__main__':
    main()
