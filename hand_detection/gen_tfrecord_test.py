import io
import os
import sys

import cv2
import PIL
import numpy as np
import tensorflow as tf

from hand_detection.config import *

from utils.RecordReader import RecordReader


def get_record_reader(tfrecord_path_list):
    features= {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),

        # `Example`s can have variable number of boxes, so 'VarLenFeature' instead of 'FixedLenFeature'
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64)
    }

    return RecordReader(features, tfrecord_path_list)


def test(tfrecord_path):
    assert(os.path.exists(tfrecord_path))

    cv2.namedWindow('boxes')

    with get_record_reader([tfrecord_path]) as record_reader:
        for record in record_reader:
            assert(type(record['image/object/bbox/xmin']) is tf.SparseTensorValue)

            xmins = record['image/object/bbox/xmin'].values
            xmaxs = record['image/object/bbox/xmax'].values
            ymins = record['image/object/bbox/ymin'].values
            ymaxs = record['image/object/bbox/ymax'].values

            with io.BytesIO(record['image/encoded']) as ibs:
                img = np.array(PIL.Image.open(ibs))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            width, height = record['image/width'], record['image/height']
            for x_min, x_max, y_min, y_max in zip(xmins, xmaxs, ymins, ymaxs):
                x_min = int(x_min * width - 1)
                x_max = int(x_max * width - 1)
                y_min = int(y_min * height - 1)
                y_max = int(y_max * height - 1)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            cv2.imshow('boxes', img)

            while True:
                key = cv2.waitKey(0)
                if key in [ord(i) for i in 'nq']:
                    break

            if chr(key) == 'q':
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    test(E_TRAIN_RECORD_PATH)
