import os
import re

import tensorflow as tf

from fingerspelling_recognition.config import *
from fingerspelling_recognition.gen_tfrecord import _func_1

from utils.RecordReader import RecordReader


features_dict = {
    'img_file': tf.FixedLenFeature([], dtype=tf.string),
    'features': tf.FixedLenFeature([63], dtype=tf.float32),
    'label_id': tf.FixedLenFeature([], dtype=tf.int64)
}


def test_1():
    _, examples_train, examples_val = _func_1(DATA_DIR)
    examples = examples_train + examples_val

    pattern = re.compile(r'^(\d+)_(\d+)_(\d+)_cam[12].bmp$')
    for example in examples:
        img_file = os.path.basename(example['img_file'])
        try:
            match = re.match(pattern, img_file)

            assert match is not None
            assert LABELS_ALL[int(match.group(2)) - 1] not in LABELS_EXCLUDED
        except AssertionError:
            print(':(', img_file)


def test_2():
    with RecordReader(features_dict, [TRAIN_RECORD_PATH, VAL_RECORD_PATH]) as record_reader:
        for record in record_reader:
            print(record)
            break

if __name__ == '__main__':
    test_2()
