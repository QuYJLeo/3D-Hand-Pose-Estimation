# Usage Examples:
#
#   python gen_tfrecord.py --resize 640 360 E
#   python gen_tfrecord.py --num_examples 15000 --percentage_train 0.7 R
#


import io
import sys
import argparse

import PIL
import numpy as np
import tensorflow as tf


sys.path.append('tensorflow_object_detection_API/object_detection')


from hand_detection.config import *

from object_detection.utils import dataset_util, label_map_util

from utils.ProgressMsgDisplayer import ProgressMsgDisplayer
from utils.DatasetReader import get_examples


def create_tf_example(example, size=None, label_map_dict=label_map_util.get_label_map_dict(LABEL_MAP_PATH)):
    img_file, boxes = example['img_file', 'boxes']

    img = PIL.Image.open(img_file)

    width, height = img.size

    img_class_text = 'hand'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for box in boxes:
        if box is not None:
            xmins.append(float(box['x_min'] + 1) / width)
            xmaxs.append(float(box['x_max'] + 1) / width)
            ymins.append(float(box['y_min'] + 1) / height)
            ymaxs.append(float(box['y_max'] + 1) / height)
            classes_text.append(img_class_text.encode('utf-8'))
            classes.append(label_map_dict[img_class_text])

    if len(xmins) == 0:
        return None

    if size is not None:
        width, height = size

        assert(width > 0 and height > 0)

        img = img.resize((width, height))

    with io.BufferedRandom(io.BytesIO()) as br:
        img.save(br, "JPEG")
        br.seek(0)
        img_encoded = br.read()
    img_encoded_format = b'jpeg'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_file.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(img_file.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(img_encoded),
        'image/format': dataset_util.bytes_feature(img_encoded_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))

    return tf_example


def gen_tfrecord(tfrecord_path, examples, size=None, verbose=False):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    if verbose:
        print('')
        print('Generating %s ......' % tfrecord_path)

    with ProgressMsgDisplayer(not verbose) as progress_msg_displayer:
        idx = 0
        for example in examples:
            progress_msg_displayer.update('%06d/%06d' % (idx + 1, len(examples)))

            tf_example = create_tf_example(example, size=size)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                idx += 1

    writer.close()


def main():
    cmdarg_parser = argparse.ArgumentParser(
        description='''Generate two tfrecords for training and evaluating respectively''')

    cmdarg_parser.add_argument('--resize', metavar=('WIDTH', 'HEIGHT'),
                               dest='size', type=int, nargs=2,
                               help='resize images before writing them into the tfrecords')
    cmdarg_parser.add_argument('--num_examples', type=int,
                               help='''the number of examples that will be used. By default
                               all examples in the dataset are used''')
    cmdarg_parser.add_argument('--percentage_train', type=float, default=0.8,
                               help='the percentage of examples for training')
    cmdarg_parser.add_argument('which_dataset', choices='ER',
                               help='the dataset from which to get examples')

    cmdargs = cmdarg_parser.parse_args()

    assert(0 <= cmdargs.percentage_train <= 1)

    if cmdargs.which_dataset == 'E':
        train_record_path = E_TRAIN_RECORD_PATH
        val_record_path = E_VAL_RECORD_PATH
    else:
        train_record_path = R_TRAIN_RECORD_PATH
        val_record_path = R_VAL_RECORD_PATH

    examples = get_examples(cmdargs.which_dataset)

    num_examples = cmdargs.num_examples
    if num_examples is None or num_examples < 0 or num_examples > len(examples):
        num_examples = len(examples)

    if num_examples == 0:
        return

    num_examples_train = int(num_examples * cmdargs.percentage_train)

    np.random.shuffle(examples)

    examples = examples[:num_examples]
    examples_train, examples_val = examples[:num_examples_train], examples[num_examples_train:]

    list(map(lambda args: gen_tfrecord(*args, verbose=True), [
        (train_record_path, examples_train, cmdargs.size),
        (val_record_path, examples_val, cmdargs.size)
    ]))


if __name__ == '__main__':
    main()
