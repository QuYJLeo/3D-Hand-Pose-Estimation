import tensorflow as tf

from fingerspelling_recognition.config import *
from fingerspelling_recognition.gen_tfrecord_test import features_dict


def cook_tfrecord(tfrecord_path, *, batch_size=BATCH_SIZE, num_epochs=None):
    with tf.variable_scope('cook_tfrecord'):
        filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epochs)

        tfrecord_reader = tf.TFRecordReader()
        _, serialized_record = tfrecord_reader.read(filename_queue)

        record = tf.parse_single_example(serialized_record, features=features_dict)
        _ = tf.train.shuffle_batch_join([[record['img_file'],
                                          record['features'],
                                          record['label_id']]],
                                        batch_size=batch_size,
                                        capacity=100,
                                        min_after_dequeue=50,
                                        enqueue_many=False)
        return dict(zip(['img_file', 'features', 'label_id'], _))
