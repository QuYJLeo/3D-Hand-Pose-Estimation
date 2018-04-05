import os
import sys
import time
import pickle

import tensorflow as tf

from pose_estimation.PoseNet import PoseNet
from pose_estimation.config import *

from key_points_estimation.cook_tfrecord import cook_tfrecord

from utils.TFSession import TFSession
from utils.misc import restore_weights
from utils.ProgressMsgDisplayer import ProgressMsgDisplayer


# to suppress some tensorflow logs, especially the 'Out of range: FIFOQueue' warnning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL


def eval(get_ckpt_fn=None):
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    record = cook_tfrecord([S_VAL_RECORD_PATH, R_VAL_RECORD_PATH], num_epochs=1)

    xyz_rel = record['xyz_rel']
    score_maps = record['score_maps']
    is_right_hand_vec = record['is_right_hand_vec']

    score_maps_pooled = tf.nn.avg_pool(score_maps,
                                       ksize=[1, 8, 8, 1],
                                       strides=[1, 8, 8, 1],
                                       padding='SAME')
    pose_net = PoseNet(score_maps_pooled, is_right_hand_vec, training=tf.constant(False))
    xyz_rel_pred = pose_net['xyz_rel']

    assert xyz_rel.shape.as_list() == [BATCH_SIZE, NUM_KEY_POINTS, 3]
    assert xyz_rel.shape.as_list() == xyz_rel_pred.shape.as_list()

    xyz_rel = tf.reshape(xyz_rel, shape=[-1, 3])
    xyz_rel_pred = tf.reshape(xyz_rel_pred, shape=[-1, 3])

    xyz_rel_diff = tf.sqrt(tf.reduce_sum(tf.square(xyz_rel_pred - xyz_rel), axis=1))

    xyz_rel_diff_hist = tf.placeholder(tf.float32)
    xyz_rel_diff_avg = tf.reduce_mean(xyz_rel_diff_hist)

    tf.summary.histogram('xyz_rel_diff_hist', xyz_rel_diff_hist)
    tf.summary.scalar('xyz_rel_diff_avg', xyz_rel_diff_avg)
    summary_op = tf.summary.merge_all()

    ckpt_path = ''
    num_examples = 0

    summary_writer = tf.summary.FileWriter(VAL_DIR)

    # continue from where we left
    to_be_continued_pickle_path = os.path.join(VAL_DIR, 'to_be_continued.pickle')
    to_be_continued = None

    try:
        with open(to_be_continued_pickle_path, 'rb') as ifs:
            to_be_continued = pickle.load(ifs)
    except FileNotFoundError:
        pass

    if to_be_continued is not None:
        ckpt_path = to_be_continued['ckpt_path']
        num_examples = to_be_continued['num_examples']
    else:
        to_be_continued = {}

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

    if num_examples == 0:
        with TFSession(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            print('Getting the number of evaluating examples ......')

            n = 0
            with ProgressMsgDisplayer() as progress_msg_displayer:
                # `tf.errors.OutOfRangeError` will be raised when we run out of evaluating examples
                try:
                    while True:
                        _ = sess.run(record['vis'])
                        n += _.shape[0]
                        progress_msg_displayer.update('%05d' % n)
                except tf.errors.OutOfRangeError:
                    pass

            num_examples = n
            to_be_continued['num_examples'] = num_examples

    try:
        while True:
            # get new checkpoint
            if get_ckpt_fn is None:
                _ = tf.train.latest_checkpoint(TRAIN_DIR)
                if _ is None or _ == ckpt_path:
                    time.sleep(VAL_PERIOD_CHECK)
                    continue
                ckpt_path = _
            else:
                ckpt_path = get_ckpt_fn()

            to_be_continued['ckpt_path'] = ckpt_path

            # then evaluate it
            with TFSession(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                print('Restoring model from %s ...... ' % ckpt_path, end='', flush=True)

                # restore weights. Here `tf.train.Saver.restore()` is not used,
                # since extra restoration will override the effect of `num_epochs=1`
                # passed to `cook_tfrecord()`.
                restore_rules = [
                    (None, r'^PosePriorNet/.*/bias(:\d)?$', None, None),
                    (None, r'^PosePriorNet/.*/kernel(:\d)?$', None, None),
                    (None, r'^ViewpointNet/.*/bias(:\d)?$', None, None),
                    (None, r'^ViewpointNet/.*/kernel(:\d)?$', None, None),
                    (None, r'.*global_step', None, None)
                ]
                restore_weights(sess, ckpt_path, restore_rules)
                print('Done')

                print('Evaluating ......')
                _global_step = sess.run(global_step)

                _xyz_rel_diff_hist = []
                with ProgressMsgDisplayer() as progress_msg_displayer:
                    # `tf.errors.OutOfRangeError` will be raised when we run out of evaluating examples
                    try:
                        n = 0
                        while True:
                            _, _xyz_rel_diff = sess.run([record['vis'], xyz_rel_diff])

                            n += _.shape[0]
                            progress_msg_displayer.update('%05d/%05d' % (n, num_examples))

                            _xyz_rel_diff_hist.extend(_xyz_rel_diff)
                    except tf.errors.OutOfRangeError:
                        pass

                print('Writting summary to %s ...... ' % summary_writer.get_logdir(), end='', flush=True)
                summary = sess.run(summary_op, feed_dict={xyz_rel_diff_hist: _xyz_rel_diff_hist})
                summary_writer.add_summary(summary, global_step=_global_step)
                print('Done')
    except KeyboardInterrupt:
        pass
    finally:
        summary_writer.close()

        if len(to_be_continued) > 0:
            with open(to_be_continued_pickle_path, 'wb') as ofs:
                pickle.dump(to_be_continued, ofs)


if __name__ == '__main__':
    eval()
