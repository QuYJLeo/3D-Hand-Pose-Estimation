import os
import re
import glob
import sys
import time
import pickle

import tensorflow as tf

from key_points_estimation.KeyPointsNet import KeyPointsNet
from key_points_estimation.cook_tfrecord import cook_tfrecord
from key_points_estimation.config import *

from utils.TFSession import TFSession
from utils.misc import restore_weights
from utils.ProgressMsgDisplayer import ProgressMsgDisplayer


# to suppress some tensorflow logs, especially the 'Out of range: FIFOQueue' warnning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL


def get_key_points_from_score_maps(score_maps):
    """
    Arguments:
        @score_maps: of shape (N, H, W, C)

    Return:
        @key_points_xy: of shape (N, C, 2)
        @key_points_scores: of shape (N, C)
    """

    N, _, W, C = score_maps.shape.as_list()

    assert C == NUM_KEY_POINTS

    max_flat = tf.argmax(tf.reshape(score_maps, [N, -1, C]), axis=1)
    max_y = tf.truncatediv(max_flat, W)
    max_x = max_flat - max_y * W
    key_points_xy = tf.stack([max_x, max_y], axis=2)

    key_points_scores = tf.squeeze(tf.reduce_max(score_maps, axis=[1, 2]))

    return key_points_xy, key_points_scores


def eval(get_ckpt_path_fn=None):
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    record = cook_tfrecord([S_VAL_RECORD_PATH, R_VAL_RECORD_PATH], num_epochs=1)

    vis = record['vis']
    uv_cr = record['uv_cr']
    img_cr = record['img_cr']

    key_points_net = KeyPointsNet(img_cr)
    score_maps_pred = key_points_net['score_maps_list'][-1]
    uv_cr_pred = get_key_points_from_score_maps(score_maps_pred)[0]
    uv_cr_pred = tf.cast(uv_cr_pred, tf.float32)

    # consider only visible key points when evaluate
    uv_cr_vis = tf.boolean_mask(uv_cr, vis)
    uv_cr_vis_pred = tf.boolean_mask(uv_cr_pred, vis)
    # `uv_cr_vis_diff` is of shape (?,)
    uv_cr_vis_diff = tf.sqrt(tf.reduce_sum(tf.square(uv_cr_vis_pred - uv_cr_vis), axis=1))

    uv_cr_vis_diff_hist = tf.placeholder(tf.float32)
    uv_cr_vis_diff_avg = tf.reduce_mean(uv_cr_vis_diff_hist)

    tf.summary.histogram('uv_cr_vis_diff_hist', uv_cr_vis_diff_hist)
    tf.summary.scalar('uv_cr_vis_diff_avg', uv_cr_vis_diff_avg)
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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

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
            if get_ckpt_path_fn is None:
                _ = tf.train.latest_checkpoint(TRAIN_DIR)
                if _ is None or _ == ckpt_path:
                    time.sleep(VAL_PERIOD_CHECK)
                    continue
                ckpt_path = _
            else:
                ckpt_path = get_ckpt_path_fn()

            # no more
            if ckpt_path is None:
                break

            to_be_continued['ckpt_path'] = ckpt_path

            # then evaluate it
            with TFSession(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                print('Restoring model from %s ...... ' % ckpt_path, end='', flush=True)

                # restore weights. Here `tf.train.Saver.restore()` is not used,
                # since extra restoration will override the `num_epochs=1` passed
                # to `cook_tfrecord()`.
                restore_rules = [
                    (None, r'^KeyPointsNet/.*/bias(:\d)?$', None, None),
                    (None, r'^KeyPointsNet/.*/kernel(:\d)?$', None, None),
                    (None, r'.*global_step', None, None)
                ]
                restore_weights(sess, ckpt_path, restore_rules)
                print('Done')

                print('Evaluating ......')
                _global_step = sess.run(global_step)

                _uv_cr_vis_diff_hist = []
                with ProgressMsgDisplayer() as progress_msg_displayer:
                    # `tf.errors.OutOfRangeError` will be raised when we run out of evaluating examples
                    try:
                        n = 0
                        while True:
                            _, _uv_cr_vis_diff = sess.run([record['vis'], uv_cr_vis_diff])

                            n += _.shape[0]
                            progress_msg_displayer.update('%05d/%05d' % (n, num_examples))

                            _uv_cr_vis_diff_hist.extend(_uv_cr_vis_diff)

                    except tf.errors.OutOfRangeError:
                        pass

                print('Writting summary to %s ...... ' % summary_writer.get_logdir(), end='', flush=True)
                summary = sess.run(summary_op, feed_dict={uv_cr_vis_diff_hist: _uv_cr_vis_diff_hist})
                summary_writer.add_summary(summary, global_step=_global_step)
                print('Done')
    except KeyboardInterrupt:
        pass
    finally:
        summary_writer.close()

        if len(to_be_continued) > 0:
            with open(to_be_continued_pickle_path, 'wb') as ofs:
                pickle.dump(to_be_continued, ofs)


def _gen_ckpt_path():
    ckpt_paths = glob.glob(os.path.join(TRAIN_DIR, 'model') + '-*.data*')

    pattern = re.compile('^' + re.escape(os.path.join(TRAIN_DIR, 'model')) + '-(\d+)')

    ckpt_paths = map(lambda _: pattern.search(_).group(0), ckpt_paths)
    ckpt_paths = sorted(ckpt_paths, key=lambda _: int(pattern.search(_).group(1)))

    for _ in ckpt_paths:
        yield _


def _get_ckpt_path(gen=_gen_ckpt_path()):
    try:
        return next(gen)
    except StopIteration:
        return None


if __name__ == '__main__':
    eval(_get_ckpt_path)
