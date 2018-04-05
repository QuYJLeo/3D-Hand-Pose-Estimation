import os
import sys
import time

import tensorflow as tf

from key_points_estimation.cook_tfrecord import cook_tfrecord, join_cooked_records_ab
from key_points_estimation.KeyPointsNet import KeyPointsNet
from key_points_estimation.config import *

from utils.TFSession import TFSession
from utils.misc import schedule_learning_rate, restore_weights


# to suppress some tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL


def train():
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    record_r = cook_tfrecord([R_TRAIN_RECORD_PATH], uv_noise=True, crop_center_noise=True)
    record_s = cook_tfrecord([S_TRAIN_RECORD_PATH], uv_noise=True, crop_center_noise=True)

    is_r = tf.equal(tf.mod(global_step, 2), 0)
    record = join_cooked_records_ab(record_r, record_s, is_r)

    vis = record['vis']
    img_cr = record['img_cr']
    score_maps = record['score_maps']
    # to be able to be masked by `vis`
    score_maps = tf.transpose(score_maps, [0, 3, 1, 2])
    # consider only visible key points when compute `loss`
    score_maps = tf.boolean_mask(score_maps, vis)

    key_points_net = KeyPointsNet(img_cr)
    score_maps_pred_list = key_points_net['score_maps_list']

    loss = 0
    for score_maps_pred in score_maps_pred_list:
        score_maps_pred = tf.transpose(score_maps_pred, [0, 3, 1, 2])
        score_maps_pred = tf.boolean_mask(score_maps_pred, vis)

        this_loss = tf.square(score_maps_pred - score_maps)
        this_loss = tf.reduce_mean(this_loss, axis=[1, 2])
        this_loss = tf.reduce_mean(tf.sqrt(this_loss))

        loss += this_loss

    learning_rate = schedule_learning_rate(global_step,
                                           step_list=TRAIN_LEARNING_RATE[0],
                                           lr_list=TRAIN_LEARNING_RATE[1])
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning rate', learning_rate)
    summary_op = tf.summary.merge_all()
    summary_writer = None

    def func(_sess):
        nonlocal summary_writer

        summary_writer = tf.summary.FileWriter(TRAIN_DIR, _sess.graph)

    saver = tf.train.Saver(max_to_keep=TRAIN_max_to_keep,
                           keep_checkpoint_every_n_hours=TRAIN_keep_checkpoint_every_n_hours)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with TFSession(func=func, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ckpt_path = tf.train.latest_checkpoint(TRAIN_DIR)
        if ckpt_path is not None:
            # restore latest checkpoint
            print('Restoring model from %s ...... ' % ckpt_path, end='', flush=True)
            saver.restore(sess, ckpt_path)
            print('Done')
        else:
            # or load pre-trained model
            restore_rules = [
                (r'^KeyPointsNet/conv[1-4]_.*/bias(:\d)?$', r'^CPM/PoseNet/conv[1-4]_.*/biases', None, None),
                (r'^KeyPointsNet/conv5_1/bias(:\d)?$', r'^CPM/PoseNet/conv5_1.*/biases', None, None),
                (r'^KeyPointsNet/conv[1-4]_.*/kernel(:\d)?$', r'^CPM/PoseNet/conv[1-4]_.*/weights', None, None),
                (r'^KeyPointsNet/conv5_1/kernel(:\d)?$', r'^CPM/PoseNet/conv5_1.*/weights', None, None)
            ]
            print('Loading pre-trained model %s ...... ' % MODEL_PATH, end='', flush=True)
            restore_weights(sess, MODEL_PATH, restore_rules)
            print('Done')

        try:
            while True:
                t0 = time.time()
                _global_step, _loss, _, summay = sess.run([global_step, loss, train_op, summary_op])
                t1 = time.time()

                if (_global_step - 1) % TRAIN_PERIOD_SHOW_LOSS == 0:
                    print('step: %05d, loss: %09.6f, running_time: %05.2f sec' % (_global_step, _loss, t1 - t0))

                if (_global_step - 1) % TRAIN_PERIOD_SUMMARY == 0:
                    print('Writting summay to %s ...... ' % summary_writer.get_logdir(), end='', flush=True)
                    summary_writer.add_summary(summay, _global_step)
                    print('Done')

                if (_global_step - 1) % TRAIN_PERIOD_CKPT == 0:
                    print('Saving model to ...... ', end='', flush=True)
                    _ = saver.save(sess, os.path.join(TRAIN_DIR, 'model'), global_step=_global_step)
                    print('%s' % _)

                if (_global_step - 1) >= TRAIN_MAX_STEP:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            summary_writer.close()


if __name__ == '__main__':
    train()
