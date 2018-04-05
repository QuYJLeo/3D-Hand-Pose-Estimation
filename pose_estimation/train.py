import os
import sys
import time

import tensorflow as tf

from key_points_estimation.cook_tfrecord import cook_tfrecord, join_cooked_records_ab

from pose_estimation.PoseNet import PoseNet
from pose_estimation.config import *

from utils.TFSession import TFSession
from utils.misc import schedule_learning_rate


# to suppress some tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL


def train():
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    record_r = cook_tfrecord([R_TRAIN_RECORD_PATH],
                             uv_noise=True,
                             crop_center_noise=True,
                             crop_scale_noise=True,
                             crop_offset_noise=True)
    record_s = cook_tfrecord([S_TRAIN_RECORD_PATH],
                             uv_noise=True,
                             crop_center_noise=True,
                             crop_scale_noise=True,
                             crop_offset_noise=True)

    is_r = tf.equal(tf.mod(global_step, 2), 0)
    record = join_cooked_records_ab(record_r, record_s, is_r)

    score_maps = record['score_maps']
    xyz_cano = record['xyz_cano']
    rot_inv = record['rot_inv']
    is_right_hand_vec = record['is_right_hand_vec']

    score_maps_pooled = tf.nn.avg_pool(score_maps,
                                       ksize=[1, 8, 8, 1],
                                       strides=[1, 8, 8, 1],
                                       padding='SAME')
    pose_net = PoseNet(score_maps_pooled, is_right_hand_vec, training=tf.constant(True))
    xyz_cano_pred = pose_net['xyz_cano']
    rot_inv_pred = pose_net['rot_inv']

    loss = tf.reduce_mean(tf.square(xyz_cano_pred - xyz_cano))
    loss = tf.reduce_mean(tf.square(rot_inv_pred - rot_inv)) + loss

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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with TFSession(func=func, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # restore latest checkpoint
        ckpt_path = tf.train.latest_checkpoint(TRAIN_DIR)
        if ckpt_path is not None:
            print('Restoring model from %s ...... ' % ckpt_path, end='', flush=True)
            saver.restore(sess, ckpt_path)
            print('Done')

        try:
            while True:
                t0 = time.time()
                _global_step, _loss, _, summary = sess.run([global_step, loss, train_op, summary_op])
                t1 = time.time()

                if (_global_step - 1) % TRAIN_PERIOD_SHOW_LOSS == 0:
                    print('step: %05d, loss: %09.6f, running_time: %05.2f sec' % (_global_step, _loss, t1 - t0))

                if (_global_step - 1) % TRAIN_PERIOD_SUMMARY == 0:
                    print('Writting summay to %s ...... ' % summary_writer.get_logdir(), end='', flush=True)
                    summary_writer.add_summary(summary, _global_step)
                    print('Done')

                if (_global_step - 1) % TRAIN_PERIOD_CKPT == 0:
                    print('Saving model to ...... ', end='', flush=True)
                    _ = saver.save(sess, '%s/model' % TRAIN_DIR, global_step=_global_step)
                    print('%s' % _)

                if (_global_step - 1) >= TRAIN_MAX_STEP:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            summary_writer.close()


if __name__ == '__main__':
    train()
