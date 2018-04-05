import os
import sys
import time

import tensorflow as tf

from fingerspelling_recognition.cook_tfrecord import cook_tfrecord
from fingerspelling_recognition.config import *
from fingerspelling_recognition.FSNet import FSNet

from utils.TFSession import TFSession
from utils.misc import schedule_learning_rate


# to suppress some tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL


def train():
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    record = cook_tfrecord(TRAIN_RECORD_PATH)
    features = record['features']
    label_id = record['label_id']
    label_oh = tf.one_hot(label_id, depth=NUM_CLASSES)

    fs_net = FSNet(features, training=tf.constant(True))
    label_logits = fs_net['out']

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_oh, logits=label_logits)
    loss = tf.reduce_mean(loss)

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

        try:
            t0 = time.time()
            while True:
                _global_step, _loss, _, summay = sess.run([global_step, loss, train_op, summary_op])

                if (_global_step - 1) % TRAIN_PERIOD_SHOW_LOSS == 0:
                    t1 = time.time()
                    print('step: %05d, loss: %09.6f, running_time: %05.3f sec' % (_global_step, _loss, t1 - t0))
                    t0 = t1

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
