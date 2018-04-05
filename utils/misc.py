import os
import re
import pickle

import tensorflow as tf


def schedule_learning_rate(global_step, step_list, lr_list):
    assert len(step_list) == len(lr_list)

    pred = tf.constant(True)
    learning_rate = tf.constant(0, dtype=tf.float32)
    for step, lr in zip(step_list, lr_list):
        pred = tf.logical_and(pred, tf.greater_equal(global_step, step))
        learning_rate = tf.cond(pred,
                                lambda: tf.constant(lr, dtype=tf.float32),
                                lambda: learning_rate)

    return learning_rate


def _schedule_learning_rate_test():
    global_step = tf.placeholder(tf.int64, shape=[])

    learning_rate = schedule_learning_rate(global_step, [0, 10, 20, 30], [0, 0.1, 0.2, 0.3])

    with tf.Session() as sess:
        for step in range(50):
            lr = sess.run(learning_rate, feed_dict={global_step: step})
            print('%02d %2.1f' % (step, lr))


def restore_weights(sess, ckpt_path, restore_rules):
    var_names_a = [_.name for _ in tf.global_variables()]
    weights_dict_a = {}


    ckpt_reader = tf.train.NewCheckpointReader(ckpt_path)
    var_names_b = ckpt_reader.get_variable_to_shape_map().keys()
    weights_dict_b = {}
    for _ in var_names_b:
        weights_dict_b[_] = ckpt_reader.get_tensor(_)

    for re_a, re_b, sort_func_a, sort_func_b in restore_rules:
        assert re_b is not None

        re_b = re.compile(re_b)
        keys_b = sorted(filter(lambda _: re_b.match(_) is not None, var_names_b), key=sort_func_b)

        if re_a is None:
            keys_a = keys_b
        else:
            re_a = re.compile(re_a)
            keys_a = sorted(filter(lambda _: re_a.match(_) is not None, var_names_a), key=sort_func_a)

        assert len(keys_a) == len(keys_b)

        for key_a, key_b in zip(keys_a, keys_b):
            weights_dict_a[key_a] = weights_dict_b[key_b]

        assign_op, feed_dict = tf.contrib.framework.assign_from_values(weights_dict_a)
        sess.run(assign_op, feed_dict=feed_dict)


if __name__ == '__main__':
    _schedule_learning_rate_test()
