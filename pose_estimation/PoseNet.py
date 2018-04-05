import sys
import tensorflow as tf

from pose_estimation.config import *

from utils.Net import Net


class PosePriorNet(Net):
    def __init__(self, X, is_right_hand_vec, *, trainable=True, training, name="PosePriorNet"):
        Net.__init__(self, X, trainable=trainable, training=training)

        with tf.variable_scope(name):
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=32, name='conv1_1')
            self.add_conv_relu(kernel_size=3, strides=2, n_filters=32, name='conv1_2')

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=64, name='conv2_1')
            self.add_conv_relu(kernel_size=3, strides=2, n_filters=64, name='conv2_2')

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=128, name='conv3_1')
            self.add_conv_relu(kernel_size=3, strides=2, n_filters=128, name='conv3_2')

            self.add_X(tf.concat([tf.layers.flatten(self.Xs[-1]), is_right_hand_vec], axis=1))

            self.add_dense_relu(512, name='dense1_1')
            self.add_dropout(0.8)

            self.add_dense_relu(512, name='dense2_1')
            self.add_dropout(0.8)

            self.add_dense(NUM_KEY_POINTS * 3, name='dense3_1')

            self.add_X(tf.reshape(self.Xs[-1], [-1, NUM_KEY_POINTS, 3]))
            self.mark_X('out')


class ViewpointNet(Net):
    def __init__(self, X, is_right_hand_vec, *, trainable=True, training, name="ViewpointNet"):
        Net.__init__(self, X, trainable=trainable, training=training)

        shape = X.shape.as_list()

        with tf.variable_scope(name):
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=64, name='conv1_1')
            self.add_conv_relu(kernel_size=3, strides=2, n_filters=64, name='conv1_2')

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=128, name='conv2_1')
            self.add_conv_relu(kernel_size=3, strides=2, n_filters=128, name='conv2_2')

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv3_1')
            self.add_conv_relu(kernel_size=3, strides=2, n_filters=256, name='conv3_2')

            self.add_X(tf.concat([tf.layers.flatten(self.Xs[-1]), is_right_hand_vec], axis=1))

            self.add_dense_relu(256, name='dense1_1')
            self.add_dropout(0.75)

            self.add_dense_relu(128, name='dense2_1')
            self.add_dropout(0.75)

            tmp = self.Xs[-1]

            self.add_X(self._dense(tmp, 1, name='dense3_1'))
            Ax = self.Xs[-1][:, 0]
            self.add_X(self._dense(tmp, 1, name='dense4_1'))
            Ay = self.Xs[-1][:, 0]
            self.add_X(self._dense(tmp, 1, name='dense5_1'))
            Az = self.Xs[-1][:, 0]

            self.add_X(self._get_rotation_matrix(Ax, Ay, Az))
            self.mark_X('out')

    # Get the rotation matrix corresponding to the rotation axis and
    # (encoded) angle. Refer to "Mathematics for 3D Game Programming
    # and Computer Graphics" (Third Edition, Eric Lengyel) Ch 4.3
    def _get_rotation_matrix(self, Ax, Ay, Az):
        theta = tf.sqrt(tf.square(Ax) + tf.square(Ay) + tf.square(Az) + 1e-8)

        c = tf.cos(theta)
        s = tf.sin(theta)
        d = 1 - c

        Ax = Ax / theta
        Ay = Ay / theta
        Az = Az / theta

        res = [c + d * Ax * Ax, d * Ax * Ay - s * Az, d * Ax * Az + s * Ay,
               d * Ax * Ay + s * Az, c + d * Ay * Ay, d * Ay * Az - s * Ax,
               d * Ax * Az - s * Ay, d * Ay * Az + s * Ax, c + d * Az * Az]
        res = tf.stack(res, axis=1)
        res = tf.reshape(res, [-1, 3, 3])
        return res


class PoseNet(Net):
    def __init__(self, X, is_right_hand_vec, *, trainable=True, training):
        Net.__init__(self, X, trainable=trainable, training=training)

        self.pose_prior_net = PosePriorNet(X, is_right_hand_vec, trainable=trainable, training=training)
        xyz_cano = self.pose_prior_net['out']
        self.mark_X('xyz_cano', xyz_cano)

        self.viewpoint_net = ViewpointNet(X, is_right_hand_vec, trainable=trainable, training=training)
        rot_inv = self.viewpoint_net['out']
        self.mark_X('rot_inv', rot_inv)

        with tf.variable_scope('Pose'):
            xyz_rel = tf.matmul(xyz_cano, rot_inv, transpose_b=True)
            self.mark_X('xyz_rel', xyz_rel)
