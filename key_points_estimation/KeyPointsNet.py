import sys
import tensorflow as tf

from key_points_estimation.config import *

from utils.Net import Net


class KeyPointsNet(Net):
    def __init__(self, X, *, trainable=True, name='KeyPointsNet'):
        Net.__init__(self, X, trainable=trainable)

        shape = X.shape.as_list()

        score_maps_list = []

        with tf.variable_scope(name):
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=64, name='conv1_1')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=64, name='conv1_2')
            self.add_max_pooling()

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=128, name='conv2_1')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=128, name='conv2_2')
            self.add_max_pooling()

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv3_1')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv3_2')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv3_3')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv3_4')
            self.add_max_pooling()

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=512, name='conv4_1')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=512, name='conv4_2')

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv4_3')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv4_4')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv4_5')
            self.add_conv_relu(kernel_size=3, strides=1, n_filters=256, name='conv4_6')

            self.add_conv_relu(kernel_size=3, strides=1, n_filters=128, name='conv4_7')
            A = self.Xs[-1]

            self.add_conv_relu(kernel_size=1, strides=1, n_filters=512, name='conv5_1')
            self.add_conv(kernel_size=1, strides=1, n_filters=NUM_KEY_POINTS, name='conv5_2')
            B = self.Xs[-1]
            score_maps_list.append(B)

            n_layers_per_RU = 5 # the number of layers per Recurrent Unit
            n_RU = 2 # the number of Recurrent Units
            for i in range(1, n_RU + 1):
                with tf.name_scope('RU-%d' % i):
                    self.add_X(tf.concat([B, A], 3))

                    for j in range(1, n_layers_per_RU + 1):
                        self.add_conv_relu(kernel_size=7, strides=1, n_filters=128,
                                           name='conv%d_%d' % (5 + i, j))

                    self.add_conv_relu(kernel_size=1, strides=1, n_filters=128,
                                       name='conv%d_%d' % (5 + i, n_layers_per_RU + 1))
                    self.add_conv(kernel_size=1, strides=1, n_filters=NUM_KEY_POINTS,
                                  name='conv%d_%d' % (5 + i, n_layers_per_RU + 2))
                    B = self.Xs[-1]
                    score_maps_list.append(B)
            self.mark_X('out', B)

            score_maps_list = [tf.image.resize_images(_, (shape[1], shape[2])) for _ in score_maps_list]
            self.mark_X('score_maps_list', score_maps_list)
