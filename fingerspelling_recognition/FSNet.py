import tensorflow as tf

from fingerspelling_recognition.config import *

from utils.Net import Net


class FSNet(Net):
    def __init__(self, X, *, trainable=True, training, name="FSNet"):
        Net.__init__(self, X, trainable=trainable, training=training)

        with tf.variable_scope(name):
            self.add_dense_relu(512, name='dense1_1')
            self.add_dropout(0.2)

            self.add_dense_relu(512, name='dense2_1')
            self.add_dropout(0.2)

            self.add_dense(NUM_CLASSES, name='dense3_1')
            self.mark_X('out')
