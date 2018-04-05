import tensorflow as tf


class Net(object):
    def __init__(self, X, *, trainable=True, training=None):
        self.Xs = []

        self.add_X(X)

        self._trainable = trainable
        self._training = training

        self._marks = {}

    def add_X(self, X):
        self.Xs.append(X)

    def mark_X(self, name, X=None):
        if X is None:
            X = self.Xs[-1]

        assert isinstance(name, str)

        self._marks[name] = X

    def __getitem__(self, s):
        return self._marks[s]

    def add_relu(self, name='relu'):
        with tf.variable_scope(name):
            X = self.Xs[-1]
            X = tf.maximum(X, X * 0.01)
            self.add_X(X)

    def add_conv(self, kernel_size, strides, n_filters, name=None):
        X = self.Xs[-1]
        X = tf.layers.conv2d(X, n_filters, kernel_size, strides, padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.0001),
                             trainable=self._trainable, name=name)
        self.add_X(X)

    def add_conv_relu(self, *args, **kwargs):
        self.add_conv(*args, **kwargs)
        self.add_relu()

    def add_max_pooling(self, name=None):
        X = self.Xs[-1]
        X = tf.layers.max_pooling2d(X, pool_size=2, strides=2, padding='valid', name=name)
        self.add_X(X)

    def _dense(self, X, n_neurons, name=None):
        X = tf.layers.flatten(X)
        X = tf.layers.dense(X, n_neurons, trainable=self._trainable,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.constant_initializer(0.0001), name=name)
        return X

    def add_dense(self, n_neurons, name=None):
        X = self.Xs[-1]
        X = self._dense(X, n_neurons, name=name)
        self.add_X(X)

    def add_dense_relu(self, *args, **kwargs):
        self.add_dense(*args, **kwargs)
        self.add_relu()

    def add_dropout(self, rate, name=None):
        X = self.Xs[-1]
        X = tf.layers.dropout(X, rate, training=self._training, name=name)
        self.add_X(X)
