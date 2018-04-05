import tensorflow as tf


class TFSession(object):
    """ A simple `tf.Session` wrapper.

    All graph building should be done before entering a `TFSession`.
    """

    def __init__(self, *args, func=None, **kwargs):
        self._sess = tf.Session(*args, **kwargs)

        if func is not None:
            func(self._sess)

    def __enter__(self):
        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self._sess.run(init_op)

        # Start input enqueue threads
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(sess=self._sess, coord=self._coord)

        return self._sess

    def __exit__(self, exc_type, exc_value, traceback):
        # when done, ask the threads to stop
        self._coord.request_stop()

        # wait for threads to finish
        self._coord.join(self._threads)
        self._sess.close()

        # return `True` to prevent the exception from being propagated
        if exc_type == tf.errors.OutOfRangeError:
            return True
