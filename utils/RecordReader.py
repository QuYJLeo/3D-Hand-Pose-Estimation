import tensorflow as tf


class RecordReader:
    def __init__(self, features_dict, tfrecord_paths_list, num_epochs=1):
        self.features_dict = features_dict
        self.tfrecord_paths_list = tfrecord_paths_list
        self.num_epochs = num_epochs

        self._entered = False

    def __enter__(self):
        assert(not self._entered)

        self._filename_queue = tf.train.string_input_producer(self.tfrecord_paths_list, num_epochs=self.num_epochs)

        self._tfrecord_reader = tf.TFRecordReader()

        _, self._serialized_example = self._tfrecord_reader.read(self._filename_queue)

        self._example = tf.parse_single_example(self._serialized_example, features=self.features_dict)

        self._sess = tf.Session()

        self._init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self._sess.run(self._init_op)

        # Start input enqueue threads.
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(sess=self._sess, coord=self._coord)

        self._entered = True

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._entered

        # when done, ask the threads to stop.
        self._coord.request_stop()

        # wait for threads to finish.
        self._coord.join(self._threads)
        self._sess.close()

        self._entered = False

        # return `True` to prevent the exception from being propagated
        if exc_type == tf.errors.OutOfRangeError:
            return True

    def __iter__(self):
        assert self._entered

        while True:
            if self._coord.should_stop():
                break
            yield self._sess.run(self._example)
