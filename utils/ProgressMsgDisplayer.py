import sys


class ProgressMsgDisplayer(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, dumb=False):
        self._opened = False
        self._previous_msg = ''
        self._dumb = dumb

    def open(self):
        if self._dumb:
            return

        assert not self._opened

        self._opened = True
        self._previous_msg = ''

    def close(self):
        if self._dumb:
            return

        assert self._opened

        sys.stdout.write('\n')

        self._opened = False
        self._previous_msg = ''

    def __enter__(self):
        self.open()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def update(self, msg):
        if self._dumb:
            return

        assert self._opened

        if len(self._previous_msg) > len(msg):
            sys.stdout.write('\r' + ' ' * len(self._previous_msg))
        sys.stdout.write('\r' + msg)

        self._previous_msg = msg


def _test_progress_msg_displayer():
    import time

    assert ProgressMsgDisplayer() is ProgressMsgDisplayer()

    with ProgressMsgDisplayer() as progress_msg_displayer:
        for i in range(16, 0, -1):
            progress_msg_displayer.update('*' * i)
            time.sleep(1)


if __name__ == '__main__':
    _test_progress_msg_displayer()
