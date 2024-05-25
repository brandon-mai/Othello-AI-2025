import threading
import sys
import time

class FunctionTimedOut(Exception):
    pass

class StoppableThread(threading.Thread):
    def __init__(self, target, args=(), kwargs={}):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

def func_timeout(timeout, func, args=(), kwargs=None):
    ret = []
    exception = []
    isStopped = False

    def funcwrap(args2, kwargs2):
        try:
            ret.append(func(*args2, **kwargs2))
        except FunctionTimedOut:
            pass
        except Exception as e:
            exc_info = sys.exc_info()
            if isStopped is False:
                e.__traceback__ = exc_info[2].tb_next
                exception.append(e)

    if not kwargs:
        kwargs = {}

    thread = StoppableThread(target=funcwrap, args=(args, kwargs))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        isStopped = True
        thread.stop()
        thread.join(min(0.1, timeout / 50.0))
        raise FunctionTimedOut()

    if exception:
        raise FunctionTimedOut()

    if ret:
        return ret[0]
