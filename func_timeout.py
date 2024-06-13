import multiprocessing
import threading
import sys
import time
from typing import Any, Callable, Dict, Tuple

class FunctionTimedOut(Exception):
    """
    Custom exception raised when a function exceeds its time limit.
    """
    pass

class StoppableThread(threading.Thread):
    """
    A thread that can be stopped using an event.

    Methods:
        stop(): Sets the stop event to stop the thread.
        stopped(): Checks if the stop event is set.
    """
    
    def __init__(self, target: Callable, args: Tuple = (), kwargs: Dict = {}):
        """
        Initializes the StoppableThread.

        Args:
            target (callable): The target function to run in the thread.
            args (tuple): Arguments to pass to the target function.
            kwargs (dict): Keyword arguments to pass to the target function.
        """
        super().__init__(target=target, args=args, kwargs=kwargs)
        
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """
        Sets the stop event to indicate the thread should stop.
        """
        self._stop_event.set()

    def stopped(self) -> bool:
        """
        Checks if the stop event is set.

        Returns:
            bool: True if the stop event is set, False otherwise.
        """
        return self._stop_event.is_set()

def func_timeout(timeout: float, func: Callable, args: Tuple = (), kwargs: Dict = None) -> Any:
    """
    Executes a function with a time limit. Raises FunctionTimedOut if the function does not complete in time.

    Args:
        timeout (float): Maximum time allowed for the function to run, in seconds.
        func (callable): The function to execute.
        args (tuple): Arguments to pass to the function.
        kwargs (dict, optional): Keyword arguments to pass to the function.

    Returns:
        Any: The return value of the function if it completes in time.

    Raises:
        FunctionTimedOut: If the function execution exceeds the specified timeout.
    """
    
    ret = []
    exception = []
    isStopped = False

    def funcwrap(args2: Tuple, kwargs2: Dict) -> None:
        """
        Wrapper function to call the target function and capture its result or exceptions.

        Args:
            args2 (tuple): Arguments to pass to the target function.
            kwargs2 (dict): Keyword arguments to pass to the target function.
        """
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
        
        # Wait a bit more to ensure the thread stops
        thread.join(min(0.1, timeout / 50.0))
        raise FunctionTimedOut()

    if exception:
        raise FunctionTimedOut()

    if ret:
        return ret[0]
