"""
Routines for timing, timestamping, time conversion, ..
"""

from contextlib import contextmanager
import time


@contextmanager
def timer(title):
    """
    generic timer with title
    Usage :
        ```python
        with timer('Title of process to time'):
            do_process()
        ```
    """
    start = time.time()
    yield
    print(f'{title} - done in {(time.time() - start):.0f}s')
