import functools
import logging
import os
import threading
import time

from utils.logger import logger


class Timer:
    def __init__(self, tag: str):
        self._tag = tag

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._end = time.perf_counter()
        interval = (self._end - self._start) * 1000  # ms
        logger.debug(f"[Perf][{self._tag}] time_cost={interval:.2f} ms")

    def time_func(self, func, *args, **kwargs):
        with self:
            return func(*args, **kwargs)


def timed(tag: str):
    def decorator(func):
        def wrapped(*args, **kwargs):
            with Timer(tag):
                return func(*args, **kwargs)

        return wrapped

    return decorator


def timed_method(method):
    """
    time an instance method when debugging
    """

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        if logger.level <= logging.DEBUG:
            method_name = method.__name__
            cls_name = (
                args[0].__class__.__name__
                if args and hasattr(args[0], "__class__")
                else ""
            )
            tag = f"{cls_name}::{method_name}"
            with Timer(tag):
                return method(*args, **kwargs)
        else:
            return method(*args, **kwargs)
