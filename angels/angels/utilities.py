# -*- coding: utf-8 -*-
"""
Utilities for ANGELS
"""

from .logger import SingletonLogger
from functools import wraps
from typing import Optional

def get_logger(
        name: str = None, logger: Optional[object] = None, filename=None
               ) -> SingletonLogger:
    """Utility function to get or create a logger instance."""
    
    #get logger only if it exists, otherwise create one
    if logger is None:
        logger = SingletonLogger(logger=logger,filename=filename)
    return logger


def with_logging(func):
    """Decorator to add logging to any class method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = self.logger

        if logger.verbose:
            class_name = self.__class__.__name__
            func_name = func.__name__
            logger.log(f"Calling {class_name}.{func_name}")

        try:
            result = func(self, *args, **kwargs)
            return result

        except Exception as e:
            if logger.verbose:
                class_name = self.__class__.__name__
                func_name = func.__name__
                logger.log(
                    f"Error in {class_name}.{func_name}: {str(e)}", level="error"
                )
            raise

    return wrapper

