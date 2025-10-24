# -*- coding: utf-8 -*-
"""
LiSBOA utilities
"""
from typing import Optional, Union
from lisboa.logger import SingletonLogger
import numpy as np
from functools import wraps
from lisboa.config import LisboaConfig

def get_logger(
        name: str = None, verbose: bool = True, logger: Optional[object] = None, filename=None
               ) -> SingletonLogger:
    """Utility function to get or create a logger instance."""
    
    #get logger only if it exists, otherwise create one
    if logger is None:
        logger = SingletonLogger(logger=logger, verbose=verbose,filename=filename)
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

def _load_configuration(config: Union[dict, LisboaConfig]):
    """
    Load configuration from either dictionary, or LisboaConfig object.

    Args:
        config (dict, or LisboaConfig): Configuration source

    Returns:
        LisboaConfig or None: Configuration parameters or None if loading fails
    """
    try:
        if isinstance(config, LisboaConfig):
            return config, "Configuration successfully loaded"
        elif isinstance(config, dict):
            return LisboaConfig(**config), "Configuration successfully loaded"
        else:
            return None, f"Invalid config type. Expected dict or LisboaConfig, got {type(config)}"
            
    except Exception as e:
        return None, f"Error loading configuration: {str(e)}"


def mid(x):
    '''
    Midpoint of 1-D vector
    '''
    return (x[1:]+x[:-1])/2

def cosd(x):
    '''
    Cosine in degrees
    '''
    return np.cos(x/180*np.pi)

def sind(x):
    '''
    Sine in degrees
    '''
    return np.sin(x/180*np.pi)


def sphere2cart(r,azi,ele):
    x=np.outer(r,np.cos(np.radians(ele))*np.cos(np.radians(90-azi)))
    y=np.outer(r,np.cos(np.radians(ele))*np.sin(np.radians(90-azi)))
    z=np.outer(r,np.sin(np.radians(ele)))
    return x,y,z


