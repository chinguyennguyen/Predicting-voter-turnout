"""
SHARED UTILITIES 

This module contains utility functions used across scripts.
Includes logging setup and common helper functions.

"""

import logging
import sys


def get_logger(name):
    """
    Create a logger for scripts.
    
    Parameters:
        name: Logger name (typically __name__)
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger