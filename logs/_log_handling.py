#!/usr/bin/env python3

# logger
import pdb, os
import logging
import sys
from sys import version_info
import time
import datetime
from . import _object_factory as object_factory
from enum import Enum,IntEnum

# Factory config
config = {
    'loggername':'main',
    'filename_debug': 'debug', # Le chemin ne doit pas être écrit en dur
    'filename_release': 'release', # Le chemin ne doit pas être écrit en dur
    'filemode': 'w',
    'format_log': '%(asctime)s [%(name)-5s] [%(levelname)-5.5s]  %(message)s',
    'format_console': '%(name)-5s: %(levelname)-8s %(message)s'
    }

class DEBUG_LEVEL(IntEnum):
    RELEASE = 0
    DEBUG_SOFT = RELEASE + 1

loggers = {}

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = config["format_console"]

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class LOG():
    global loggers
    def __init__(self):
        # Create a custom logger
        #self.logger = logging.getLogger(__name__)
        pass

class LOG_DEBUG():

    def __init__(self):
        LOG.__init__(self)


    def __call__(self,loggername,filename_debug,filemode,format_log,format_console,**_ignored):

        logger = logging.getLogger(loggername)

        if logger.handlers: # Already defined ?
            return logger
        else:
            # Prevent logging from propagating to the root logger
            logger.propagate = False
            logger.setLevel(logging.DEBUG)

            # use the current time for log
            now = datetime.datetime.now()

            # File Handler
            f_handler = logging.FileHandler(filename_debug+'_'+ now.strftime("%Y-%m-%d") +'.log')
            f_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(format_log)
            f_handler.setFormatter(formatter)

            # Stream handler
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.DEBUG)
            #formatter = logging.Formatter(format_console)
            c_handler.setFormatter(CustomFormatter())

            # Add handler
            logger.addHandler(c_handler)
            logger.addHandler(f_handler)

        return logger


class LOG_RELEASE():

    def __init__(self):
        LOG.__init__(self)

    def __call__(self,loggername,filename_release,filemode,format_log,format_console,**_ignored):

        logger = logging.getLogger(loggername)

        if logger.handlers:
            return logger
        else:
            # Prevent logging from propagating to the root logger
            logger.propagate = False
            logger.setLevel(logging.INFO)

            # use the current time for log
            now = datetime.datetime.now()

            # File Handler
            f_handler = logging.FileHandler(filename_release+'_'+ now.strftime("%Y-%m-%d") +'.log')
            f_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(format_log)
            f_handler.setFormatter(formatter)

            # Stream handler
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.INFO)
            #formatter = logging.Formatter(format_console)
            c_handler.setFormatter(CustomFormatter())

            # Add handler
            logger.addHandler(c_handler)
            logger.addHandler(f_handler)

        return logger

factory = object_factory.ObjectFactory()
factory.register_builder('DEBUG', LOG_DEBUG())
factory.register_builder('RELEASE', LOG_RELEASE())
