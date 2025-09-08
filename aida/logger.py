#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:50:47 2023

@author: ben
"""

import logging
import time

class AIDALoggerList:
    names = []

    def addName(self, name):
        AIDALoggerList.names.append(name)


def AIDAlogger(name):
    logger = logging.getLogger(name)

    if logger.handlers == []:
        logger.propagate = False
        AIDALoggerList().addName(name)

        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03dZ [%(levelname)s]:%(name)s:%(message)s", datefmt=r"%Y-%m-%dT%H:%M:%S"
        )
        formatter.converter = time.gmtime
        screen_handler = logging.StreamHandler()
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)

    return logger
