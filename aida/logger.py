#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:50:47 2023

@author: ben
"""

import logging


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
            fmt="%(asctime)s:%(levelname)s:%(name)s:%(message)s", datefmt="%H:%M:%S"
        )
        screen_handler = logging.StreamHandler()
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)

    return logger
