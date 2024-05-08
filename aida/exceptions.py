#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ben
"""


class ConfigurationMismatch(Exception):
    def __init__(self, message=" input file does not match state"):
        self.message = message
        super().__init__(self.message)
