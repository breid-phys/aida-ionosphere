#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:54:38 2023

@author: ben
"""
import aida
import numpy as np
import unittest


class Test_assimilation(unittest.TestCase):
    def test_01_read_modip(self):
        modip = aida.Modip()
        Mod = modip.interp(45.0, 55.0)

        np.testing.assert_allclose(Mod, 53.08643723)
