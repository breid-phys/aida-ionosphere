#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:54:38 2023

@author: ben
"""
import aida
import numpy as np
import unittest
import datetime


class Test_modip(unittest.TestCase):
    def test_01_read_modip(self):
        modip = aida.Modip()
        Mod = modip.interp(45.0, 55.0)

        np.testing.assert_allclose(Mod, 53.08643723)

    def test_02_make_modip(self):
        modip = aida.Modip()
        modip_igrf = aida.Modip(use_IGRF=True, igrf_time=datetime.datetime(2020, 1, 1))

        glat = np.linspace(-90.0, 90.0, 45)
        glon = np.linspace(-180.0, 180.0, 75)
        glat, glon = np.meshgrid(glat, glon)

        Mod = modip.interp(glat, glon)
        Mod_igrf = modip_igrf.interp(glat, glon)

        np.testing.assert_allclose(Mod, Mod_igrf, atol=1.5)
