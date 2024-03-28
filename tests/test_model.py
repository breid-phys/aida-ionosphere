#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:55:29 2023

@author: ben
"""


import aida
import numpy as np
import unittest


class Test_model_bounds(unittest.TestCase):
    def _boundsTester(self, ModelState):
        h = np.linspace(70.0, 3000.0, 100)
        glat = np.linspace(-90.0, 90.0, num=90)  # latitudes
        glon = np.linspace(-180.0, 180.0, num=180)  # longitudes
        Data = ModelState.calc(glat, glon, h, grid="3D", TEC=True, MUF3000=True)

        for Char in Data.data_vars:
            if "glat" not in list(Data[Char].dims):
                continue

            if "Nmp" in Char:
                assert Data[Char].max() > 1e6
                assert Data[Char].max() < 1e11
                assert Data[Char].min() > 0.0
            elif "Nm" in Char:
                assert Data[Char].max() > 1e10
                assert Data[Char].max() < 1e13
                assert Data[Char].min() > 0.0
            elif "hm" in Char:
                assert Data[Char].max() > 90.0
                assert Data[Char].max() < 500
                assert Data[Char].min() > 0.0
            elif "B" in Char:
                assert Data[Char].max() > 1.0
                assert Data[Char].max() < 200
                assert Data[Char].min() > 0.0
            elif "Hp" in Char:
                assert Data[Char].max() > 100.0
                assert Data[Char].max() < 20e3
                assert Data[Char].min() > 0.0
            elif "fo" in Char:
                assert Data[Char].max() > 2.0
                assert Data[Char].max() < 40.0
                assert Data[Char].min() > 0.0
            elif "MUF" in Char:
                assert Data[Char].max() > 2.0
                assert Data[Char].max() < 80.0
                assert Data[Char].min() > 2.0
            elif "TEC" in Char:
                assert Data[Char].max() > 10.0
                assert Data[Char].max() < 200.0
                assert Data[Char].min() > 0.0
            elif "Ne" in Char:
                assert Data[Char].max() > 1e11
                assert Data[Char].max() < 1e13
                assert Data[Char].min() > 0.0
            else:
                raise ValueError(f" unrecognized output {Char}")

    def test_chars_bounded_NeQuick(self):
        ModelState = aida.AIDAState()

        ModelState.readFile("./tests/data/output_1_221201_053500.h5")

        self._boundsTester(ModelState)

    def test_chars_bounded_AIDA(self):
        ModelState = aida.AIDAState()

        ModelState.readFile("./tests/data/output_3_231201_042500.h5")

        self._boundsTester(ModelState)
