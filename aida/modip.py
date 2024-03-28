#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:51:32 2022

@author: breid
"""
from __future__ import annotations

import os
import h5py
import numpy as np
import scipy.interpolate as spInt
from importlib import resources


###############################################################################


class Modip(object):
    """
    object to interpolate modip

    """

    def __init__(self, InputFile=None):
        """ """
        if InputFile is None:
            InputFile = (
                resources.files("aida")
                .joinpath("data")
                .joinpath("data_hires.h5")
                .expanduser()
            )

        if os.path.exists(os.path.join(InputFile)):
            fileLoc = os.path.join(InputFile)
            openFile = h5py.File(fileLoc, "r")
            latModip = openFile["MODIP/latitude"][()]
            lonModip = openFile["MODIP/longitude"][()]
            modip = openFile["MODIP/modip"][()]
            openFile.close()
        else:
            FileNotFoundError(InputFile)

        self.Interpolant = spInt.RectBivariateSpline(
            latModip, lonModip, modip, kx=3, ky=3
        )

    def interp(self, lat: np.array, lon: np.array) -> np.array:
        if not isinstance(lat, np.ndarray):
            lat = np.array(lat, dtype=float)
        if not isinstance(lon, np.ndarray):
            lon = np.array(lon, dtype=float)

        if not lat.shape == lon.shape:
            raise Exception("lat and lon inputs must have the same shape")

        return self.Interpolant.ev(lat, np.mod(lon + 180.0, 360.0) - 180.0)
