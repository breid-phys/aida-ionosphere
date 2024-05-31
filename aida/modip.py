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
import datetime

from .igrf import inclination, inc2modip


###############################################################################


class Modip(object):
    """
    object to interpolate modip

    """

    def __init__(
        self,
        InputFile=None,
        use_IGRF: bool = False,
        igrf_time: datetime.datetime = None,
    ):
        """ """

        self.use_IGRF = use_IGRF

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

        if self.use_IGRF:
            date_decimal = (
                float(igrf_time.year)
                + float(igrf_time.month - 1) / 12.0
                + float(igrf_time.day - 1) / 365.0
            )
            glon = np.mod(lonModip + 180.0, 360.0) - 180.0
            glat = np.sign(latModip) * np.abs(np.mod(latModip + 90.0, 180.0) - 90.0)
            glon, glat = np.meshgrid(glon, glat)
            inc = inclination(date_decimal, glon, glat)
            modip = inc2modip(inc, glat)

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
