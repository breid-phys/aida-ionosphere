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

from ..config import AIDAConfig
from ..logger import AIDAlogger


logger = AIDAlogger(__name__)


###############################################################################


class Modip(object):
    """
    object to interpolate modip

    """

    def __init__(self, InputFile=None):
        """ """
        if InputFile is None:
            InputFile = (
                AIDAConfig()["config"]["moduledata"]
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

    def interp(self, lat, lon):
        if not lat.shape == lon.shape:
            raise Exception("lat and lon inputs must have the same shape")

        return self.Interpolant.ev(lat, np.mod(lon + 180.0, 360.0) - 180.0)
