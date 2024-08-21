#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:51:32 2022

@author: breid
"""
from __future__ import annotations

import numpy as np

from .logger import AIDAlogger

logger = AIDAlogger(__name__)


###############################################################################


class Parameter(object):
    """
     object to store all relevant fields to recreate a profile parameter

     Attributes
     ==========
     name:string
         name of parameter e.g. 'NmF2', 'hmF2'
         (default: None)

     ptype:string
         controls behaviour of parameter
         possible values:
             'active': used in assimilation
             'static': not changed in assimilation, still varies spatially
             'constant': single, unchanging value e.g. hmE=120 km

     scale:string
         determines if fit is linear or log scale
         possible values:
             'abs': absolute, linear
             'log': log of values

     numParticles:int, float
         number of particles used in parameter
         if ptype is anything other than active this is overwritten to 1

     order:int, float
         highest order of spherical hamonics to use
         if ptype is constant this is overwritten to 0

    numDim:int
         number of dimensions of the state vector
         equal to (order+1)**2

     parameters:ndarray, float
         state parameters for the assimilation
         size will be (numDim, numParticles)

     velocity:ndarray, float
         state velocity for the assimilation
         size will be (numDim, numParticles)

     coords:string
         indicates if this parameters uses geographic (lat/lon) or
         modip (modip/lon) coordinates
         possible values:
             'geo': geographic
             'modip': modip, default


    """

    _statesized = ["parameters", "velocity", "bkgvelocity", "acceleration"]
    _bkgsized = ["bkgparameters", "temperature"]
    _output = ["parameters", "bkgparameters"]

    def __init__(self, **kwargs):
        self.name = None
        self.ptype = None
        self.scale = None
        self.numParticles = None
        self.order = None
        self.parameters = None
        self.velocity = None
        self.coords = None
        self.bkgparameters = None
        self.bkgvelocity = None
        self.temperature = None
        self.acceleration = None
        self.kx = None
        self.kv = None
        self.kT = None
        self.k_umin = None
        self.k_uk = None

        for key in kwargs:
            if key in dir(self):
                setattr(self, key, kwargs[key])
            else:
                logger.error(f"Unrecognized keyword {key}")

        if self.ptype is None:
            self.ptype = "static"

        if self.scale is None:
            self.scale = "abs"

        if self.numParticles is None:
            self.numParticles = 1

        if self.order is None:
            self.order = 18

        if self.coords is None:
            self.coords = "modip"

        if self.kx is None:
            self.kx = 0.95

        if self.kv is None:
            self.kv = 0.7

        if self.kT is None:
            self.kT = 0.95

        if self.k_umin is None:
            self.k_umin = 2.0

        if self.k_uk is None:
            self.k_uk = 0.5e-4

        # Check inputs
        chkFlag = self.__inputCheck__()

        if isinstance(chkFlag, Exception):
            raise (chkFlag)

        return

    ######################################################

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            pass
        elif not isinstance(value, str):
            value = f"{value}"

        self._name = value

    ######################################################

    @property
    def ptype(self):
        return self._ptype

    @ptype.setter
    def ptype(self, value):
        if value is None:
            pass
        elif value not in ["active", "static", "constant"]:
            raise ValueError(f" Unrecognized ptype {value}")
        self._ptype = value

    ######################################################

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value):
        if value is None:
            pass
        elif value not in ["modip", "geo"]:
            raise ValueError(f" Unrecognized coords {value}")
        self._coords = value

    ######################################################

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is None:
            pass
        elif value not in ["abs", "log"]:
            raise ValueError(f" Unrecognized coords {value}")
        self._scale = value

    ######################################################

    @property
    def numParticles(self):
        return self._numParticles

    @numParticles.setter
    def numParticles(self, value):
        if value is None:
            pass
        else:
            if not isinstance(value, int):
                value = int(value)

            if value < 1:
                return ValueError(
                    f"Parameter {self.name}: "
                    f"Number of particles must be a positive integer " + f"({value})"
                )
        self._numParticles = value

    ######################################################

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if value is None:
            pass
        else:
            if not isinstance(value, int):
                value = int(value)

            if value < 0:
                return ValueError(
                    f"Parameter {self.name}: "
                    f"Order must be a non-negative integer " + f"({value})"
                )

        self._order = value

    ######################################################

    @property
    def numDim(self):
        if self._order is None:
            return None
        else:
            return int(self._order + 1) ** 2

    @numDim.setter
    def numDim(self, value):
        pass

    ######################################################

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if value is None:
            self._parameters = None
        else:
            self._parameters = np.array(value, dtype=float, ndmin=2)

    ######################################################

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if value is None:
            self._velocity = None
        else:
            self._velocity = np.array(value, dtype=float, ndmin=2)

    ######################################################

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value):
        if value is None:
            self._acceleration = None
        else:
            self._acceleration = np.array(value, dtype=float, ndmin=2)

    ######################################################

    @property
    def bkgparameters(self):
        return self._bkgparameters

    @bkgparameters.setter
    def bkgparameters(self, value):
        if value is None:
            self._bkgparameters = None
        else:
            self._bkgparameters = np.array(value, dtype=float, ndmin=2)

    ######################################################

    @property
    def bkgvelocity(self):
        return self._bkgvelocity

    @bkgvelocity.setter
    def bkgvelocity(self, value):
        if value is None:
            self._bkgvelocity = None
        else:
            self._bkgvelocity = np.array(value, dtype=float, ndmin=2)

    ######################################################

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value is None:
            self._temperature = None
        else:
            self._temperature = np.array(value, dtype=float, ndmin=2)

    ######################################################

    @property
    def kx(self):
        return self._kx

    @kx.setter
    def kx(self, value):
        if value is None:
            pass
        else:
            if not isinstance(value, float):
                value = float(value)

            value = np.clip(value, 0.0, 1.0).ravel()
        self._kx = value

    ######################################################

    @property
    def kv(self):
        return self._kv

    @kv.setter
    def kv(self, value):
        if value is None:
            pass
        else:
            if not isinstance(value, float):
                value = float(value)

            value = np.clip(value, 0.0, 1.0).ravel()
        self._kv = value

        ######################################################

    @property
    def kT(self):
        return self._kT

    @kT.setter
    def kT(self, value):
        if value is None:
            pass
        else:
            if not isinstance(value, float):
                value = float(value)

            value = np.clip(value, 0.0, 1.0).ravel()
        self._kT = value

    ######################################################

    @property
    def k_umin(self):
        return self._k_umin

    @k_umin.setter
    def k_umin(self, value):
        if value is None:
            pass
        else:
            if not isinstance(value, float):
                value = float(value)

            value = np.fmax(value, 0.0).ravel()
        self._k_umin = value

    ######################################################

    @property
    def k_uk(self):
        return self._k_uk

    @k_uk.setter
    def k_uk(self, value):
        if value is None:
            pass
        else:
            if not isinstance(value, float):
                value = float(value)

            value = np.fmax(value, 0.0).ravel()
        self._k_uk = value

    ######################################################

    def __inputCheck__(self):
        """Checks validity of input parameters

        Returns
        =======
        chkFlag:int
            Integer flag. 0=good, 1=fail


        .. todo:: None

        |"""
        chkFlag = 0

        if self.ptype == "constant":
            if self.numParticles != 1:
                logger.warning(
                    f"Parameter {self.name} with {self.ptype} type has more "
                    + f"than one particle ({self.numParticles})"
                )
                self.numParticles = 1
            self.order = 0
        elif self.ptype == "static":
            if self.numParticles != 1:
                logger.warning(
                    f"Parameter {self.name} with {self.ptype} type has more "
                    + f"than one particle ({self.numParticles})"
                )
                self.numParticles = 1
        elif self.ptype == "active":
            pass
        else:
            return ValueError(
                f"Parameter {self.name}: " f"Unrecognized type {self.ptype}"
            )

        # Make sure all relevant inputs are numpy arrays
        for prop in self._statesized:
            attr = getattr(self, prop)
            if attr is None:
                continue

            if not attr.shape == (
                self.numParticles,
                self.numDim,
            ):
                return ValueError(
                    f"Parameter {self.name}: {prop} "
                    "Input parameters {0} do not match expected size {1}".format(
                        attr.shape, (self.numParticles, self.numDim)
                    )
                )

        # Make sure all relevant inputs are numpy arrays
        for prop in self._bkgsized:
            attr = getattr(self, prop)
            if attr is None:
                continue

            if not attr.shape == (
                1,
                self.numDim,
            ):
                return ValueError(
                    f"Parameter {self.name}: {prop} "
                    "Input parameters {0} do not match expected size {1}".format(
                        attr.shape, (1, self.numDim)
                    )
                )

        return chkFlag

    def __str__(self):
        out = ""
        for attr in [key for key in dir(self) if key[0] != "_"]:
            if getattr(self, attr) is None:
                continue

            if attr in self._statesized + self._bkgsized:
                out += f"{attr}: {getattr(self, attr).shape}\n"
            else:
                out += f"{attr}: {getattr(self, attr)}\n"
        return out
