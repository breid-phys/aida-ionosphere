#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:51:32 2022

@author: breid
"""
from __future__ import annotations

import copy
import datetime as dt
import os
from pathlib import Path
import h5py
import numpy as np
import xarray

from .time import dt2epoch, epoch2dt, epoch2npdt, npdt2epoch
from .ne import Ne_AIDA, Ne_NeQuick, sph_harmonics, _Nm2sNm
from .logger import AIDAlogger
from .parameter import Parameter
from .modip import Modip
from .exceptions import ConfigurationMismatch

logger = AIDAlogger(__name__)


###########################################################################


class AIDAState(object):
    """
    Object to store and interact with the state of the assimilation

    This includes handling the specific configuration of the various profile
    parameters, e.g. NmF2, hmF2, etc.
        See class Parameter

    Contains methods for saving and loading the State to HDF5 files, or
    populating the State from NeQuick
        readFile(inputFile)
        saveFile(outputFile)
        fromNeQuick(*)

    Allows calculation of parameters and densities for given coordinates
        calcValue(lat, lon)
        calcNe(lat, lon, alt)

    Attributes
    ==========
    Time:numeric
        The time (UNIX Epoch) described by the State
        For assimilated outputs, this time corresponds to the centre of the
        assimilation window

    Version:string
        Version information of the file. Not currently used.

    Metadata:dictionary
        Dictionary containing information about how the file was created
        e.g. State.Metadata = {"NeQuickFlux":99.0, "NeQuickVersion":"G"}
        This should only contain data which can be directly written

    Filter:dictionary
        Dictionary containing information relevant to the particle filter
        e.g. State.Filter.Weight gives the weight of each particle
        This should be empty if not produced by the assimilation, and can be
        missing from valid files with a single 'particle' e.g. NeQuick outputs

    **Parameters:Parameter class
        Each profile parameter in NeQuick is described with an instance of the
        Parameter class (included in this file)
        Each of these parameters is individually configureable, and will appear
        as an attribute of this class

        See State.__init__() for details on how to configure the Parameters

    """

    # Parameter names NeQuick uses
    NeQuickCharNames = [
        "NmF2",
        "hmF2",
        "B2top",
        "B2bot",
        "sNmF1",
        "hmF1",
        "B1top",
        "B1bot",
        "sNmE",
        "hmE",
        "Bebot",
        "Betop",
        "Nmpl",
        "Hpl",
        "Nmpt",
        "Hpt",
    ]

    # AIDA Char names
    AIDACharNames = [
        "NmF2",
        "hmF2",
        "B2top",
        "B2bot",
        "NmF1",
        "hmF1",
        "B1top",
        "B1bot",
        "NmE",
        "hmE",
        "Bebot",
        "Betop",
        "Nmpl",
        "Hpl",
        "Nmpt",
        "Hpt",
    ]

    npsmNames = ["Nmpl", "Hpl", "Nmpt", "Hpt"]

    fluxNames = [
        "NeQuickFluxfoF2",
        "NeQuickFluxMUF3000",
        "NeQuickFluxEF1",
        "NPSMFlux",
    ]

    def __init__(
        self,
        Config=None,
        ModipFile=None,
        Parameterization="NeQuick",
        strict_config=False,
    ):
        """
        Setup the State object

        Inputs (Optional)
        =================

        Config:dictionary
            By default, dasp.Parameter uses the following default values:
                ptype = 'static'  - not actively involved in the assimilation
                scale = 'abs'     - absolute scaling (i.e. not logarithmic)
                numParticles = 1  - one particle
                order = 18        - 18 orders of spherical harmonics
                coords='modip'    - modip coordinates (modip/lon)

            Each parameter is individually configurable by passing a dictionary
            with the following format:

            Config = {ParameterName0:{attr0:value0, attr1:value1, ...}\
                      ParameterName1:{attr0:value0, attr1:value1, ...}\
                          ... \
                      ParameterNameN: {attr0:value0, attr1:value1, ...}}

            This does not have to be a complete list, only specify the values
            that need to be changed from the default.

            e.g.
            Config = {'NmF2':{'scale':'log'}, 'hmF2:{'order':12}}
            This input would produce a State where NmF2 is stored as log(NmF2),
            and where hmF2 is stored at a lower resolution

            See dasp.Parameter class for more information on configuration

        ModipFile:string (path):default 'data_hires.h5'

            Specify custom modip file.

            See dasp.Modip class for more information

        Parameterization:

            which parameterization scheme to use (NeQuick or AIDA)

        strict_config:

            If false, when reading a file, the configuration of the file will overwrite the state
            i.e. if the input config says hmF2 has 8 orders, but the file has 10, the state has 10
            If True, any mismatch will raise an ConfigurationMismatch exception

        """
        self._strict_config = strict_config
        self.Time = 0.0
        self.Version = "0.0"
        self.Metadata = {}
        self.Filter = {}

        self.Modip = Modip(ModipFile)

        self.Parameterization = Parameterization

        if Config is None:
            Config = {}

        for Char in self.CharNames:
            if Char in Config.keys():
                temp_p = Parameter(name=Char, **Config[Char])
            elif Char.lower() in Config.keys():
                temp_p = Parameter(name=Char, **Config[Char.lower()])
            else:
                logger.debug(f" model: {Char} not set in model config")
                temp_p = Parameter(name=Char)

            setattr(self, Char, temp_p)

        return

    def __str__(self):
        out = ""
        for attr in self.__dict__:
            out += f"{attr}: {getattr(self, attr)}\n"
        return out

    ###########################################################################

    @property
    def Parameterization(self):
        return self._Parameterization

    @Parameterization.setter
    def Parameterization(self, value):
        if not hasattr(value, "lower"):
            raise TypeError("Parameterization must be str")
        elif (
            self._strict_config
            and hasattr(self, "_Parameterization")
            and value.lower() != self._Parameterization.lower()
        ):
            raise ConfigurationMismatch(" state parameterization does not match input file")
        elif value.lower() == "nequick":
            self._Parameterization = "NeQuick"
            self.CharNames = AIDAState.NeQuickCharNames
        elif value.lower() == "aida":
            self._Parameterization = "AIDA"
            self.CharNames = AIDAState.AIDACharNames
        else:
            raise ValueError(f"Invalid parameterization {value}")

        logger.debug(f" setting charlist to {self.CharNames}")

    ###########################################################################

    @property
    def Time(self):
        return self._Time

    @Time.setter
    def Time(self, value):
        if isinstance(value, np.datetime64):
            self._Time = npdt2epoch(value)
        elif isinstance(value, dt.datetime):
            self._Time = dt2epoch(value)
        elif isinstance(value, (float, int)):
            self._Time = float(value)
        else:
            raise TypeError("Invalid type for State.Time: {type(value)}")

        time = epoch2dt(self._Time)
        UT = time.hour + time.minute / 60.0

        doy = time.month * 30.5 - 15.0

        t = doy + (18.0 - UT) / 24.0
        amrad = np.deg2rad(0.9856 * t - 3.289)
        aLrad = amrad + np.deg2rad(
            1.916 * np.sin(amrad) + 0.020 * np.sin(2.0 * amrad) + 282.634
        )

        self._sdelta = 0.39782 * np.sin(aLrad)
        self._cdelta = np.sqrt(1.0 - self._sdelta * self._sdelta)
        self._UT = UT

    ###########################################################################

    def readFile(self, inputFile):
        """

        Reads HDF5 file

        """
        # Open HDF file
        inputFile = Path(inputFile)

        if not inputFile.exists():
            raise (FileNotFoundError(f" Input file {inputFile.expanduser()} not found"))
            return

        with h5py.File(inputFile, "r") as openFile:
            self.Time = openFile["Time"][()]
            self.Version = openFile["Version"][()]
            if isinstance(self.Version, bytes):
                self.Version = self.Version.decode("utf8")

            self.Metadata = {}

            if "Parameterization" in openFile:
                Parameterization = openFile["Parameterization"][()]
                if isinstance(Parameterization, bytes):
                    Parameterization = Parameterization.decode("utf8")
                self.Parameterization = Parameterization
            else:
                logger.debug("No parameterization metadata, assuming NeQuick")
                self.Parameterization = "NeQuick"

            for key in openFile["Metadata"].keys():
                mdata = openFile["Metadata/" + key][()]
                if isinstance(mdata, bytes):
                    mdata = mdata.decode("utf8")
                self.Metadata[key] = mdata

            for Char in self.CharNames:
                args = {"ptype": "constant", "parameters": 0.0}

                if "Parameters/" + Char not in openFile:
                    if self._strict_config:
                        raise ValueError(f" parameter {Char} missing from input file.")
                    else:
                        logger.error(f" parameter {Char} missing from input file.")

                for attr in [key for key in dir(Parameter()) if key[0] != "_"]:
                    if f"Parameters/{Char}/{attr}" in openFile:
                        args[attr] = openFile[f"Parameters/{Char}/{attr}"][()]
                        if isinstance(args[attr], bytes):
                            args[attr] = args[attr].decode("utf8")

                temp_p = Parameter(**args)

                if self._strict_config:
                    P = getattr(self, Char)

                    for atr in [
                        "name",
                        "numParticles",
                        "order",
                        "ptype",
                        "scale",
                        "coords",
                    ]:
                        atr_1 = getattr(P, atr)
                        atr_2 = getattr(temp_p, atr)
                        if atr_1 != atr_2:
                            raise ConfigurationMismatch(
                                f" parameter {Char} has mismatched {atr}: "
                                f" expected {atr_1}, received {atr_2}"
                            )

                setattr(self, Char, temp_p)

            # set up filter
            N = self.maxParticles()
            self.Filter = {}
            self.Filter["Weight"] = np.ones(N) / N

            if "Filter" in openFile:
                for key in openFile["Filter"].keys():
                    fdata = openFile["Filter/" + key][()]
                    if isinstance(fdata, bytes):
                        fdata = fdata.decode("utf8")
                    self.Filter[key] = fdata

        return

    ###########################################################################

    def saveFile(self, outputFile, is_output: bool = False):
        """

        Saves state to HDF5 file

        """

        outputFile = Path(outputFile)

        if outputFile.exists():
            os.remove(outputFile.expanduser())

        with h5py.File(outputFile, "w") as dataFile:
            # time and version
            tmp_data = dataFile.create_dataset("Time", data=self.Time)
            tmp_data = dataFile.create_dataset("Version", data=self.Version)
            tmp_data = dataFile.create_dataset(
                "Parameterization", data=self.Parameterization
            )

            # metadata
            metaGroup = dataFile.create_group("Metadata")

            for key in self.Metadata.keys():
                if isinstance(self.Metadata[key], list):
                    tmp_data = metaGroup.create_dataset(
                        key, data=str(self.Metadata[key])
                    )
                else:
                    tmp_data = metaGroup.create_dataset(key, data=self.Metadata[key])

            # filter info, if exists
            if bool(self.Filter):
                filterGroup = dataFile.create_group("Filter")

                for key in self.Filter.keys():
                    tmp_data = filterGroup.create_dataset(key, data=self.Filter[key])

            # parameters
            paramGroup = dataFile.create_group("Parameters")

            for Char in self.CharNames:
                charGroup = paramGroup.create_group(Char)

                dP = getattr(self, Char)

                for tAttr in dir(dP):
                    if tAttr.startswith("_"):
                        continue

                    tmp_data = getattr(dP, tAttr)
                    if tmp_data is None:
                        continue

                    if is_output and tAttr not in dP._output:
                        # only save necessary fields for output files
                        continue

                    if tAttr in dP._statesized + dP._bkgsized:
                        tmp_data = charGroup.create_dataset(
                            tAttr,
                            data=tmp_data,
                            shuffle=True,
                            fletcher32=True,
                            compression="gzip",
                            compression_opts=1,
                        )
                    else:
                        tmp_data = charGroup.create_dataset(tAttr, data=tmp_data)

        return

    ###########################################################################
    def background(self):
        """
        returns the background of an AIDAState object

        Returns
        -------
        Background:State

        """

        allCharsHaveBkg = True
        Background = copy.deepcopy(self)
        for Char in Background.CharNames:
            P = getattr(Background, Char)
            P.numParticles = 1
            if P.ptype == "active":
                P.ptype = "static"
            if P.bkgparameters is not None and not np.all(np.isnan(P.bkgparameters)):
                P.parameters = P.bkgparameters
            else:
                allCharsHaveBkg = False
            setattr(Background, Char, P)

        if not allCharsHaveBkg:
            raise ValueError(" missing background parameters.")

        return Background

    ###########################################################################

    def calc(
        self,
        lat,
        lon,
        alt=None,
        grid="1D",
        particleIndex=None,
        as_dict=False,
        collapse_particles=False,
        MUF3000=False,
        TEC=False,
        Weight=True,
    ):
        """

        function to return full output of the model in xarray.Dataset format
        If no alt is given, electron density will not be calculated

        for grid='3D': (3D Regular Grid Mode)
            lat, lon, alt should be 1D vectors, to be broadcast together
            output model parameters (e.g. NmF2) are (nparticle, nlat, nlon)
            output Ne is (nparticle, nlat, nlon, nalt)

        for grid='2D': (Profile Mode / Irregular Lat/Lon Grid)
            lat and lon must have the same shape (nx, [ny,]), either 1D or 2D
            output model parameters (e.g. NmF2) are (nparticle, nx, [ny,])
            output Ne is (nparticle, nx, [ny,], nalt)

        for grid='1D': (Satellite / Line / Irregular Mesh / Point Cloud)
            lat, lon, alt, must have the same shape (nx, [ny, nz,]), up to 3D
            output model parameters (e.g. NmF2) are (nparticle, nx, [ny, nz,])
            output Ne is (nparticle, nx, [ny, nz,])

        Parameters
        ----------
        lat:np.array
            DESCRIPTION.
        lon:np.array
            DESCRIPTION.
        alt:np.array, optional
            DESCRIPTION.
        grid:str, optional
            DESCRIPTION. The default is '1D'.
        particleIndex:np.array, optional
            DESCRIPTION. The default is None.
        as_dict:boolean, optional
            DESCRIPTION. The default is Falses.
            If True, output is in dict form rather than xarray.Dataset
        collapse_particles:boolean, optional
            DESCRIPTION. The default is Falses.
            If True, squeezes output along 'particle' dimension
            If more than one particle, this option is ignored

        Returns
        -------
        Output:xarray.Dataset
            DESCRIPTION.

        If using xarray output, output is easiest to use in '3D' mode:

        <xarray.Dataset>
        Dimensions:  (glat: [nlat], glon: [nlon], alt: [nalt],
                      particle: [nparticles])
        Coordinates:
          * glat     (glat) float64
          * glon     (glon) float64
          * alt      (alt) float64

        for other grids, the glat, glon, (and alt for 1D) will be indexed by
        dimensions labelled "x", "y", and "z", as needed:

        e.g. grid="1D", glat/glon/alt have size (nx, ny, nz)
        <xarray.Dataset>
        Dimensions:  (x: [nx], y: [ny], z: [nz], particle: 3)
        Coordinates:
            glat     (x, y, z) float64
            glon     (x, y, z) float64
            alt      (x, y, z) float64

        If as_dict == True, output is just in a python dict object

        """
        oldsettings = np.geterr()
        np.seterr(over="ignore", invalid="ignore")

        lat = np.atleast_1d(np.array(lat, dtype=float))
        lon = np.atleast_1d(np.array(lon, dtype=float))
        alt = np.atleast_1d(np.array(alt, dtype=float))

        maxParticles = self.maxParticles()

        Values, Size = self._calc(lat, lon, alt, grid, particleIndex, TEC, MUF3000)

        coords = Values.pop("coords")

        ValueSize = {}
        for key in Values:
            if Size["2DShape"] == Values[key].shape:
                ValueSize[key] = Size["2D"]
            elif Size["2DShape"][1:] == Values[key].shape:
                ValueSize[key] = Size["2D"][1:]
            elif Size["3DShape"] == Values[key].shape:
                ValueSize[key] = Size["3D"]
            elif Size["3DShape"][1:] == Values[key].shape:
                ValueSize[key] = Size["3D"][1:]
            else:
                raise ValueError(
                    f"Unexpected parameter shape {key} {Values[key].shape}"
                    f" allowed sizes: {Size}"
                )

        Output = xarray.Dataset(
            coords=coords,
            data_vars={key: (ValueSize[key], Values[key]) for key in ValueSize},
        )

        Output["glat"].attrs = {
            "units": "degrees N",
            "description": "geodetic latitude",
        }
        Output["glon"].attrs = {
            "units": "degrees E",
            "description": "geodetic longitude",
        }

        if "alt" in Output.coords:
            Output["alt"].attrs = {"units": "km", "description": "geodetic altitude"}

        if "Ne" in Output:
            Output["Ne"].attrs = {"units": "m-3", "description": "electron density"}

        if "TEC" in Output:
            Output["TEC"].attrs = {
                "units": "TECU",
                "description": "Total Electron Content to 20 000 km",
            }

        if "MUF3000" in Output:
            Output["MUF3000"].attrs = {
                "units": "MHz",
                "description": "maximum useable frequency "
                "of the F2 layer for 3000 km circuit",
            }

        for Char in Output:

            if Char in ["sNmF1", "sNmE"]:
                continue
            elif "Nmp" in Char:
                Output[Char] = 1e11 * Output[Char]
                CharAttributes = {
                    "units": "m-3",
                    "description": f"peak density of the {Char[2:]} layer",
                }
            elif "Nm" in Char:
                Output[Char] = Output[Char]
                CharAttributes = {
                    "units": "m-3",
                    "description": f"peak density of the {Char[2:]} layer",
                }
            elif "fo" in Char:
                CharAttributes = {
                    "units": "MHz",
                    "description": f"critical frequency of the {Char[2:]} layer",
                }
            elif "hm" in Char:
                Output[Char] = np.fmax(Output[Char], 0.0)
                CharAttributes = {
                    "units": "km",
                    "description": f"altitude of the {Char[2:]} layer peak density",
                }
            elif "B" == Char[0]:
                Output[Char] = np.fmax(Output[Char], 1.0)

                if Char[1] == "e":
                    Layer = "E"
                else:
                    Layer = f"F{Char[1]}"

                CharAttributes = {
                    "units": "km",
                    "description": "thickness of the "
                    f"{Char[-3:]} of the {Layer} layer",
                }
            elif "iB" == Char[:2]:
                Output[Char[1:]] = np.fmax(1.0 / Output[Char], 1.0)

                if Char[2] == "e":
                    Layer = "E"
                else:
                    Layer = f"F{Char[2]}"

                CharAttributes = {
                    "units": "km",
                    "description": "thickness of the "
                    f"{Char[-4:]} of the {Layer} layer",
                }

            Output[Char].attrs = CharAttributes

        Output.attrs["Epoch"] = self.Time
        Output.attrs["Time"] = epoch2npdt(self.Time).astype(str)
        Output.attrs["Version"] = self.Version
        Output.attrs["Grid"] = grid

        for meta in self.Metadata:
            Output.attrs[meta] = self.Metadata[meta]

        if "Weight" in self.Filter and Weight:
            if particleIndex is not None:
                Output["Weight"] = (
                    ("particle",),
                    np.atleast_1d(self.Filter["Weight"][particleIndex]),
                )
            elif particleIndex is None and maxParticles == 1:
                # some output files do not have correct weights
                Output["Weight"] = (("particle",), np.atleast_1d(1.0))
            else:
                Output["Weight"] = (("particle",), np.atleast_1d(self.Filter["Weight"]))

            Output["Weight"].attrs = {"units": "", "description": "unnormalized weight"}

        Output = Output.transpose(*Size["3D"], missing_dims="ignore").drop_vars(
            ["sNmF1", "sNmE"], errors="ignore"
        )

        if collapse_particles and (maxParticles == 1 or len(Output["particle"]) == 1):
            Output = Output.squeeze(dim="particle")

        if as_dict:
            XOutput = Output.copy(deep=True)
            Output = dict()
            for key in XOutput.keys():
                Output[key] = XOutput[key].data

            for key in XOutput.coords.keys():
                Output[key] = XOutput.coords[key].data

            for key in XOutput.attrs.keys():
                Output[key] = XOutput.attrs[key]

        np.seterr(**oldsettings)
        return Output

    ###########################################################################
    def _calc(
        self,
        lat,
        lon,
        alt=None,
        grid="1D",
        particleIndex=None,
        TEC=False,
        MUF3000=False,
    ):
        lat = np.atleast_1d(np.array(lat, dtype=float))
        lon = np.atleast_1d(np.array(lon, dtype=float))
        alt = np.atleast_1d(np.array(alt, dtype=float))

        maxParticles = self.maxParticles()

        if particleIndex is not None:
            particleIndex = np.array(particleIndex, dtype=int, ndmin=1)
            if np.any(particleIndex >= maxParticles):
                raise (IndexError(" particle index out of range."))
            maxParticles = len(particleIndex)

        Output = {}
        if grid == "1D" and np.all(np.isnan(alt)):
            if lon.shape != lat.shape:
                raise (ValueError(f"***{grid} grid: lat and lon must be same shape*"))

            glat = lat
            glon = lon

            crd = ["x", "y", "z"][0 : np.ndim(glat)]

            Output["coords"] = {"glat": (crd, glat), "glon": (crd, glon)}

            Size = {
                "3D": ("particle",) + tuple(crd),
                "2D": ("particle",) + tuple(crd),
                "3DShape": (maxParticles,) + glat.shape,
                "2DShape": (maxParticles,) + glat.shape,
            }

        elif grid == "1D":
            if (
                alt.shape != lat.shape
                or alt.shape != lon.shape
                or lat.shape != lon.shape
            ):
                raise (
                    ValueError(f"***{grid} grid: lat, lon, and alt must be same shape*")
                )

            glat = lat
            glon = lon

            crd = ["x", "y", "z"][0 : np.ndim(glat)]

            Output["coords"] = {
                "glat": (crd, glat),
                "glon": (crd, glon),
                "alt": (crd, alt),
            }

            Size = {
                "3D": ("particle",) + tuple(crd),
                "2D": ("particle",) + tuple(crd),
                "3DShape": (maxParticles,) + glat.shape,
                "2DShape": (maxParticles,) + glat.shape,
            }

        if grid == "2D":
            if lat.shape != lon.shape:
                raise (ValueError(f"{grid} grid:" " lat and lon must be same shape"))
            glat = lat
            glon = lon

            crd = ["x", "y"][0 : np.ndim(glat)]

            Output["coords"] = {"glat": (crd, glat), "glon": (crd, glon), "alt": alt}

            Size = {
                "3D": ("particle",) + tuple(crd) + ("alt",),
                "2D": ("particle",) + tuple(crd),
                "3DShape": (maxParticles,) + glat.shape + (alt.size,),
                "2DShape": (maxParticles,) + glat.shape,
            }

        if grid == "3D":
            if np.ndim(lat) > 1 or np.ndim(lon) > 1 or np.ndim(alt) > 1:
                raise (ValueError(f"{grid} grid:" " lat, lon, and alt must be 1D"))
            glat, glon = np.meshgrid(lat, lon)

            Output["coords"] = {"glat": lat, "glon": lon, "alt": alt}

            Size = {
                "3D": ("particle", "glon", "glat", "alt"),
                "2D": ("particle", "glon", "glat"),
                "3DShape": (
                    maxParticles,
                    lon.size,
                    lat.size,
                    alt.size,
                ),
                "2DShape": (maxParticles, lon.size, lat.size),
            }

        for Char, Data in self._calcValueIterator(
            glat.ravel(), glon.ravel(), particleIndex=particleIndex
        ):
            Output[Char] = Data

        if not np.all(np.isnan(alt)):

            if grid == "1D":
                talt = np.reshape(alt, (1, alt.size))
                tOutput = {Char: np.atleast_2d(Output[Char]) for Char in self.CharNames}
                tglat = np.atleast_2d(glat.ravel())
                tglon = np.atleast_2d(glon.ravel())
            else:
                talt = np.reshape(alt, (1, 1, alt.size))
                tOutput = {Char: np.atleast_3d(Output[Char]) for Char in self.CharNames}
                tglat = np.atleast_3d(glat.ravel())
                tglon = np.atleast_3d(glon.ravel())

            Output["Ne"] = np.reshape(
                self._calcNe(
                    glat=tglat,
                    glon=tglon,
                    alt=talt,
                    **tOutput,
                ),
                Size["3DShape"],
            )

        if TEC:
            tecAlt = np.hstack(
                (
                    np.arange(70.0, 450.0, 5.0),
                    np.arange(450.0, 2000.0, 25.0),
                    np.arange(2000.0, 20000.0, 200.0),
                )
            )
            tecAlt = np.reshape(tecAlt, (1, 1, tecAlt.size))
            tOutput = {Char: np.atleast_3d(Output[Char]) for Char in self.CharNames}
            tglat = np.atleast_3d(glat.ravel())
            tglon = np.atleast_3d(glon.ravel())

            # tecAlt = xarray.DataArray(tecAlt, coords={"alt": tecAlt})
            M = self._calcNe(glat=tglat, glon=tglon, alt=tecAlt, **tOutput)
            Output["TEC"] = np.reshape(
                np.trapz(M, x=tecAlt) * 1e3 / 1e16, Size["2DShape"]
            )

        for Char in self.CharNames:
            Param = getattr(self, Char)
            if Param.ptype == "active":
                Output[Char] = np.reshape(Output[Char], Size["2DShape"])
            else:
                Output[Char] = np.reshape(Output[Char], Size["2DShape"][1:])

        # calculated separately so that bottomside effects can occur (B2bot)
        NmF1 = self._calcNe(glat=glat, glon=glon, alt=Output["hmF1"], **Output)
        NmE = self._calcNe(glat=glat, glon=glon, alt=Output["hmE"], **Output)

        chi, cchi = self.solzen(glat, glon)
        NmF1 = np.where(chi < 90.0, NmF1, np.nan)

        if self.Parameterization == "NeQuick":
            sNmF1 = Output["sNmF1"]
            sNmE = Output["sNmE"]
        elif self.Parameterization == "AIDA":
            sNmF1, sNmE = _Nm2sNm(
                Output["NmF2"],
                Output["hmF2"],
                Output["B2bot"],
                Output["NmF1"],
                Output["hmF1"],
                Output["B1bot"],
                Output["NmE"],
                Output["hmE"],
                Output["Betop"],
            )

        Output["NmF1"] = np.where(sNmF1 > 0.0, NmF1, np.nan)
        Output["NmE"] = np.where(sNmE > 0.0, NmE, np.nan)

        Output["foE"] = np.sqrt(Output["NmE"] / 0.124e11)

        Output["foF1"] = np.sqrt(Output["NmF1"] / 0.124e11)

        # only NmF2 needs to be rescaled
        Output["NmF2"] = 1e11 * Output["NmF2"]
        Output["NmF2"] = np.fmax(Output["NmF2"], 0.0)
        Output["foF2"] = np.sqrt(Output["NmF2"] / 0.124e11)

        if MUF3000:
            x = np.fmax(np.fmin(Output["foF2"] / Output["foE"], 1e6), 1.7)

            a = 1890 - 355 / (x - 1.4)
            b = (2.5 * x - 3) ** (-2.35) - 1.6
            MD = (Output["hmF2"] / a) ** (1.0 / b)

            Output["MUF3000"] = MD * Output["foF2"]

        return Output, Size

    ###########################################################################

    def calcNe(self, lat, lon, alt, grid="1D", particleIndex=None):
        """
        Calculates electron density for a given lat, lon, alt

        Output is always of the shape [NumParticles, :*]

        If possible, using grid = '2D' or '3D' is more efficient, as it can
        re-use parameters for every altitude with the same coordinates.

        Inputs
        ======

        lat(deg), lon(deg), alt(km)


        Inputs (Optional)
        =================

        grid:string
            '1D':lat, lon, alt must have the same size.
                Ne.shape() = [NumParticles, NumLat]

            '2D':lat, lon, must have the same size.
                Ne.shape() = [NumParticles, NumLat, NumAlt]

            '3D':lat, lon, alt can be any size
                Ne.shape() = [NumParticles, NumLat, NumLon, NumAlt]

        particleIndex:list
            Allows slicing the State, so only need to calculate the particles
            that we are interested in.
            By default, all particles are calculated

        """
        oldsettings = np.geterr()
        np.seterr(over="ignore", invalid="ignore")

        lat = np.array(lat, dtype=float)
        lon = np.array(lon, dtype=float)
        alt = np.array(alt, dtype=float)

        if grid == "1D":
            if (
                alt.shape != lat.shape
                or alt.shape != lon.shape
                or lat.shape != lon.shape
            ):
                raise (
                    ValueError(f"***{grid} grid: lat, lon, and alt must be same shape*")
                )

            lat = lat.flatten()
            lon = lon.flatten()
            alt = alt.flatten()

        if grid == "2D":
            if lat.shape != lon.shape:
                raise (ValueError(f"***{grid} grid: lat and lon must be same shape***"))

        if grid == "3D":
            lat, lon = np.meshgrid(lat, lon)

        maxParticles = self.maxParticles()

        if particleIndex is not None:
            particleIndex = np.array(particleIndex, dtype=int, ndmin=1)
            if np.any(particleIndex >= maxParticles):
                raise (IndexError(" particle index out of range."))

        if grid == "1D":
            Chars = self._calcValueIterator(
                lat.ravel(), lon.ravel(), particleIndex=particleIndex
            )
            Ne = self._calcNe(glat=lat, glon=lon, alt=alt, **dict(Chars))

        else:
            alt = np.reshape(alt, (1, 1, alt.size))
            Chars = self._calcValueIterator(
                lat.ravel(),
                lon.ravel(),
                particleIndex=particleIndex,
                ufunc=np.atleast_3d,
            )
            Ne = self._calcNe(
                glat=np.atleast_3d(lat.ravel()),
                glon=np.atleast_3d(lon.ravel()),
                alt=alt,
                **dict(Chars),
            )
        if grid == "3D":  # reshape array to get 3d output
            Ne = np.reshape(Ne, (Ne.shape[0], lon.shape[1], lon.shape[0], Ne.shape[2]))

        np.seterr(**oldsettings)
        # rescale Ne
        return Ne

    def _calcNe(self, **kwargs):
        arg_is_xarray = [isinstance(kwargs[key], xarray.DataArray) for key in kwargs]

        # if any(arg_is_xarray) and not all(arg_is_xarray):
        #    raise NotImplementedError("mixed xarray/numpy inputs not supported")
        if any(arg_is_xarray):
            if self.Parameterization == "NeQuick":
                Ne = xarray.apply_ufunc(
                    Ne_NeQuick,
                    kwargs["glat"],
                    kwargs["glon"],
                    kwargs["alt"],
                    kwargs["NmF2"],
                    kwargs["hmF2"],
                    kwargs["B2top"],
                    kwargs["B2bot"],
                    kwargs["sNmF1"],
                    kwargs["hmF1"],
                    kwargs["B1top"],
                    kwargs["B1bot"],
                    kwargs["sNmE"],
                    kwargs["hmE"],
                    kwargs["Betop"],
                    kwargs["Bebot"],
                    kwargs["Nmpt"],
                    kwargs["Hpt"],
                    kwargs["Nmpl"],
                    kwargs["Hpl"],
                )
            elif self.Parameterization == "AIDA":
                chi, _ = self.solzen(kwargs["glat"], kwargs["glon"])
                Ne = xarray.apply_ufunc(
                    Ne_AIDA,
                    kwargs["glat"],
                    kwargs["glon"],
                    kwargs["alt"],
                    kwargs["NmF2"],
                    kwargs["hmF2"],
                    kwargs["B2top"],
                    kwargs["B2bot"],
                    kwargs["NmF1"],
                    kwargs["hmF1"],
                    kwargs["B1top"],
                    kwargs["B1bot"],
                    kwargs["NmE"],
                    kwargs["hmE"],
                    kwargs["Betop"],
                    kwargs["Bebot"],
                    kwargs["Nmpt"],
                    kwargs["Hpt"],
                    kwargs["Nmpl"],
                    kwargs["Hpl"],
                    chi,
                )
            else:
                raise NotImplementedError("not implemented")
        else:
            if self.Parameterization == "NeQuick":
                Ne = self._calcNe_NeQuick(**kwargs)
            elif self.Parameterization == "AIDA":
                Ne = self._calcNe_AIDA(**kwargs)
            else:
                raise NotImplementedError("not implemented")

        return 1e11 * Ne

    ###########################################################################

    def calcValue(self, lat, lon, charList=None, particleIndex=None):
        """
        Calculates parameters for given lat, lon

        Output is a list of arrays corresponding to charList,
        each sized [NumParticles, NumLat]

        Inputs
        ======

        lat(deg), lon(deg)
            lat, lon must have the same size


        Inputs (Optional)
        =================

        charList:list (default is all)
            allows calculating specific parameters only

        particleIndex:list
            Allows slicing the State, so only need to calculate the particles
            that we are interested in.
            By default, all particles are calculated

        """

        lat = np.array(lat, dtype=float)
        lon = np.array(lon, dtype=float)

        if charList is None:
            charList = self.CharNames
        elif not isinstance(charList, list):
            charList = [charList]

        maxParticles = self.maxParticles()

        if particleIndex is not None:
            particleIndex = np.array(particleIndex, dtype=int, ndmin=1)
            if np.any(particleIndex >= maxParticles):
                raise (IndexError(" particle index out of range."))

        Basis = sph_harmonics(lat, lon, self.maxOrder())
        mdip = self.Modip.interp(lat, lon)
        MBasis = sph_harmonics(mdip, lon, self.maxOrder())

        return self._calcValueBasis(
            Basis, MBasis, charList=charList, particleIndex=particleIndex
        )

        # for Value in self._calcValueBasis(
        #    Basis, MBasis, charList=charList, particleIndex=particleIndex
        # ):
        #    yield Value

    ###########################################################################

    def _calcValueIterator(
        self, lat, lon, charList=None, particleIndex=None, ufunc=None
    ):
        if charList is None:
            charList = self.CharNames
        if ufunc is None:
            return zip(charList, self.calcValue(lat, lon, charList, particleIndex))
        else:
            return zip(
                charList,
                (ufunc(i) for i in self.calcValue(lat, lon, charList, particleIndex)),
            )

    def _calcValueBasisIterator(self, Basis, MBasis, charList=None, particleIndex=None):
        if charList is None:
            charList = self.CharNames
        return zip(
            charList, self._calcValueBasis(Basis, MBasis, charList, particleIndex)
        )

    ###########################################################################

    def _calcValueBasis(self, Basis, MBasis, charList=None, particleIndex=None):
        """
        Calculates parameters for given basis set

        Output is a list of arrays corresponding to charList,
        each sized [NumParticles, NumLat]

        Inputs
        ======

        lat(deg), lon(deg)
            lat, lon must have the same size


        Inputs (Optional)
        =================

        charList:list (default is all)
            allows calculating specific parameters only

        particleIndex:list
            Allows slicing the State, so only need to calculate the particles
            that we are interested in.
            By default, all particles are calculated

        """

        if charList is None:
            charList = self.CharNames
        elif not isinstance(charList, list):
            charList = [charList]

        if particleIndex is not None:
            maxParticles = self.maxParticles()
            particleIndex = np.array(particleIndex, dtype=int, ndmin=1)
            if np.any(particleIndex >= maxParticles):
                raise (IndexError(" particle index out of range."))

        for Char in charList:
            P = getattr(self, Char)

            if P.coords == "geo":
                B = Basis[0 : P.numDim, :]

            if P.coords == "modip":
                B = MBasis[0 : P.numDim, :]

            if P.ptype == "constant":
                B = np.array(B[0, :], ndmin=2)
                B = np.ones_like(B)

            if particleIndex is None or not P.ptype == "active":
                if P.scale == "abs":
                    Value = P.parameters @ B
                elif P.scale == "log":
                    Value = np.exp(P.parameters @ B)
                else:
                    raise (ValueError(f"Unrecognized parameter scale {P.scale}"))
            else:
                if P.scale == "abs":
                    Value = P.parameters[particleIndex, :] @ B
                elif P.scale == "log":
                    Value = np.exp(P.parameters[particleIndex, :] @ B)
                else:
                    raise (ValueError(f"Unrecognized parameter scale {P.scale}"))

            if "snm" in P.name.lower():
                # sNmF1 and sNmE
                yield np.fmax(Value, 0.0)
            elif "nm" in P.name.lower():
                # NmF2, NmF1, NmE, Nmpl, Nmpt
                yield np.fmax(Value, 1e-3)
            elif "hm" in P.name.lower():
                # hmF2, hmF1, hmE
                yield np.fmax(Value, 1.0)
            elif "b" in P.name.lower():
                # b2, b1, be
                yield np.fmax(Value, 1.0)
            elif "hp" in P.name.lower():
                # Hpl, Hpt
                yield np.fmax(Value, 1.0)
            else:
                yield Value

    ###########################################################################

    def expectation(self):
        # Create single-particle output state
        ModelState = copy.deepcopy(self)

        W = ModelState.Filter["Weight"]
        W = np.atleast_2d(W / np.sum(W))

        ModelState.Filter["Weight"] = np.atleast_1d(1.0)

        for i, Char in enumerate(ModelState.CharNames):
            Param = getattr(ModelState, Char)

            if Param.ptype == "active":
                Param.numParticles = 1
                for attr in Param._statesized:
                    value = getattr(Param, attr)
                    if value is not None:
                        value = np.atleast_2d(np.sum(W * value.T, axis=1))
                        setattr(Param, attr, value)

        return ModelState

    ###########################################################################

    def resample(
        self, N: int = None, return_i: bool = False, use_i: np.array = None
    ) -> AIDAState:
        """
        resample performs optimal resampling of the state

        Parameters
        ----------
        N : int, optional
            desired final size of ensemble, by default stays the same (None)
        return_i : bool, optional
            if True, returns (State, i) where i is the resampling index, by default False
        use_i : dp.array, optional
            if provided, uses this index to resample instead of the model weights, by default None
        Returns
        -------
        AIDAState
            resampled AIDA State object
        """
        ModelState = copy.deepcopy(self)

        if N is None:
            N = ModelState.maxParticles()

        Version = self.Version

        logger.debug(f" {Version}: " "set up weights")
        if "Weight" in ModelState.Filter:
            W = ModelState.Filter["Weight"]
        else:
            W = np.ones(ModelState.maxParticles())

        if np.any(np.isnan(W)):
            logger.warning(f" {Version}: " "NaN detected in incoming model weights")
            W = np.ones(W.shape)
            W = W / np.sum(W)

        W = W / np.sum(W)

        if use_i is None:
            ri = np.sum(
                (
                    np.random.uniform(high=1.0 / (N + 1), size=1)
                    + np.linspace(0, 1, num=N + 1)
                )
                > np.atleast_2d(np.cumsum(W)).T,
                axis=0,
            )[:-1]
        else:
            ri = use_i
            if N is not None and ri.size != N:
                raise ValueError(
                    f" requested size {N} and provided index of size {ri.size} do not match"
                )

        W = (1.0 / N) * np.ones(N)
        ModelState.Filter["Weight"] = W

        for Char in ModelState.CharNames:
            Param = getattr(ModelState, Char)

            if Param.ptype != "active":
                continue

            for attr in Param._statesized:
                value = getattr(Param, attr)
                if value is not None:
                    value = value[ri, :]
                    setattr(Param, attr, value)

            Param.numParticles = N
            setattr(ModelState, Char, Param)

        if return_i:
            return (ModelState, ri)
        else:
            return ModelState

    ###########################################################################

    def maxOrder(self):
        """
        returns maximum order of all parameters
        """
        maxOrder = 0
        for Char in self.CharNames:
            maxOrder = max(maxOrder, getattr(self, Char).order)
        return maxOrder

    ###########################################################################

    def maxParticles(self):
        """
        returns maximum number of particles in all parameters
        """
        maxParticles = 0
        for Char in self.CharNames:
            maxParticles = max(maxParticles, getattr(self, Char).numParticles)
        return maxParticles

    ###########################################################################

    def hasActive(self):
        """
        returns true if any parameters are active
        """
        for Char in self.CharNames:
            if getattr(self, Char).ptype == "active":
                return True
        else:
            return False

    ###########################################################################

    def solzen(self, glat, glon):
        """


        Parameters
        ----------
        glat : TYPE
            DESCRIPTION.
        glon : TYPE
            DESCRIPTION.

        Returns
        =======
        chi : ndarray
            Solar zenith angle
        cchi : ndarray
            cosine of the solar zenith angle
        cch_eff : ndarray
            Effective solar zenith angle

        """
        xlt = np.mod((self._UT + glon / 15.0 + 24.0), 24.0)

        cchi = np.sin(np.deg2rad(glat)) * self._sdelta - np.cos(
            np.deg2rad(glat)
        ) * self._cdelta * np.cos(np.pi * xlt / 12.0)

        chi = np.rad2deg(np.arctan2(np.sqrt(1.0 - cchi * cchi), cchi))

        return chi, cchi

    def chi_eff(self, chi):
        chi0 = 86.23292796211615

        # NeQuick 2 and G pg 71
        return djoin(90.0 - 0.24 * np.exp(20.0 - 0.2 * chi), chi, 12.0, chi - chi0)

    ##########################################################################

    def _calcNe_NeQuick(
        self,
        glat,
        glon,
        alt,
        NmF2,
        hmF2,
        B2top,
        B2bot,
        sNmF1,
        hmF1,
        B1top,
        B1bot,
        sNmE,
        hmE,
        Betop,
        Bebot,
        Nmpt,
        Hpt,
        Nmpl,
        Hpl,
        **kwargs,
    ):
        return Ne_NeQuick(
            glat,
            glon,
            alt,
            NmF2,
            hmF2,
            B2top,
            B2bot,
            sNmF1,
            hmF1,
            B1top,
            B1bot,
            sNmE,
            hmE,
            Betop,
            Bebot,
            Nmpt,
            Hpt,
            Nmpl,
            Hpl,
        )

    def _calcNe_AIDA(
        self,
        glat,
        glon,
        alt,
        NmF2,
        hmF2,
        B2top,
        B2bot,
        NmF1,
        hmF1,
        B1top,
        B1bot,
        NmE,
        hmE,
        Betop,
        Bebot,
        Nmpt,
        Hpt,
        Nmpl,
        Hpl,
        **kwargs,
    ):
        chi, cchi = self.solzen(glat, glon)

        return Ne_AIDA(
            glat,
            glon,
            alt,
            NmF2,
            hmF2,
            B2top,
            B2bot,
            NmF1,
            hmF1,
            B1top,
            B1bot,
            NmE,
            hmE,
            Betop,
            Bebot,
            Nmpt,
            Hpt,
            Nmpl,
            Hpl,
            chi,
        )

    ###########################################################################

    def Config(self):
        """

        Returns
        -------
        dictionary containing the configuration

        """
        Exclude = set(("parameters", "velocity"))
        Config = dict(
            zip(
                self.CharNames,
                (
                    {
                        k: getattr(getattr(self, Char), k)
                        for k in dir(getattr(self, Char))
                        if k[0] != "_" and k not in Exclude
                    }
                    for Char in self.CharNames
                ),
            )
        )

        return Config


###############################################################################


def djoin(f1, f2, alpha, x):
    ee = np.exp(alpha * x)

    return (f1 * ee + f2) / (ee + 1.0)
