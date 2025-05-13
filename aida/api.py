#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import configparser
from pathlib import Path
from importlib import resources
import traceback
import numpy as np
import datetime
import xarray
import requests
import os

from .logger import AIDAlogger
from .time import npdt2dt


logger = AIDAlogger(__name__)

###############################################################################


def api_config(filename: str | Path = None):
    """Read a AIDA input file (in .ini format)

    This function reads a (well-formmated) AIDA input file and adds all the
    parameters to a dictionary which is carried around and used by the model

    Parameters
    ==========
    filename : string
        Full path for the AIDA input file (usually AIDA.inp)


    Returns
    =======
    inputs : dictionary
        Dictionary containing the AIDA input paramters


    .. todo:: None

    |"""

    # default config file
    if filename is None:
        filename = resources.files("aida").joinpath("api_config.ini")

    if not isinstance(filename, Path):
        filename = Path(filename)

    # Check filename exists
    if not filename.exists():
        raise FileNotFoundError(f"ERROR: file {filename.expanduser()} not found")

    config = configparser.ConfigParser(delimiters=(";", "="), strict=True)

    try:
        config.read(filename.expanduser())
    except Exception:
        logger.error(traceback.format_exc())

    config_struct = {}
    for section in config.sections():
        config_struct[section] = {}

        for option in config.options(section):
            value = config.get(section, option)

            if value.lower() == "none":
                value = None

            config_struct[section][option] = value

    if np.sum(int(config['api']['token'])) == 0:
        raise APIConfigurationError(f" invalid token in file {filename.expanduser()}, check configuration file and edit if needed.")

    if config['cache']['folder'] == '/path/to/cache/':
        raise APIConfigurationError(f" invalid cache in file {filename.expanduser()}, check configuration file and edit if needed.")
    elif not Path(config['cache']['folder']).exists():
        raise APIConfigurationError(f" invalid cache in file {filename.expanduser()}, check configuration file and edit if needed.")

    return config_struct
###############################################################################


def createFilename(pattern, time=None, **kwargs):
    """


    Parameters
    ----------
    pattern : string, mandatory
        string containing tokens for date/time formatting

    time : datetime, optional
        list of times to use

    **kwargs : custom tags, optional
        allows user-defined tags as named arguments
        {tag} must match the name of the input argument
        tags must all have the same size, tags with a single element will be
        broadcast.

    default tags :
        {yyyy} : 4 digit, zero-padded year
        {yy}   : 2 digit, zero-padded year
        {mm}   : 2 digit, zero-padded month
        {doy}  : 3 digit, zero-padded day of year
        {dd}   : 2 digit, zero-padded day of month
        {HH}   : 2 digit, zero-padded hour
        {H}    : 1 character, alphanumeric hour (RINEX v2)
        {MM}   : 2 digit, zero-padded minute
        {SS}   : 2 digit, zero-padded second
        {GPSW} : 4 digit, zero-padded GPS week number
        {GPSD} : 1 digit day of GPS week

    e.g. pattern = ~/data/{yyyy}/{doy}/{name}{doy}{H}{MM}.{yy}{type}.Z
            time = [datetime(2022,2,1),datetime(2022,2,1,3,15,0)]
            name = ['gilc', 'abmf']
            type = 'd'

        ['~/data/2022/032/gilc032a00.22d.Z',
         '~/data/2022/032/abmf032a00.22d.Z',
         '~/data/2022/032/gilc032d15.22d.Z',
         '~/data/2022/032/abmf032d15.22d.Z']

    you can also add format instructions in the string.format()
    by default this function adds leading zeros
    e.g. ~/data/{yyyy}/{doy:d}/{name}{doy}{H}{MM}.{yy}d.Z
         ~/data/2022/32/gilc032c15.22d.Z

    RINEX v2:
        daily:  {name}{doy}0.{yy}d
        hourly: {name}{doy}{H}.{yy}d
        15-min: {name}{doy}{H}{MM}.{yy}d

    RINEX v3:
        {name}00{country}_R_{yyyy}{doy}{HH}{MM}_01H_01S_MO.rnx
        BRDC00WRD_S_{yyyy}{doy}{HH}{MM}_01D_MN.rnx.gz



    Returns
    -------
    list of filenames expanded along date and names

    """

    if isinstance(pattern, list):
        output = []
        for p in pattern:
            output += createFilename(p, time=time, **kwargs)
        return output

    for key in kwargs:  # make sure keyword tags are in an iterable form
        if isinstance(kwargs[key], str):
            kwargs[key] = [kwargs[key]]
        elif not hasattr(kwargs[key], "__iter__"):
            kwargs[key] = [kwargs[key]]

    # check if keywords have broadcastable sizes
    arg_length = [len(i) for i in kwargs.values()]

    if len(arg_length) == 0:
        arg_length = [0]

    invalid_args = [
        k for k in kwargs if len(
            kwargs[k]) != 1 and len(
            kwargs[k]) != max(arg_length)]

    if any(invalid_args):
        raise ValueError(
            f"Unable to broadcast tags {invalid_args} "
            + f"to size {max(arg_length)}")

    # build array of dicts
    name = [
        {k: kwargs[k][min(i, len(kwargs[k]) - 1)] for k in kwargs}
        for i in range(max(arg_length))
    ]

    if len(name) == 0:
        name = [{}]

    # modify pattern to include zero-padding as a default
    pattern = (
        pattern.replace("{yyyy}", "{yyyy:04d}")
        .replace("{yy}", "{yy:02d}")
        .replace("{mm}", "{mm:02d}")
        .replace("{doy}", "{doy:03d}")
        .replace("{dd}", "{dd:02d}")
        .replace("{HH}", "{HH:02d}")
        .replace("{MM}", "{MM:02d}")
        .replace("{SS}", "{SS:02d}")
        .replace("{GPSW}", "{GPSW:04d}")
    )

    # make sure time is iterable
    if not hasattr(time, "__iter__"):
        time = [time]
    elif isinstance(time, xarray.DataArray):
        time = np.atleast_1d(time)

    # make generator for time tags
    time_data = (_date_dict(t, kwargs.keys()) for t in time)

    # t.update overwrites dict keys
    try:
        return [
            pattern.format_map(t)
            for t in time_data
            for n in name
            if t.update(n) is None
        ]

    except KeyError as error:
        error.args = (
            f"User-defined tag {error} was not provided as an "
            + "argument",
        )
        raise error
###############################################################################


def _date_dict(time, keys):
    """
    helper function for create_filename
    """

    if isinstance(time, np.datetime64):
        time = npdt2dt(time)
    elif time is not None:
        time = time.replace(tzinfo=None)

    if time is not None:
        chr_hour = "abcdefghijklmnopqrstuvwx"
        date_dict = {
            "yyyy": time.year,
            "yy": np.mod(time.year, 100),
            "mm": time.month,
            "dd": time.day,
            "doy": (time - datetime.datetime(time.year, 1, 1)).days + 1,
            "H": chr_hour[time.hour],
            "HH": time.hour,
            "MM": time.minute,
            "SS": time.second,
            "GPSW": int((time - datetime.datetime(1980, 1, 6)).days / 7),
            "GPSD": np.mod((time - datetime.datetime(1980, 1, 6)).days, 7),
        }
    else:
        date_dict = dict()

    for key in keys:
        date_dict[key] = None

    return date_dict

###############################################################################


def downloadOutput(
        config: Path | dict,
        time: np.datetime64,
        model: str) -> Path:

    if not isinstance(config, dict):
        config = api_config(config)

    if model == "ultra":
        model_tag = "u"
    elif model == "rapid":
        model_tag = "r"
    elif model == "daily" or model == "final":
        model_tag = "d"
        model = "final"
    else:
        raise ValueError(f" unrecognized model {model}")

    cacheFolder = Path(config['cache']['folder'])

    if not cacheFolder.exists():
        raise FileNotFoundError(
            ' attempted to write to a cache folder which does not exist. Check that api_config.ini is configured correctly.'
        )

    outputFile = cacheFolder.joinpath(
        createFilename(
            config['cache']['subfolder'],
            time)[0]).joinpath(
        createFilename(
            "output_{model}_{yy}{mm}{dd}_{HH}{MM}{SS}.h5",
            time,
            model=model_tag)[0])

    if outputFile.exists():
        return outputFile

    if not outputFile.parent.exists():
        os.makedirs(outputFile.parent.expanduser())

    payload = {
        "file_time": time.astype('str'),
        "product": model,
        "file_type": "raw"}

    response = requests.get(
        "https://spaceweather.bham.ac.uk/api/download-output/",
        headers={
            'Authorization': f"Token {config['api']['token']}"},
        data=payload)

    if response.status_code == 200:
        with open(outputFile, 'wb') as f:
            f.write(response.content)
        return outputFile
    else:
        raise ValueError(f"{response.status_code} {response.text}")


###############################################################################


class APIConfigurationError(Exception):
    def __init__(self, message=" problem in the API config file"):
        self.message = message
        super().__init__(self.message)
