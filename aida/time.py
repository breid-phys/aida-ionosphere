#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:49:17 2022

@author: ben
"""

import datetime
import numpy as np
import astropy.time


###############################################################################


def dt2epoch(time):
    """
    datetime to epoch
    """
    epoch = (
        time.replace(tzinfo=datetime.timezone.utc)
        - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    ).total_seconds()
    epoch = epoch + time.microsecond / 1e6
    return epoch


###############################################################################


def epoch2dt(epoch):
    """
    epoch to datetime
    """
    time = datetime.datetime(
        1970, 1, 1, tzinfo=datetime.timezone.utc
    ) + datetime.timedelta(seconds=epoch)
    return time


###############################################################################


def npdt2epoch(time):
    """
    np.datetime64 to epoch
    """
    epoch = (time - np.datetime64("1970-01-01")) / np.timedelta64(1, "ns")
    return epoch * 1e-9


###############################################################################


def epoch2npdt(epoch):
    """
    epoch to np.datetime64
    """
    time = epoch * np.timedelta64(1, "s") + np.datetime64("1970-01-01", "ns")
    return time


###############################################################################


def dt2npdt(time):
    """
    datetime to np.datetime64
    """
    return np.datetime64(time.isoformat()).astype("datetime64[ns]")


###############################################################################


def npdt2dt(time):
    """
    np.datetime64 to datetime
    """
    return datetime.datetime.fromisoformat(time.astype("datetime64[ms]").astype("str"))


###############################################################################


def npdt2gps(time):
    """
    np.datetime64 to gps epoch
    """
    return astropy.time.Time(astropy.time.Time(time, scale="utc"), format="gps").value


###############################################################################


def gps2npdt(epoch):
    """
    gps epoch to np.datetime64
    """

    return astropy.time.Time(
        astropy.time.Time(epoch, format="gps"), format="datetime64", scale="utc"
    ).value


###############################################################################
