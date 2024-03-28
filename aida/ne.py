#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:13:55 2022

@author: ben
"""

import math
import traceback

import numpy as np
from scipy.special import expit

REm = 6371211.266  # Earth's mean radius in meter
REkm = REm / 1000  # km
Lpp = 5  # position of the plasmapause measured in units of Earth radius
N_ms = 1.0e-4  # in 10^11 el/m^3, plasmapause density

try:
    from numba import float64, guvectorize, vectorize, njit

    hasnumba = True
except BaseException:
    hasnumba = False
# hasnumba = False
target = "cpu"

###############################################################################


@njit([float64(float64)], nogil=True, fastmath=True, error_model="numpy")
def _raw_sech2(x):
    # pure python version
    y = 1.0 / math.cosh(x)
    return y * y


try:  # def sech2()
    if not hasnumba:
        raise Exception(" Numba compilation not available or disabled.")

    @vectorize([float64(float64)], nopython=True, target=target, fastmath=True)
    def sech2(x):
        return _raw_sech2(x)

except Exception:
    print(traceback.format_exc())

    def sech2(x):
        y = np.array(1.0) / np.cosh(x)
        return y * y


###############################################################################


@njit([float64(float64)], nogil=True, fastmath=True, error_model="numpy")
def _raw_asech(x):
    # pure python version
    ix = 1.0 / x
    return math.log(ix + math.sqrt(ix * ix - 1.0))


try:  # def asech()
    if not hasnumba:
        raise Exception(" Numba compilation not available or disabled.")

    @vectorize([float64(float64)], nopython=True, target=target, fastmath=True)
    def asech(x):
        return _raw_asech(x)

except Exception:
    print(traceback.format_exc())

    def asech(x):
        ix = 1.0 / x
        return np.log(ix + np.sqrt(ix * ix - 1.0))


###############################################################################


@njit([float64(float64, float64)], nogil=True, fastmath=True, error_model="numpy")
def _raw_geoLat2magLat(lat_rad, lon_deg):
    """
    geodetic latitude to geomagnetic latitude conversion using Dipole Approx
    Syntax:
        latm_rad = geoLat2magLat(lat_rad, lon_deg)
    Description
        lat_rad = geographic latitude Vector in radians
        lon_deg = geographic longitude Vector in degrees
        latm_rad = geomagnetic latitude Vector in radians
    """
    # latitude of geomagnetic North Pole used
    phi_GNP = math.radians(79.74)
    # longitude of geomagnetic North Pole use
    lambda_GNP = math.radians(-71.78)
    return math.asin(
        math.sin(lat_rad) * math.sin(phi_GNP)
        + math.cos(lat_rad)
        * math.cos(phi_GNP)
        * math.cos(math.radians(lon_deg) - lambda_GNP)
    )  # in radians


try:
    if not hasnumba:
        raise Exception(" Numba compilation not available or disabled.")

    @vectorize(
        [float64(float64, float64)],
        nopython=True,
        target=target,
        fastmath=True,
    )
    def geoLat2magLat(lat_rad, lon_deg):
        return _raw_geoLat2magLat(lat_rad, lon_deg)

except Exception:
    print(traceback.format_exc())

    def geoLat2magLat(lat_rad, lon_deg):
        phi_GNP = np.deg2rad(79.74)  # latitude of geomagnetic North Pole used
        # longitude of geomagnetic North Pole use
        lambda_GNP = np.deg2rad(-71.78)
        return np.arcsin(
            np.sin(lat_rad) * np.sin(phi_GNP)
            + np.cos(lat_rad)
            * np.cos(phi_GNP)
            * np.cos(np.deg2rad(lon_deg) - lambda_GNP)
        )  # in radians


###############################################################################
try:
    if not hasnumba:
        raise Exception(" Numba compilation not available or disabled.")
    """
      function to calculate lMax orders of spherical harmonics at given lat/lon
      lat, lon in degrees
      lat and lon must have the same size
      output will have size [(lMax+1)^2, numel(lat/lon)]
      harmonics are ordered:
          l=0 m=0
          l=1 m=0
          l=1 l=1
          l=1 l=-1
          l=2 m=0
          ...
          l=lMax m=lMax
          l=lMax m=-lMax
      """

    def sph_harmonics(lat, lon, lMax):
        return np.atleast_2d(
            gusph_harmonics(
                np.array(lat, dtype="float64").ravel(),
                np.array(lon, dtype="float64").ravel(),
                np.array(lMax, dtype="int64"),
                np.zeros((lMax + 1) ** 2, dtype=float),
            )
        ).T

    @guvectorize(
        ["void(float64, float64, int64, float64[:], float64[:])"],
        "(),(),(),(n)->(n)",
        nopython=True,
        target=target,
        fastmath=True,
    )
    def gusph_harmonics(lat, lon, lMax, zero, output):
        lat = math.radians(lat)
        lon = math.radians(lon)

        coLat = 0.5 * math.pi - lat

        x = math.cos(coLat)
        y = math.sin(coLat)

        output[0] = math.sqrt(0.5)

        # calculate legendre functions
        for L in range(1, lMax + 1):
            m = L
            n = L**2 + 2 * m
            nm = (L - 1) ** 2 + 2 * (m - 1)

            output[n] = math.sqrt(1.0 + (0.5 / m)) * y * output[nm]
            output[n - 1] = output[n]

            output[n - 2] = np.sqrt(2 * (m - 1) + 3) * x * output[nm]

            if L > 1:
                output[n - 3] = output[n - 2]

            for m in range(0, (L - 2) + 1):
                a = math.sqrt((4 * (L**2) - 1) / (L**2 - m**2))

                b = -math.sqrt(((L - 1) ** 2 - m**2) / (4 * (L - 1) ** 2 - 1))

                n = L**2 + 2 * m
                n1 = (L - 1) ** 2 + 2 * m
                n2 = (L - 2) ** 2 + 2 * m
                output[n] = a * (x * output[n1] + b * output[n2])

                if m > 0:
                    output[n - 1] = output[n]

        # fast angular components
        C1 = math.cos(lon)
        C0 = 2.0 * C1
        C2 = 1.0

        S1 = math.sin(lon)
        S2 = 0.0

        S = S1
        C = C1

        for m in range(1, lMax + 1):
            if m > 1:
                if np.mod(m, 2) == 1:
                    S2 = S
                    C2 = C

                    C = C0 * C - C1
                    S = C0 * S - S1
                else:
                    S1 = S
                    C1 = C

                    C = C0 * C - C2
                    S = C0 * S - S2

            for L in range(m, lMax + 1):
                n = L**2 + 2 * m - 1
                output[n] = C * output[n]
                n = L**2 + 2 * m + 0
                output[n] = S * output[n]

except Exception:
    print(traceback.format_exc())

    def sph_harmonics(lat, lon, lMax=12):
        lMax = int(lMax)
        N = int((lMax + 1) ** 2)

        lat = np.array(lat, dtype=float).ravel()
        lon = np.array(lon, dtype=float).ravel()

        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        if not lat.shape == lon.shape:
            raise Exception("lat and lon inputs must have the same shape")

        coLat = 0.5 * np.pi - lat

        output = np.zeros((N, np.size(lat)), dtype=float)

        x = np.cos(coLat)
        y = np.sin(coLat)

        output[0, :] = np.sqrt(0.5)

        # calculate legendre functions
        for L in range(1, lMax + 1):
            m = L
            n = L**2 + 2 * m
            nm = (L - 1) ** 2 + 2 * (m - 1)

            output[n, :] = np.sqrt(1.0 + (0.5 / m)) * y * output[nm, :]
            output[n - 1, :] = output[n, :]

            output[n - 2, :] = np.sqrt(2 * (m - 1) + 3) * x * output[nm, :]

            if L > 1:
                output[n - 3, :] = output[n - 2, :]

            for m in range(0, (L - 2) + 1):
                a = np.sqrt((4 * (L**2) - 1) / (L**2 - m**2))

                b = -np.sqrt(((L - 1) ** 2 - m**2) / (4 * (L - 1) ** 2 - 1))

                n = L**2 + 2 * m
                n1 = (L - 1) ** 2 + 2 * m
                n2 = (L - 2) ** 2 + 2 * m
                output[n, :] = a * (x * output[n1, :] + b * output[n2, :])

                if m > 0:
                    output[n - 1, :] = output[n, :]

        # fast angular components
        C1 = np.cos(lon)
        C0 = 2.0 * C1
        C2 = np.ones(C1.shape)

        S1 = np.sin(lon)
        S2 = np.zeros(S1.shape)

        S = S1
        C = C1

        for m in range(1, lMax + 1):
            if m > 1:
                if np.mod(m, 2) == 1:
                    S2 = S
                    C2 = C

                    C = C0 * C - C1
                    S = C0 * S - S1
                else:
                    S1 = S
                    C1 = C

                    C = C0 * C - C2
                    S = C0 * S - S2

            for L in range(m, lMax + 1):
                n = L**2 + 2 * m - 1
                output[n, :] = C * output[n, :]
                n = L**2 + 2 * m + 0
                output[n, :] = S * output[n, :]

        return output


###############################################################################
try:
    if not hasnumba:
        raise Exception(" Numba compilation not available or disabled.")

    def Ne_NeQuick(
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
    ):
        return guNe_NeQuick(
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

    @vectorize(
        [
            float64(
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
                float64,
            )
        ],
        nopython=True,
        target=target,
        fastmath=True,
    )
    def guNe_NeQuick(
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
    ):
        k2 = alt - hmF2

        # NmF2
        if k2 > 0.0:  # topside
            r = (max(Hpl, 2 * B2top) - B2top) / B2top

            # plasmasphere
            lat_rad = math.radians(glat)
            cmlat = math.cos(_raw_geoLat2magLat(lat_rad, glon))
            cmlat = REkm * cmlat * cmlat

            hg = max(Lpp * cmlat - REkm, 5e3)

            Nmin = min(Nmpl, NmF2)
            Ag = asech(math.sqrt((Nmin / NmF2) * math.exp(-hg / Hpl)))
            Hg = -(hmF2 - hg) / (2.0 * Ag)

            g = max(
                ((Hg - B2top) * r * B2top) / ((hg - hmF2) * (B2top * r - Hg + B2top)),
                0.01,
            )

            Lval = (alt + REkm) / cmlat
            Rpp = Lpp * cmlat
            F_pp = math.atan((Rpp - REkm - alt) * 0.002) / math.pi + 0.5
            F_ip = 2 * math.atan(1e-2 * k2) / math.pi

            Ne_tp = Nmpt * F_ip * F_pp * (math.exp(-REkm * (Lval - 1.0) / Hpt)) + N_ms

            H2 = B2top * (1.0 + (r * g * k2) / (r * B2top + g * k2))
            NeF2 = NmF2 * _raw_sech2(0.5 * k2 / H2)

            Ne = NeF2 + Ne_tp

        else:  # bottomside
            k1 = alt - hmF1
            ke = alt - hmE

            CutoffDRegion = 0.5 * (1.0 + math.tanh(0.3333 * (alt - 102.0)))
            CutoffBottomside = math.exp(10.0 / (1.0 + math.fabs(k2)))
            H2 = B2bot
            NeF2 = NmF2 * _raw_sech2(0.5 * k2 / H2)

            # NmF1 and NmE
            if k1 > 0.0:
                H1 = B1top
                HE = Betop
            else:
                H1 = B1bot
                if ke > 0.0:
                    HE = Betop
                else:
                    HE = Bebot

            NeF1 = sNmF1 * _raw_sech2(0.5 * CutoffBottomside * k1 / H1)
            NeE = sNmE * _raw_sech2(0.5 * CutoffBottomside * ke / HE)

            Ne = CutoffDRegion * (NeF2 + NeF1 + NeE)

        return max(Ne, 0.0)

except Exception:
    print(traceback.format_exc())

    def Ne_NeQuick(
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
    ):
        k2 = alt - hmF2
        k1 = alt - hmF1
        ke = alt - hmE

        CutoffBottomside = np.exp(10.0 / (1.0 + np.abs(k2)))

        # plasmasphere
        mlat = geoLat2magLat(np.deg2rad(glat), glon)

        cmlat = np.cos(mlat)
        cmlat = REkm * cmlat * cmlat

        r = (np.fmax(Hpl, 2 * B2top) - B2top) / B2top
        hg = np.fmax(Lpp * cmlat - REkm, 5e3)
        Ag = asech(np.sqrt((np.fmin(Nmpl, NmF2) / NmF2) * np.exp(-hg / Hpl)))
        Hg = -(hmF2 - hg) / (2.0 * Ag)
        g = np.fmax(
            ((Hg - B2top) * r * B2top) / ((hg - hmF2) * (B2top * r - Hg + B2top)), 0.01
        )

        Lval = (alt + REkm) / cmlat
        Rpp = Lpp * cmlat
        F_pp = np.arctan((Rpp - REkm - alt) * 0.002) / np.pi + 0.5
        F_ip = 2 * np.arctan(1e-2 * k2) / np.pi

        Ne_tp = Nmpt * F_ip * F_pp * (np.exp(-REkm * (Lval - 1.0) / Hpt)) + N_ms

        # NmF2
        H2 = np.where(
            k2 < 0.0, B2bot, (B2top * (1.0 + (r * g * k2) / (r * B2top + g * k2)))
        )

        NeF2 = NmF2 * sech2(0.5 * k2 / H2)

        # NmF1
        H1 = np.where(k1 < 0.0, B1bot, B1top)

        NeF1 = sNmF1 * sech2(0.5 * CutoffBottomside * k1 / H1)

        # NmE
        HE = np.where(ke < 0.0, Bebot, Betop)

        NeE = sNmE * sech2(0.5 * CutoffBottomside * ke / HE)

        return np.fmax(np.where(k2 < 0.0, NeF2 + NeF1 + NeE, NeF2 + Ne_tp), 0.0)


###############################################################################


def Ne_AIDA(
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
):
    sNmF1, sNmE = _Nm2sNm(NmF2, hmF2, B2bot, NmF1, hmF1, B1bot, NmE, hmE, Betop)
    # sNmF1 = np.where(chi < 90.0, sNmF1, 0.0)
    scut = expit(90.0 - chi)
    sNmF1 = sNmF1 * scut
    
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


###############################################################################


try:
    if not hasnumba:
        raise Exception()

    @guvectorize(
        [
            "void(float64, float64, float64,"
            " float64, float64, float64,"
            " float64, float64, float64,"
            " float64[:], float64[:])"
        ],
        ("(),(),()," "(),(),()," "(),(),()" "->(),()"),
        nopython=True,
        target=target,
        fastmath=True,
    )
    def _Nm2sNm(NmF2, hmF2, B2bot, NmF1, hmF1, B1bot, NmE, hmE, Betop, sNmF1, sNmE):
        AF2_F1 = NmF2 * _raw_sech2((0.5 * (hmF2 - hmF1) / B2bot))
        AF2_E = NmF2 * _raw_sech2((0.5 * (hmF2 - hmE) / B2bot))

        AF1_E = 1.0 * _raw_sech2((0.5 * (hmF1 - hmE) / B1bot))
        AE_F1 = 1.0 * _raw_sech2((0.5 * (hmE - hmF1) / Betop))

        sNmF1[0] = max(NmF1 - AF2_F1, 0.0)
        sNmE[0] = max(NmE - AF2_E - sNmF1[0] * AF1_E, 0.0)
        sNmF1[0] = max(NmF1 - AF2_F1 - sNmE[0] * AE_F1, 0.0)
        sNmE[0] = max(NmE - AF2_E - sNmF1[0] * AF1_E, 0.0)
        sNmF1[0] = max(NmF1 - AF2_F1 - sNmE[0] * AE_F1, 0.0)
        sNmE[0] = max(NmE - AF2_E - sNmF1[0] * AF1_E, 0.0)

except Exception:
    print(traceback.format_exc())

    def _Nm2sNm(NmF2, hmF2, B2bot, NmF1, hmF1, B1bot, NmE, hmE, Betop):
        AF2_F1 = NmF2 * sech2(0.5 * (hmF2 - hmF1) / B2bot)
        AF2_E = NmF2 * sech2(0.5 * (hmF2 - hmE) / B2bot)

        AF1_E = sech2(0.5 * (hmF1 - hmE) / B1bot)
        AE_F1 = sech2(0.5 * (hmE - hmF1) / Betop)

        sNmF1 = np.fmax(NmF1 - AF2_F1, 0.0)
        sNmE = np.fmax(NmE - AF2_E - sNmF1 * AF1_E, 0.0)
        sNmF1 = np.fmax(NmF1 - AF2_F1 - sNmE * AE_F1, 0.0)
        sNmE = np.fmax(NmE - AF2_E - sNmF1 * AF1_E, 0.0)
        sNmF1 = np.fmax(NmF1 - AF2_F1 - sNmE * AE_F1, 0.0)
        sNmE = np.fmax(NmE - AF2_E - sNmF1 * AF1_E, 0.0)

        return sNmF1, sNmE
