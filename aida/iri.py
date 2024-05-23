import math

from numba import njit

target = "cpu"


###############################################################################
@njit(nogil=True, fastmath=True, error_model="numpy")
def _season(doy, glat):
    seasn = int((doy + 45.0) / 92.0)
    if glat < 0.0:
        seasn = seasn - 2
    return ((seasn - 1) % 4) + 1


###############################################################################


@njit(nogil=True, fastmath=True, error_model="numpy")
def _eps_0(x, sc, hx):
    """
    IRI epstein 0
    """
    ARGMAX = 87.3
    D1 = (x - hx) / sc
    if math.fabs(D1) < ARGMAX:
        return 1 / (1 + math.exp(-D1))
    else:
        return math.copysign(1.0, D1)


###############################################################################


@njit(nogil=True, fastmath=True, error_model="numpy")
def _eptr(x, sc, hx):
    """
    IRI epstein 0
    """
    ARGMAX = 87.3
    D1 = (x - hx) / sc
    if math.fabs(D1) < ARGMAX:
        return math.log(1 + math.exp(D1))
    else:
        return max(0.0, D1)


###############################################################################


@njit(nogil=True, fastmath=True, error_model="numpy")
def _hpol(t, YD, YN, SR, SS, DSA, DSU):
    """
    IRI hpol function
    """
    if math.fabs(SS) > 25.0:
        if SS > 0.0:
            return YD
        else:
            return YN

    return YN + (YD - YN) * _eps_0(t, DSA, SR) + (YN - YD) * _eps_0(t, DSU, SS)


###############################################################################


@njit(
    "float64(float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _xe2(x, B1):
    return math.exp(-(math.pow(x, B1))) / math.cosh(x)


###############################################################################


@njit(
    "float64(float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _dxe2(x, B1):
    return -_xe2(x, B1) * (math.tanh(x) + B1 * math.pow(x, B1 - 1.0))


###############################################################################


@njit(nogil=True, fastmath=True, error_model="numpy")
def _h_star(hmF1, C1, h):
    """
    IRI h*
    """
    return hmF1 * (1.0 - ((hmF1 - h) / hmF1) ** (1.0 + C1))


###############################################################################
@njit(
    "float64(float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _newton(A, B1):
    """ """

    # this is the tolerance used in all IRI
    tol = 0.01

    # first guess
    x = (-math.log(A)) ** (1.0 / B1)
    x = (-math.log(2.0 * A - _xe2(x, B1))) ** (1.0 / B1)

    f = _xe2(x, B1) - A
    df = _dxe2(x, B1)

    for i in range(1000):
        dx = f / df
        x = x - dx

        if math.fabs(dx) < tol:
            break

        f = _xe2(x, B1) - A
        df = _dxe2(x, B1)

    return x


###############################################################################


@njit(
    "float64(float64, float64, float64, float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _newton_hmF1(NmF2, hmF2, B0, B1, NmF1, NmE, hmE):
    """
    _regfal_hmF1 Regula Falsi (internal use only)

    Parameters
    ----------
    NmF2 : _type_
        _description_
    NmF1 : _type_
        _description_
    hmF2 : _type_
        _description_
    hmE : _type_
        _description_
    B0 : _type_
        _description_
    B1 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    if 0.9 * NmF1 < NmE:
        # no F1 layer
        return 0.0

    A = NmF1 / NmF2

    h = hmF2 - _newton(A, B1) * B0

    if h < hmE:
        return 0.0
    else:
        return h


@njit(
    "float64(float64, float64, float64, float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _newton_hst_F1(NmF2, hmF2, B0, B1, hmF1, C1, NmE):
    """
    _regfal_hst Regula Falsi (internal use only)
    F1 layer present

    """

    A = NmE / NmF2

    hs3 = hmF2 - _newton(A, B1) * B0

    h = hmF1 - hmF1 * (1.0 - hs3 / hmF1) ** (1.0 / (1.0 + C1))

    if h >= 100.0:
        return h
    else:
        return 0.0


@njit(
    "float64(float64, float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _newton_hst(NmF2, hmF2, B0, B1, NmE):
    """
    _regfal_hst Regula Falsi (internal use only)

    """

    A = NmE / NmF2

    return hmF2 - _newton(A, B1) * B0


@njit(
    "UniTuple(float64, 8)(float64, float64, float64, float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _d_region(modip, hour, SAX80, SUX80, NmE, hmE, NmD):

    amodip = math.fabs(modip)

    if amodip >= 18.0:
        DELA = 1.0 + math.exp(-(amodip - 30.0) / 10.0)
    else:
        DELA = 4.32

    hmD = _hpol(hour, 81.0, 88.0, SAX80, SUX80, 1.0, 1.0)

    F1 = _hpol(
        hour,
        0.02 + 0.03 / DELA,
        0.05,
        SAX80,
        SUX80,
        1.0,
        1.0,
    )
    F2 = _hpol(hour, 4.6, 4.5, SAX80, SUX80, 1.0, 1.0)
    F3 = _hpol(hour, -11.5, -4.0, SAX80, SUX80, 1.0, 1.0)

    FP1 = F1
    FP2 = -FP1 * FP1 / 2.0
    FP30 = (-F2 * FP2 - FP1 + 1.0 / F2) / (F2 * F2)
    FP3U = (-F3 * FP2 - FP1 - 1.0 / F3) / (F3 * F3)
    HDX = hmD + F2

    X = HDX - hmD
    XDX = NmD * math.exp(X * (FP1 + X * (FP2 + X * FP30)))
    DXDX = XDX * (FP1 + X * (2.0 * FP2 + X * 3.0 * FP30))
    X = hmE - HDX
    XKK = -DXDX * X / (XDX * math.log(XDX / NmE))

    if XKK <= 5.0:
        D1 = DXDX / (XDX * XKK * X ** (XKK - 1.0))
    else:
        XKK = 5.0
        D1 = -math.log(XDX / NmE) / (X**5.0)

    return hmD, XKK, D1, HDX, FP1, FP2, FP30, FP3U


@njit(
    "UniTuple(float64, 5)(float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _tal(SHABR, SDELTA, SHBR, SDTDH0):
    #      SUBROUTINE TAL(SHABR,SDELTA,SHBR,SDTDH0,AUS6,SPT)
    # C-----------------------------------------------------------
    # C CALCULATES THE COEFFICIENTS SPT FOR THE POLYNOMIAL
    # C Y(X)=1+SPT(1)*X**2+SPT(2)*X**3+SPT(3)*X**4+SPT(4)*X**5
    # C TO FIT THE VALLEY IN Y, REPRESENTED BY:
    # C Y(X=0)=1, THE X VALUE OF THE DEEPEST VALLEY POINT (SHABR),
    # C THE PRECENTAGE DEPTH (SDELTA), THE WIDTH (SHBR) AND THE
    # C DERIVATIVE DY/DX AT THE UPPER VALLEY BOUNDRY (SDTDH0).
    # C IF THERE IS AN UNWANTED ADDITIONAL EXTREMUM IN THE VALLEY
    # C REGION, THEN AUS6=.TRUE., ELSE AUS6=.FALSE..
    # C FOR -SDELTA THE COEFF. ARE CALCULATED FOR THE FUNCTION
    # C Y(X)=EXP(SPT(1)*X**2+...+SPT(4)*X**5).
    # C-----------------------------------------------------------
    #      DIMENSION SPT(4)
    #      LOGICAL AUS6
    #      AUS6=.FALSE.
    AUS6 = 0.0
    if SHBR <= 0.0:
        AUS6 = 1.0
        return 0.0, 0.0, 0.0, 0.0, 1.0

    Z1 = -SDELTA / (100.0 * SHABR * SHABR)
    if not (SDELTA > 0.0):
        SDELTA = -SDELTA
        Z1 = math.log(1.0 - SDELTA / 100.0) / (SHABR * SHABR)

    Z3 = SDTDH0 / (2.0 * SHBR)
    Z4 = SHABR - SHBR
    SPT4 = (
        2.0
        * (Z1 * (SHBR - 2.0 * SHABR) * SHBR + Z3 * Z4 * SHABR)
        / (SHABR * SHBR * Z4 * Z4 * Z4)
    )
    SPT3 = (
        Z1 * (2.0 * SHBR - 3.0 * SHABR) / (SHABR * Z4 * Z4)
        - (2.0 * SHABR + SHBR) * SPT4
    )
    SPT2 = -2.0 * Z1 / SHABR - 2.0 * SHABR * SPT3 - 3.0 * SHABR * SHABR * SPT4
    SPT1 = Z1 - SHABR * (SPT2 + SHABR * (SPT3 + SHABR * SPT4))
    B = 4.0 * SPT3 / (5.0 * SPT4) + SHABR
    C = -2.0 * SPT1 / (5 * SPT4 * SHABR)
    Z2 = B * B / 4.0 - C
    if Z2 < 0.0:
        # success
        return SPT1, SPT2, SPT3, SPT4, AUS6

    Z3 = math.sqrt(Z2)
    Z1 = B / 2.0
    Z2 = -Z1 + Z3
    if Z2 > 0.0 and Z2 < SHBR:
        AUS6 = 1.0

    if math.fabs(Z3) > 1.0e-15:
        Z2 = -Z1 - Z3
        if Z2 > 0.0 and Z2 < SHBR:
            AUS6 = 1.0
    else:
        Z2 = C / Z2
        if Z2 > 0.0 and Z2 < SHBR:
            AUS6 = 1.0

    return SPT1, SPT2, SPT3, SPT4, AUS6


@njit(nogil=True, fastmath=True, error_model="numpy")
def _enight(hour, sax110, sux110):

    if math.fabs(sax110) > 25.0:
        # polar regions
        return sax110 < 0.0
    elif sax110 <= sux110:
        return (hour > sux110) or (hour < sax110)
    else:
        return (hour > sux110) and (hour < sax110)


@njit(
    "UniTuple(float64, 5)(float64, float64, float64, float64, float64, int64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _E_valley(modip, hour, SAX110, SUX110, WIDTH, seasn):
    """
    E-layer valley
    WIDTH = hvt - hmE

    if WIDTH < 0.0 then WIDTH will be calculated using the IRI
    """
    XDELS = [5.0, 5.0, 5.0, 10.0]
    DNDS = [0.016, 0.01, 0.016, 0.016]

    amodip = math.fabs(modip)

    if amodip >= 18.0:
        DELA = 1.0 + math.exp(-(amodip - 30.0) / 10.0)
    else:
        DELA = 4.32

    XDEL = XDELS[seasn - 1] / DELA
    DNDHBR = DNDS[seasn - 1] / DELA
    HDEEP = _hpol(hour, 10.5 / DELA, 28.0, SAX110, SUX110, 1.0, 1.0)
    if WIDTH < 0.0:
        WIDTH = _hpol(hour, 17.8 / DELA, 45.0 + 22.0 / DELA, SAX110, SUX110, 1.0, 1.0)

    DEPTH = _hpol(hour, XDEL, 81.0, SAX110, SUX110, 1.0, 1.0)
    DLNDH = _hpol(hour, DNDHBR, 0.06, SAX110, SUX110, 1.0, 1.0)

    if DEPTH < 1.0:
        WIDTH = 0.0

    eNight = _enight(hour, SAX110, SUX110)
    if eNight:
        DEPTH = -DEPTH

    X1, X2, X3, X4, AUS6 = _tal(HDEEP, DEPTH, WIDTH, DLNDH)
    if AUS6 > 0.0:
        # fitting failed, just make it flat
        X1 = 0.0
        X2 = 0.0
        X3 = 0.0
        X4 = 0.0
        WIDTH = 0.0

    return X1, X2, X3, X4, WIDTH


###############################################################################


@njit(
    "float64(float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _C1(modip, hour, SR, SS):
    """
    F1 layer shape parameters

    """

    amodip = math.fabs(modip)

    if amodip >= 18.0:
        C10 = 0.09 + 0.11 * _eps_0(amodip, 10.0, 30.0)
    else:
        C10 = 0.1155

    if SR == SS:
        return 2.5 * C10

    C1 = 2.5 * C10 * math.cos(math.pi * (hour - 12.0) / (SS - SR))

    return max(C1, 0.0)


@njit(
    "UniTuple(float64, 4)(float64, float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _soco(doy, hour, lat, lon, height):
    #        subroutine soco (ld,t,flat,Elon,height,
    #     &          DECLIN, ZENITH, SUNRSE, SUNSET)
    # c--------------------------------------------------------------------
    # c       s/r to calculate the solar declination, zenith angle, and
    # c       sunrise & sunset times  - based on Newbern Smith's algorithm
    # c       [leo mcnamara, 1-sep-86, last modified 16-jun-87]
    # c       {dieter bilitza, 30-oct-89, modified for IRI application}
    # c
    # c in:   ld      local day of year
    # c       t       local hour (decimal)
    # c       flat    northern latitude in degrees
    # c       elon    east longitude in degrees
    # c		height	height in km
    # c
    # c out:  declin      declination of the sun in degrees
    # c       zenith      zenith angle of the sun in degrees
    # c       sunrse      local time of sunrise in hours
    # c       sunset      local time of sunset in hours
    # c-------------------------------------------------------------------
    # c
    # c amplitudes of Fourier coefficients  --  1955 epoch.................
    p1 = 0.017203534
    p2 = 0.034407068
    p3 = 0.051610602
    p4 = 0.068814136
    p6 = 0.103221204
    humr = math.pi / 12.0
    # c
    # c s/r is formulated in terms of WEST longitude.......................
    wlon = 360.0 - (lon % 360.0)
    # c
    # c time of equinox for 1980...........................................
    td = doy + (hour + wlon / 15.0) / 24.0
    te = td + 0.9369
    # c
    # c declination of the sun..............................................
    dcl = (
        23.256 * math.sin(p1 * (te - 82.242))
        + 0.381 * math.sin(p2 * (te - 44.855))
        + 0.167 * math.sin(p3 * (te - 23.355))
        - 0.013 * math.sin(p4 * (te + 11.97))
        + 0.011 * math.sin(p6 * (te - 10.41))
        + 0.339137
    )

    declin = dcl
    dc = math.radians(dcl)  # rads now
    # c
    # c the equation of time................................................
    tf = te - 0.5
    eqt = (
        -7.38 * math.sin(p1 * (tf - 4.0))
        - 9.87 * math.sin(p2 * (tf + 9.0))
        + 0.27 * math.sin(p3 * (tf - 53.0))
        - 0.2 * math.cos(p4 * (tf - 17.0))
    )

    et = math.radians(eqt) / 4.0
    # c
    fa = math.radians(lat)
    phi = humr * (hour - 12.0) + et
    # c
    a = math.sin(fa) * math.sin(dc)
    b = math.cos(fa) * math.cos(dc)
    cosx = a + b * math.cos(phi)
    if math.fabs(cosx) > 1.0:
        cosx = math.copysign(1.0, cosx)

    zenith = math.degrees(math.acos(cosx))
    #
    # calculate sunrise and sunset times --  at the ground...........
    # see Explanatory Supplement to the Ephemeris (1961) pg 401......
    # sunrise at height h metres is at...............................
    h = height * 1000.0
    chih = 90.83 + 0.0347 * math.sqrt(h)
    # this includes corrections for horizontal refraction and........
    # semi-diameter of the solar disk................................
    ch = math.cos(math.radians(chih))
    cosphi = (ch - a) / b
    # if abs(secphi) > 1., sun does not rise/set.....................
    # allow for sun never setting - high latitude summer.............
    secphi = 999999.0
    if cosphi != 0.0:
        secphi = 1.0 / cosphi
    sunset = 99.0
    sunrse = 99.0
    if secphi > -1.0 and secphi < 0:
        return (declin, zenith, sunrse, sunset)
    # allow for sun never rising - high latitude winter..............
    sunset = -99.0
    sunrse = -99.0
    if secphi > 0.0 and secphi < 1.0:
        return (declin, zenith, sunrse, sunset)
    # c
    cosx = cosphi
    if abs(cosx) > 1.0:
        cosx = math.copysign(1.0, cosx)
    phi = math.acos(cosx)
    et = et / humr
    phi = phi / humr
    sunrse = (12.0 - phi - et) % 24
    sunset = (12.0 + phi - et) % 24

    # c special case sunrse > sunset
    if sunrse > sunset:
        sunx = math.copysign(99.0, lat)
        if doy > 91 and doy < 273:
            sunset = sunx
            sunrse = sunx
        else:
            sunset = -sunx
            sunrse = -sunx

    return (declin, zenith, sunrse, sunset)


###############################################################################


@njit(
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _Ne_iri(
    glat, glon, alt, NmF2, hmF2, B0, B1, PF1, NmF1, NmE, hmE, modip, doy, hour, NmD
):

    if alt > hmE:
        # bottomside

        if PF1 > 0.5 and 0.9 * NmF1 >= NmE:
            # F1 layer present
            hmF1 = _newton_hmF1(NmF2, hmF2, B0, B1, NmF1, NmE, hmE)
            if alt > hmF1 and hmF1 > 0.0:
                # between hmF2 and hmF1
                x = (hmF2 - alt) / B0
                return NmF2 * _xe2(x, B1)
        else:
            hmF1 = 0.0

        hl_max = hmE + 67.0

        # only calculate sun parameters once
        calc_sun = False

        if hmF1 > 0.0:
            calc_sun = True
            decl, zenith, sax110, sux110 = _soco(doy, hour, glat, glon, 110.0)

            C1 = _C1(modip, hour, sax110, sux110)
            hst = _newton_hst_F1(NmF2, hmF2, B0, B1, hmF1, C1, NmE)

            hF1 = hmF1

            if hst > 0.0:
                HZ_max = 0.5 * (hst + hF1)
            else:
                # edge case, hst == 0.0
                HZ_max = 0.5 * (hl_max + hF1)

            if alt > HZ_max:
                # Guaranteed above HZ
                # Skip calculating lower layers
                x = (hmF2 - _h_star(hmF1, C1, alt)) / B0
                return NmF2 * _xe2(x, B1)

        else:
            hst = _newton_hst(NmF2, hmF2, B0, B1, NmE)
            # temporary value
            hF1_max = 0.5 * (hmF2 + hl_max)

            if hst > hF1_max:
                # fix for high latitudes
                hF1_max = hmF2

            if hst > 0.0:
                HZ_max = 0.5 * (hst + hF1_max)
            else:
                # edge case, hst == 0.0
                HZ_max = 0.5 * (hl_max + hF1_max)

            if alt > HZ_max:
                # Guaranteed above HZ
                # Skip calculating lower layers
                x = (hmF2 - alt) / B0
                return NmF2 * _xe2(x, B1)

        # below the maximum possible value of HZ
        # This part of the profile is complicated
        if not calc_sun:
            decl, zenith, sax110, sux110 = _soco(doy, hour, glat, glon, 110.0)

        eNight = _enight(hour, sax110, sux110)

        seasn = _season(doy, glat)

        X1, X2, X3, X4, WIDTH = _E_valley(modip, hour, sax110, sux110, -1.0, seasn)

        hvt = hmE + WIDTH

        hl = max(hvt, hmE)

        # find proper hF1
        if hmF1 > 0.0:
            hF1 = hmF1
        else:
            hF1 = 0.5 * (hmF2 + hl)

        if hst > hF1:
            # fix for high latitudes
            hF1 = hmF2

        if hst > 0.0:
            HZ = 0.5 * (hst + hF1)
        else:
            # edge case, hst == 0.0
            HZ = 0.5 * (hl + hF1)

        if alt > hmF1 and alt > HZ:
            # Bottomside
            x = (hmF2 - alt) / B0
            return NmF2 * _xe2(x, B1)
        elif alt > HZ:
            # F1 Layer
            x = (hmF2 - _h_star(hmF1, C1, alt)) / B0
            return NmF2 * _xe2(x, B1)

        # check for valley modification case
        if hst > 0.0 and hst < hl:
            T = ((HZ - hst) ** 2) / (hst - hl)

            nhl = HZ + 0.5 * T + math.sqrt(T * (0.25 * T - (hl - HZ)))
            if hmF1 > 0.0:
                # F1 layer present
                xl = (hmF2 - _h_star(hmF1, C1, nhl)) / B0
            else:
                xl = (hmF2 - nhl) / B0

            while ((NmF2 * _xe2(xl, B1) - NmE) / NmE) > 0.01:
                # adjust valley
                if hl <= hmE or hl <= hst:
                    break
                hl = hl - 1.0

                # update intermediate heights
                hvt = hl
                if hmF1 > 0.0:
                    hF1 = hmF1
                else:
                    hF1 = 0.5 * (hmF2 + hl)
                if hst > 0.0:
                    HZ = 0.5 * (hst + hF1)
                else:
                    # edge case, hst == 0.0
                    HZ = 0.5 * (hl + hF1)

                T = ((HZ - hst) ** 2) / (hst - hl)

                nhl = HZ + 0.5 * T + math.sqrt(T * (0.25 * T - (hl - HZ)))
                if hmF1 > 0.0:
                    # F1 layer present
                    xl = (hmF2 - _h_star(hmF1, C1, nhl)) / B0
                else:
                    xl = (hmF2 - nhl) / B0

            X1, X2, X3, X4, WIDTH = _E_valley(
                modip, hour, sax110, sux110, hvt - hmE, seasn
            )

            if X1 == 0.0 and X2 == 0.0 and X3 == 0.0 and X4 == 0.0:
                # valley failed
                hvt = hmE
                hl = hmE

        if alt >= hl:
            # intermediate layer
            if hst > 0.0:

                T = ((HZ - hst) ** 2) / (hst - hl)

                if hst > hl:

                    nh = HZ + 0.5 * T - math.sqrt(T * (0.25 * T - (alt - HZ)))
                elif hst < hl:

                    nh = HZ + 0.5 * T + math.sqrt(T * (0.25 * T - (alt - HZ)))
                else:
                    nh = alt

                if hmF1 > 0.0:
                    # F1 layer present
                    x = (hmF2 - _h_star(hmF1, C1, nh)) / B0
                else:
                    x = (hmF2 - nh) / B0

                return NmF2 * _xe2(x, B1)
            else:
                # edge case, hst == 0.0
                if hmF1 > 0.0:
                    XNEHZ = NmF2 * _xe2((hmF2 - _h_star(hmF1, C1, HZ)) / B0, B1)
                else:
                    XNEHZ = NmF2 * _xe2((hmF2 - HZ) / B0, B1)

                T = (XNEHZ - NmE) / (HZ - hl)

                return NmE + T * (alt - hl)

        elif alt > hmE and alt < hvt:
            # E valley

            t = alt - hmE

            T = t * t * (X1 + t * (X2 + t * (X3 + t * X4)))

            if eNight:
                return NmE * math.exp(T)
            else:
                return NmE * (1 + T)

        else:
            # should not happen
            return 0.0
    else:
        # D region
        decl, zenith, sax80, sux80 = _soco(doy, hour, glat, glon, 80.0)

        hmD, DK, D1, HDX, FP1, FP2, FP30, FP3U = _d_region(
            modip, hour, sax80, sux80, NmE, hmE, NmD
        )
        if alt > HDX:
            return NmE * math.exp(-D1 * (hmE - alt) ** DK)
        else:
            z = alt - hmD
            if z > 0:
                FP3 = FP30
            else:
                FP3 = FP3U
            return NmD * math.exp(z * (FP1 + z * (FP2 + z * FP3)))
