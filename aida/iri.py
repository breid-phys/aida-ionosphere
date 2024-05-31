import math

from numba import njit, vectorize

target = "cpu"


###############################################################################
@njit(nogil=True, fastmath=True, error_model="numpy")
def NmE_min() -> float:
    """
    NmE_min returns the absolute minimum NmE possible in the IRI, corresponding to a minimum of
        solar activity where F107_365 = 60 s.f.u.

    Returns
    -------
    NmE_min : float
        IRI Minimum NmE
    """
    return 0.124e11 * 0.121


###############################################################################
@njit(nogil=True, fastmath=True, error_model="numpy")
def _season(doy: float, glat: float) -> int:
    """
    _season is the integer season index used in the IRI, with Spring starting of Day of Year 45.
        Seasons in the Southern hemisphere are adjusted to match (i.e. Autumn starts on DoY 45)

    Parameters
    ----------
    doy : float
        Day Of Year
    glat : float
        Geodetic Latitude (degrees)

    Returns
    -------
    season : int

        Spring = 1
        Summer = 2
        Autumn = 3
        Winter = 4

    """
    seasn = int((doy + 45.0) / 92.0)
    if glat < 0.0:
        seasn = seasn - 2
    return ((seasn - 1) % 4) + 1


###############################################################################


@njit(nogil=True, fastmath=True, error_model="numpy")
def _eps_0(x: float, sc: float, hx: float) -> float:
    """
    _eps_0 Epstein_0 sigmoid function used in IRI to smoothly step between values.
        Adapted from EPST in IRIFUN.FOR
        See Bilitza et al. 2022 doi:10.1029/2022RG000792

    Parameters
    ----------
    x : float
        Argument
    sc : float
        Thickness
    hx : float
        Center

    Returns
    -------
    float
        Value on (0, 1)
    """
    ARGMAX = 87.3
    D1 = (x - hx) / sc
    if math.fabs(D1) < ARGMAX:
        return 1.0 / (1.0 + math.exp(-D1))
    else:
        return math.copysign(1.0, D1)


###############################################################################


@njit(nogil=True, fastmath=True, error_model="numpy")
def _eptr(x: float, sc: float, hx: float) -> float:
    """
    _eptr Epstein_-1 transition function used in IRI to smoothly step between values.
        Adapted from EPTR in IRIFUN.FOR
        See Bilitza et al. 2022 doi:10.1029/2022RG000792

    Parameters
    ----------
    x : float
        Argument
    sc : float
        Thickness
    hx : float
        Center

    Returns
    -------
    float
        Value on (0, +inf)
    """
    ARGMAX = 87.3
    D1 = (x - hx) / sc
    if math.fabs(D1) < ARGMAX:
        return math.log(1.0 + math.exp(D1))
    else:
        return max(0.0, D1)


###############################################################################


@njit(nogil=True, fastmath=True, error_model="numpy")
def _hpol(
    t: float, YD: float, YN: float, SR: float, SS: float, DSA: float, DSU: float
) -> float:
    """
    _hpol IRI transition function to smoothly convert between two values. Usually used for
        day-night transitions.
        Adapted from HPOL in IRIFUN.FOR
        See Bilitza et al. 2022 doi:10.1029/2022RG000792

    Parameters
    ----------
    t : float
        Time of day (decimal hours)
    YD : float
        Daytime value
    YN : float
        Nigthttime value
    SR : float
        Sunrise Local Time (decimal hours)
    SS : float
        Sunset Local Time (decimal hours)
    DSA : float
        Sunrise Step Width
    DSU : float
        Sunset Step Width

    Returns
    -------
    float
        Value on (YD, YN)

    Notes
    -------
    See also _eps_0, _soco
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
def _xe2(x: float, B1: float) -> float:
    """
    _xe2 IRI Bottomside function. Needs to be multiplied by the peak density (NmF2 or NmF1).
        Equivalent to XE2 in IRIFUN.FOR
        See Bilitza et al. 2022 doi:10.1029/2022RG000792

    Parameters
    ----------
    x : float
        (h - alt) / B (Unitless)
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)

    Returns
    -------
    float
        Electron Density (as fraction of peak density)
    """
    return math.exp(-(math.pow(x, B1))) / math.cosh(x)


###############################################################################


@njit(
    "float64(float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _dxe2(x: float, B1: float) -> float:
    """
    _dxe2 Derivative of IRI bottomside function XE2. Used for root-finding.

    Parameters
    ----------
    x : float
        (h - alt) / B (Unitless)
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)

    Returns
    -------
    float
        Derivative of XE2 at x
    """
    return -_xe2(x, B1) * (math.tanh(x) + B1 * math.pow(x, B1 - 1.0))


###############################################################################


@njit(nogil=True, fastmath=True, error_model="numpy")
def _h_star(hmF1: float, C1: float, h: float) -> float:
    """
    _h_star give the modified height h* used for the F1 layer in the IRI.
        See Bilitza et al. 2022 doi:10.1029/2022RG000792

    Parameters
    ----------
    hmF1 : float
        F1 Layer Peak Altitude (km)
    C1 : float
        IRI F1 Layer Shape Parameter (Unitless)
    h : float
        height (km)

    Returns
    -------
    float
        modified height h*
    """
    return hmF1 * (1.0 - ((hmF1 - h) / hmF1) ** (1.0 + C1))


###############################################################################
@njit(nogil=True, fastmath=True, error_model="numpy")
def _asech(x: float) -> float:
    """
    _asech Inverse hypoerbolic secant.

    Parameters
    ----------
    x : float
        (h - alt) / B (Unitless)

    Returns
    -------
    float
        arcsech(x)
    """
    ix = 1.0 / x
    return math.log(ix + math.sqrt(ix * ix - 1.0))


@njit(
    "float64(float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _newton_guess(A: float, B1: float) -> float:
    """
    _newton_guess returns an initial guess for root finding.
        Gives a result within 0.01 for most values of A and B1 likely to occur for physically
        realistic profiles.

    Parameters
    ----------
    A : float
        _description_
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)

    Returns
    -------
    float
        _description_
    """
    lB = math.log(B1)
    lA = math.log(A)

    XB1 = -0.920140793640201 + 0.384069525658598 * lB
    XB2 = -0.227098194315023 + 0.385002900867688 * lB
    XB3 = -0.049711817247645 + 0.080686056815511 * lB

    x = (XB1 * lA + XB2 * lA**2 + XB3 * lA**3) ** (1.0 / B1)
    x = min(x, _asech(A))
    return x


###############################################################################
@njit(
    "float64(float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _newton(A: float, B1: float) -> float:
    """
    _newton _summary_

    Parameters
    ----------
    A : float
        _description_
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)

    Returns
    -------
    float
        _description_
    """

    # this is the tolerance used in all IRI
    tol = 0.01

    # first guess
    x = _newton_guess(A, B1)

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
def _newton_hmF1(
    NmF2: float, hmF2: float, B0: float, B1: float, NmF1: float, NmE: float, hmE: float
) -> float:
    """
    _newton_hmF1 _summary_

    Parameters
    ----------
    NmF2 : float
        F2 Layer Peak Density (m^-3)
    hmF2 : float
        F2 Layer Peak Altitude (km)
    B0 : float
        IRI Bottomside Thickness Parameter (km)
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)
    NmF1 : float
        F1 Layer Peak Density (m^-3)
    NmE : float
        E Layer Peak Density (m^-3)
    hmE : float
        E Layer Peak Altitude (km)

    Returns
    -------
    float
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


@vectorize(
    "float64(float64, float64, float64, float64, float64, float64, float64)",
    nopython=True,
    target=target,
    fastmath=True,
)
def newton_hmF1(
    NmF2: float, hmF2: float, B0: float, B1: float, NmF1: float, NmE: float, hmE: float
) -> float:
    """
    newton_hmF1 _summary_

    Parameters
    ----------
    NmF2 : float
        F2 Layer Peak Density (m^-3)
    hmF2 : float
        F2 Layer Peak Altitude (km)
    B0 : float
        IRI Bottomside Thickness Parameter (km)
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)
    NmF1 : float
        F1 Layer Peak Density (m^-3)
    NmE : float
        E Layer Peak Density (m^-3)
    hmE : float
        E Layer Peak Altitude (km)

    Returns
    -------
    float
        _description_
    """
    return _newton_hmF1(NmF2, hmF2, B0, B1, NmF1, NmE, hmE)


@njit(
    "float64(float64, float64, float64, float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _newton_hst_F1(
    NmF2: float, hmF2: float, B0: float, B1: float, hmF1: float, C1: float, NmE: float
) -> float:
    """
    _newton_hst_F1 _summary_

    Parameters
    ----------
    NmF2 : float
        F2 Layer Peak Density (m^-3)
    hmF2 : float
        F2 Layer Peak Altitude (km)
    B0 : float
        IRI Bottomside Thickness Parameter (km)
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)
    hmF1 : float
        F1 Layer Peak Altitude (km)
    C1 : float
        IRI F1 Layer Shape Parameter (Unitless)
    NmE : float
        E Layer Peak Density (m^-3)

    Returns
    -------
    float
        _description_
    """

    A = NmE / NmF2

    hs3 = hmF2 - _newton(A, B1) * B0

    h = hmF1 - hmF1 * (1.0 - hs3 / hmF1) ** (1.0 / (1.0 + C1))

    return h


@njit(
    "float64(float64, float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _newton_hst(NmF2: float, hmF2: float, B0: float, B1: float, NmE: float) -> float:
    """
    _newton_hst _summary_

    Parameters
    ----------
    NmF2 : float
        F2 Layer Peak Density (m^-3)
    hmF2 : float
        F2 Layer Peak Altitude (km)
    B0 : float
        IRI Bottomside Thickness Parameter (km)
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)
    NmE : float
        E Layer Peak Density (m^-3)

    Returns
    -------
    float
        _description_
    """

    A = NmE / NmF2
    h = hmF2 - _newton(A, B1) * B0
    return h


@njit(
    "UniTuple(float64, 9)(float64, float64, float64, float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _d_region(
    modip: float,
    hour: float,
    SAX80: float,
    SUX80: float,
    NmE: float,
    hmE: float,
    NmD: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    _d_region _summary_

    Parameters
    ----------
    modip : float
        Modified Magnetic Dip Angle (degrees)
    hour : float
        Local Time (0-24 Decimal Hours))
    SAX80 : float
        _description_
    SUX80 : float
        _description_
    NmE : float
        E Layer Peak Density (m^-3)
    hmE : float
        E Layer Peak Altitude (km)
    NmD : float
        D Layer Peak Density (m^-3)

    Returns
    -------
    tuple[float, float, float, float, float, float, float, float, float]
        _description_
    """
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

    if XDX > 0.95 * NmE:
        NmD = 0.95 * NmE / math.exp(X * (FP1 + X * (FP2 + X * FP30)))
        XDX = NmD * math.exp(X * (FP1 + X * (FP2 + X * FP30)))

    DXDX = XDX * (FP1 + X * (2.0 * FP2 + X * 3.0 * FP30))
    X = hmE - HDX
    XKK = -DXDX * X / (XDX * math.log(XDX / NmE))

    if XKK <= 5.0:
        D1 = DXDX / (XDX * XKK * X ** (XKK - 1.0))
    else:
        XKK = 5.0
        D1 = -math.log(XDX / NmE) / (X**XKK)

    return hmD, XKK, D1, HDX, FP1, FP2, FP30, FP3U, NmD


@njit(
    "UniTuple(float64, 5)(float64, float64, float64, float64)",
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _tal(
    SHABR: float, SDELTA: float, SHBR: float, SDTDH0: float
) -> tuple[float, float, float, float, float]:
    """
    _tal _summary_

    Parameters
    ----------
    SHABR : float
        _description_
    SDELTA : float
        _description_
    SHBR : float
        _description_
    SDTDH0 : float
        _description_

    Returns
    -------
    tuple[float, float, float, float, float]
        _description_
    """
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
def _enight(hour: float, sax110: float, sux110: float) -> float:
    """
    _enight _summary_

    Parameters
    ----------
    hour : float
        Local Time (0-24 Decimal Hours))
    sax110 : float
        _description_
    sux110 : float
        _description_

    Returns
    -------
    float
        _description_
    """

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
def _E_valley(
    modip: float, hour: float, SAX110: float, SUX110: float, WIDTH: float, seasn: int
) -> tuple[float, float, float, float, float]:
    """
    _E_valley _summary_

    Parameters
    ----------
    modip : float
        Modified Magnetic Dip Angle (degrees)
    hour : float
        Local Time (0-24 Decimal Hours))
    SAX110 : float
        _description_
    SUX110 : float
        _description_
    WIDTH : float
        _description_
    seasn : int
        _description_

    Returns
    -------
    tuple[float, float, float, float, float]
        _description_
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
def _C1(modip: float, hour: float, SR: float, SS: float) -> float:
    """
    _C1 _summary_

    Parameters
    ----------
    modip : float
        Modified Magnetic Dip Angle (degrees)
    hour : float
        Local Time (0-24 Decimal Hours))
    SR : float
        _description_
    SS : float
        _description_

    Returns
    -------
    float
        _description_
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
def _soco(
    doy: float, hour: float, glat: float, glon: float, alt: float
) -> tuple[float, float, float, float]:
    """
    _soco _summary_

    Parameters
    ----------
    doy : float
        Day Of Year
    hour : float
        Local Time (0-24 Decimal Hours))
    glat : float
        Geodetic Latitude (degrees)
    lon : float
        _description_
    height : float
        _description_

    Returns
    -------
    tuple[float, float, float, float]
        _description_
    """
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
    wlon = 360.0 - (glon % 360.0)
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
    fa = math.radians(glat)
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
    h = alt * 1000.0
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
        sunx = math.copysign(99.0, glat)
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
    glat: float,
    glon: float,
    alt: float,
    NmF2: float,
    hmF2: float,
    B0: float,
    B1: float,
    PF1: float,
    NmF1: float,
    NmE: float,
    hmE: float,
    modip: float,
    doy: float,
    hour: float,
    NmD: float,
) -> float:
    """
    _Ne_iri _summary_

    Parameters
    ----------
    glat : float
        Geodetic Latitude (degrees)
    glon : float
        Geodetic Longitude (degrees)
    alt : float
        altitude (km)
    NmF2 : float
        F2 Layer Peak Density (m^-3)
    hmF2 : float
        F2 Layer Peak Altitude (km)
    B0 : float
        IRI Bottomside Thickness Parameter (km)
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)
    PF1 : float
        Probability of F1 Layer (0-1)
    NmF1 : float
        F1 Layer Peak Density (m^-3)
    NmE : float
        E Layer Peak Density (m^-3)
    hmE : float
        E Layer Peak Altitude (km)
    modip : float
        Modified Magnetic Dip Angle (degrees)
    doy : float
        Day Of Year
    hour : float
        Local Time (0-24 Decimal Hours))
    NmD : float
        D Layer Peak Density (m^-3)

    Returns
    -------
    float
        _description_

    Notes
    -------

    IRI expects the NmF2, NmF1, NmE, NmD to be in m^-3
    """

    # NmE min
    NmE = max(NmE, NmE_min())

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

        # only calculate sun parameters once
        calc_sun = False

        if hmF1 > 0.0:
            calc_sun = True
            decl, zenith, sax110, sux110 = _soco(doy, hour, glat, glon, 110.0)

            C1 = _C1(modip, hour, sax110, sux110)
            hst = _newton_hst_F1(NmF2, hmF2, B0, B1, hmF1, C1, NmE)
        else:
            hst = _newton_hst(NmF2, hmF2, B0, B1, NmE)

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
            hF1 = min(hst + 0.25 * B0, hmF2)

        # NEWT
        HZ = 0.5 * (max(hst, hl) + hF1)

        if alt > hmF1 and alt > HZ:
            # Bottomside
            x = (hmF2 - alt) / B0
            return NmF2 * _xe2(x, B1)
        elif alt > HZ:
            # F1 Layer
            x = (hmF2 - _h_star(hmF1, C1, alt)) / B0
            return NmF2 * _xe2(x, B1)

        if alt >= hl:
            # intermediate layer

            T = ((HZ - hst) ** 2) / (hst - hl)

            if T < 0:

                a = (hl - hst) / (2 * HZ * hl - HZ * HZ - hl * hl)
                b = 1 - 2 * a * HZ
                c = HZ - a * HZ**2 - b * HZ
                nh = a * (alt**2) + b * (alt) + c

            elif hst > hl:

                nh = HZ + 0.5 * T - math.sqrt(T * (0.25 * T - (alt - HZ)))
            elif hst < hl:
                # should not happen
                nh = HZ + 0.5 * T + math.sqrt(T * (0.25 * T - (alt - HZ)))
            else:
                nh = alt

            if hmF1 > 0.0:
                # F1 layer present
                x = (hmF2 - _h_star(hmF1, C1, nh)) / B0
            else:
                x = (hmF2 - nh) / B0

            return NmF2 * _xe2(x, B1)

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

        hmD, DK, D1, HDX, FP1, FP2, FP30, FP3U, NmD = _d_region(
            modip, hour, sax80, sux80, NmE, hmE, NmD
        )

        if alt < 50:
            return modip
        elif alt < 51:
            return hour
        elif alt < 52:
            return sax80
        elif alt < 53:
            return sux80
        elif alt < 54:
            return NmE
        elif alt < 55:
            return hmE
        elif alt < 56:
            return NmD
        elif alt < 57:
            return DK
        elif alt < 58:
            return D1

        if alt > HDX:
            return NmE * math.exp(-D1 * (hmE - alt) ** DK)
        else:
            z = alt - hmD
            if z > 0:
                FP3 = FP30
            else:
                FP3 = FP3U
            return NmD * math.exp(z * (FP1 + z * (FP2 + z * FP3)))


###############################################################################


@njit(
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def _Ne_iri_stec(
    glat: float,
    glon: float,
    alt: float,
    NmF2: float,
    hmF2: float,
    B0: float,
    B1: float,
    PF1: float,
    NmF1: float,
    NmE: float,
    hmE: float,
    modip: float,
    doy: float,
    hour: float,
    NmD: float,
) -> float:
    """
    _Ne_iri_stec _summary_

    Parameters
    ----------
    glat : float
        Geodetic Latitude (degrees)
    glon : float
        Geodetic Longitude (degrees)
    alt : float
        altitude (km)
    NmF2 : float
        F2 Layer Peak Density (m^-3)
    hmF2 : float
        F2 Layer Peak Altitude (km)
    B0 : float
        IRI Bottomside Thickness Parameter (km)
    B1 : float
        IRI Bottomside Shape Parameter (Unitless)
    PF1 : float
        Probability of F1 Layer (0-1)
    NmF1 : float
        F1 Layer Peak Density (m^-3)
    NmE : float
        E Layer Peak Density (m^-3)
    hmE : float
        E Layer Peak Altitude (km)
    modip : float
        Modified Magnetic Dip Angle (degrees)
    doy : float
        Day Of Year
    hour : float
        Local Time (0-24 Decimal Hours)
    NmD : float
        D Layer Peak Density (m^-3)

    Returns
    -------
    float
        _description_
    """

    if alt > hmE:
        # bottomside

        if PF1 > 0.5 and 0.9 * NmF1 >= NmE:
            # F1 layer present
            A = NmF1 / NmF2

            hmF1 = hmF2 - _newton_guess(A, B1) * B0

            if alt > hmF1 and hmF1 > hmE:
                # between hmF2 and hmF1
                x = (hmF2 - alt) / B0
                return max(NmE, NmF2 * _xe2(x, B1))

        else:
            hmF1 = 0.0

        if hmF1 > 0.0:
            decl, zenith, sax110, sux110 = _soco(doy, hour, glat, glon, 110.0)

            C1 = _C1(modip, hour, sax110, sux110)

            A = NmE / NmF2

            hs3 = hmF2 - _newton_guess(A, B1) * B0

            hst = hmF1 - hmF1 * (1.0 - hs3 / hmF1) ** (1.0 / (1.0 + C1))

            HZ = 0.5 * (max(hst, hmE) + hmF1)

            if alt > HZ:

                x = (hmF2 - _h_star(hmF1, C1, alt)) / B0

                return max(NmE, NmF2 * _xe2(x, B1))

            else:
                XNEHZ = NmF2 * _xe2((hmF2 - _h_star(hmF1, C1, HZ)) / B0, B1)

                T = (XNEHZ - NmE) / (HZ - hmE)

                return max(NmE, NmE + T * (alt - hmE))

        else:

            A = NmE / NmF2

            hst = hmF2 - _newton_guess(A, B1) * B0

            HZ = 0.5 * (max(hst, hmE) + 0.5 * (hmF2 + hmE))

            if alt > HZ:
                x = (hmF2 - alt) / B0
                return max(NmE, NmF2 * _xe2(x, B1))
            else:
                XNEHZ = NmF2 * _xe2((hmF2 - HZ) / B0, B1)
                T = (XNEHZ - NmE) / (HZ - hmE)
                return max(NmE, NmE + T * (alt - hmE))
    else:
        return NmE * _eps_0(alt, 1.0, 0.5 * (hmE + 85.0))
