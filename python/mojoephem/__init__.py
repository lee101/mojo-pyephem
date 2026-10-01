"""mojo-pyephem: PyEphem's solar-system ephemeris, ported to Mojo.

The public API mirrors upstream `ephem` (the PyPI package that ships
XEphem's `libastro` as a C extension): the same class names, the same
method names, the same argument order and defaults. Every number that
upstream computes in C is computed here by `src/ported.mojo` and crosses
the boundary through `ctypes`; the Python layer is the same shape as
upstream's own `ephem/__init__.py`, which is pure Python over the C
extension.

Covered: Date, Observer, Body/Planet and the ten built-in bodies, the
rise/set/culmination search, the solstice and equinox helpers, the moon
phase helpers, and the module-level julian_date, delta_t, separation,
unrefract, now, localtime and to_timezone. See README.md for what is not
covered.
"""

from __future__ import annotations

import ctypes
import re
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
from datetime import tzinfo as _tzinfo
from math import acos as _acos
from math import cos as _cos
from math import fmod as _fmod
from math import isnan as _isnan
from math import pi as pi
from math import radians as _radians
from math import sin as _sin
from time import localtime as _localtime

import numpy as np

from . import _lib
from . import _tables

__version__ = "0.1.0"

# ------------------------------------------------------------------
# Constants, from upstream ephem/__init__.py.
tau = 6.283185307179586476925287
twopi = pi * 2.0
halfpi = pi / 2.0
quarterpi = pi / 4.0
eighthpi = pi / 8.0

degree = pi / 180.0
arcminute = degree / 60.0
arcsecond = arcminute / 60.0
half_arcsecond = arcsecond / 2.0
tiny = arcsecond / 360.0

c = 299792458.0  # exact speed of light in meters/second
meters_per_au = 1.4959787e11
earth_radius = 6.37816e6
moon_radius = 1.740e6
sun_radius = 6.95e8

MJD0 = 2415020.0
B1900 = 2415020.3135 - MJD0
B1950 = 2433282.4235 - MJD0
J2000 = 2451545.0 - MJD0

_slightly_less_than_zero = -1e-15
_slightly_more_than_pi = pi + 1e-15

hour = 1.0 / 24.0
minute = hour / 60.0
second = minute / 60.0

default_newton_precision = second / 10.0
rise_set_iterations = tuple(range(7))


# ------------------------------------------------------------------
# Angle: a float carrying the per-radian scale of its unit, exactly as
# upstream's _libastro.Angle does. Formatting follows the same
# [-]d:mm:ss.ss convention.
class Angle(float):
    __slots__ = ("unit",)

    def __new__(cls, value, unit=1.0):
        self = float.__new__(cls, value)
        self.unit = unit
        return self

    def __str__(self):
        return self._string()

    def __repr__(self):
        return "<Angle %s>" % self._string()

    def __format__(self, spec):
        if not spec:
            return self._string()
        return format(float(self), spec)

    def _string(self):
        sign = "-" if self < 0 else ""
        v = abs(float(self)) * self.unit
        degrees = int(v)
        minutes = int((v - degrees) * 60)
        seconds = (v - degrees - minutes / 60.0) * 3600.0
        if round(seconds, 2) >= 60.0:
            seconds -= 60.0
            minutes += 1
        if minutes >= 60:
            minutes -= 60
            degrees += 1
        return "%s%d:%02d:%05.2f" % (sign, degrees, minutes, seconds)

    @property
    def radians(self):
        return float(self)

    @property
    def hours(self):
        return float(self) * self.unit

    @property
    def norm(self):
        """This angle normalized to [0, 2*pi)."""
        a = float(self)
        if a < 0.0:
            a = _fmod(a, twopi) + twopi
        elif a >= twopi:
            a = _fmod(a, twopi)
        return Angle(a, self.unit)

    @property
    def znorm(self):
        """This angle normalized to (-pi, pi]."""
        a = float(self)
        if a <= -pi:
            a = _fmod(a + pi, twopi) + pi
        elif a > pi:
            a = _fmod(a - pi, twopi) - pi
        return Angle(a, self.unit)


# Upstream compiles this once and hands it to its C scanner: a colon, or
# whitespace that is neither leading nor trailing, separates sexagesimal
# fields.
_scansexa_split = re.compile(
    r"""
    \s*:\s*         # A colon optionally surrounded by whitespace,
    |               # or,
    (?<!^)\s+(?!$)  # whitespace not at the start or end of the string.
    """,
    re.X,
).split


def _sexagesimal(text: str) -> float:
    """Parse '51:28:40.2', '51 28 40.2' or '51.4778' as degrees."""
    parts = _scansexa_split(text.strip())
    if not parts or not parts[0]:
        raise ValueError("could not parse angle %r" % (text,))
    sign = 1.0
    first = parts[0]
    if first[0] in "+-":
        if first[0] == "-":
            sign = -1.0
        first = first[1:]
    values = [float(first)]
    for extra in parts[1:]:
        values.append(float(extra))
    if len(values) > 3:
        raise ValueError("too many fields in angle %r" % (text,))
    total = values[0]
    scale = 60.0
    for v in values[1:]:
        total += v / scale
        scale *= 60.0
    return sign * total


def degrees(x):
    """An Angle in radians whose natural input is degrees."""
    if isinstance(x, str):
        x = _radians(_sexagesimal(x))
    elif isinstance(x, Angle):
        x = float(x)
    return Angle(x, 180.0 / pi)


def hours(x):
    """An Angle in radians whose natural input is hours of right ascension."""
    if isinstance(x, str):
        x = _radians(_sexagesimal(x) * 15.0)
    elif isinstance(x, Angle):
        x = float(x)
    return Angle(x, 12.0 / pi)


def radians(x):
    return Angle(x, 1.0)


# ------------------------------------------------------------------
# Date: a float of days since 1900 January 0.5, with the printing and
# parsing behaviour of upstream's Date.
_DATE_PATTERNS = (
    # "YYYY/MM/DD", optionally with "HH:MM" or "HH:MM:SS" or ".ffffff"
    re.compile(
        r"^\s*(-?\d+)[/-](\d+)[/-](\d+)"
        r"(?:[ T]+(\d+):(\d+)(?::(\d+(?:\.\d+)?))?)?\s*$"
    ),
    # "M/D/YYYY", optionally with a time
    re.compile(
        r"^\s*(\d+)[/-](\d+)[/-](-?\d+)"
        r"(?:[ T]+(\d+):(\d+)(?::(\d+(?:\.\d+))?)?)?\s*$"
    ),
)


class Date(float):
    """A moment in time, in days since 1900 January 0.5 UTC."""

    __slots__ = ()

    def __new__(cls, value=None):
        if value is None:
            import time

            return float.__new__(cls, time.time() / 86400.0 + 25567.5)
        return float.__new__(cls, _parse_mjd(value))

    def __str__(self):
        y, mo, d, H, M, S = _mjd_six(float(self))
        return "%d/%d/%d %02d:%02d:%02d" % (y, mo, d, H, M, int(round(S)))

    def __repr__(self):
        return "ephem.Date('%s')" % self

    def triple(self):
        """The (year, month, day) calendar date, as upstream's Date does.

        The day carries the fraction of the day past midnight, matching
        upstream's mjd_cal.
        """
        mn, dy, yr = _mjd_cal(float(self))
        return yr, mn, dy

    # Arithmetic keeps upstream's shape: a Date in, a Date out; a
    # difference of two Dates is a plain number of days; anything else
    # (a numpy array, a Decimal) falls through to float's own behaviour.
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Date(float(self) + float(other))
        return float(self) + other

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Date):
            return float(self) - float(other)
        if isinstance(other, (int, float)):
            return Date(float(self) - float(other))
        return float(self) - other

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Date(float(other) - float(self))
        return other - float(self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Date(float(self) * float(other))
        return float(self) * other

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Date):
            return float(self) / float(other)
        if isinstance(other, (int, float)):
            return Date(float(self) / float(other))
        return float(self) / other

    def __neg__(self):
        return Date(-float(self))


date = Date
angle = Angle


_NUMERIC = frozenset((float, int, np.float64, np.float32, np.int64, np.int32))


def _parse_mjd(value) -> float:
    """Accept what upstream's parse_mjd accepts for the common cases."""
    # The common case is a bare number or a Date, and this sits under every
    # per-call wrapper below, so the exact-type checks come first.
    t = value.__class__
    if t is float or t is Date:
        return value
    if t in _NUMERIC:
        return float(value)
    if isinstance(value, Observer):
        return float(value.date)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, _datetime):
        base = (
            _datetime(
                value.year, value.month, value.day, value.hour, value.minute, value.second
            )
            - _datetime(1900, 1, 1)
        ).total_seconds() / 86400.0
        return base + value.microsecond / 1e6 / 86400.0
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        for i, pattern in enumerate(_DATE_PATTERNS):
            m = pattern.match(text)
            if m is None:
                continue
            if i == 0:
                year, month, day, H, M, S = m.groups()
            else:
                month, day, year, H, M, S = m.groups()
            year, month, day = int(year), int(month), int(day)
            H, M = int(H or 0), int(M or 0)
            S = float(S or 0)
            if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= H < 24 and 0 <= M < 60):
                raise ValueError("invalid date %r" % value)
            if S >= 60:
                raise ValueError("invalid seconds %r" % value)
            # cal_mjd already returns midnight of the given calendar day.
            base = _lib.lib().mpe_cal_mjd(month, float(day), year)
            return base + (H / 24.0 + M / 1440.0 + S / 86400.0)
        raise ValueError("unrecognized date %r" % (value,))
    raise TypeError("cannot interpret %r as a Date" % (value,))


def _mjd_six(mj: float):
    """(year, month, day, hour, minute, second) for a PyEphem MJD.

    mjd_cal reports the calendar date whose 12h falls on `mj`, so the day
    number carries the fraction of a day since midnight. Times are rounded
    to the nearest second, which can carry into the next day.
    """
    mn, dy, yr = _mjd_cal(mj)
    day = int(dy)
    sec = round((dy - day) * 86400.0)
    if sec >= 86400:
        mn, dy, yr = _mjd_cal(mj + 1.0)
        day = int(dy)
        sec -= 86400
    return yr, mn, day, int(sec // 3600), int(sec % 3600 // 60), float(sec % 60)


_mn = np.zeros(1, dtype=np.int64)
_dy = np.zeros(1)
_yr = np.zeros(1, dtype=np.int64)


_mjd_cal_entry = None


def _mjd_cal(mj: float):
    """(month, day-with-fraction, year) for a PyEphem MJD."""
    global _mjd_cal_entry
    if _mjd_cal_entry is None:
        # numpy rebuilds a ctypes object on every `.ctypes.data`, and this
        # runs under every Date.__str__ and Date.triple.
        _mjd_cal_entry = (
            _fn("mpe_mjd_cal"),
            _mn.ctypes.data,
            _dy.ctypes.data,
            _yr.ctypes.data,
        )
    entry, mn_addr, dy_addr, yr_addr = _mjd_cal_entry
    entry(mj, mn_addr, dy_addr, yr_addr)
    return int(_mn[0]), float(_dy[0]), int(_yr[0])


# ------------------------------------------------------------------
# Thin, typed wrappers over the kernels. Each names the upstream function
# it forwards to.
#
# A `ctypes` call is about as expensive as the smallest kernel here, so
# every wrapper resolves its entry point, its registry address and its
# scratch buffer once rather than per call. numpy in particular rebuilds a
# ctypes object for `.ctypes.data` on every access, which costs more than
# the call it feeds.
_entry: dict[str, object] = {}
_reg_addr: int | None = None


def _fn(name):
    """The bound ctypes entry point, memoised.

    `CDLL.__getattr__` builds a new `_FuncPtr` on every miss, which costs
    more than the call itself.
    """
    f = _entry.get(name)
    if f is None:
        f = _entry[name] = getattr(_lib.lib(), name)
    return f


def _reg() -> int:
    global _reg_addr
    if _reg_addr is None:
        _reg_addr = _tables.reg()
    return _reg_addr


_RAD = (ctypes.c_double * 3)()
_RAD_ADDR = ctypes.addressof(_RAD)
_PAIR = (ctypes.c_double * 2)()
_PAIR_ADDR = ctypes.addressof(_PAIR)
_HOUR_UNIT = 12.0 / pi
_DEG_UNIT = 180.0 / pi


def julian_date(value=None):
    """The Julian date of the given Date, or of now."""
    return _parse_mjd(value) + MJD0


_deltat_entry = None


def delta_t(value=None):
    """Ephemeris Time minus Universal Time, in seconds."""
    global _deltat_entry
    if _deltat_entry is None:
        _deltat_entry = (_fn("mpe_deltat"), _reg())
    entry, reg = _deltat_entry
    t = value.__class__
    if t in _NUMERIC or t is Date:
        return entry(float(value), reg)
    return entry(_parse_mjd(value), reg)


_unrefract_entry = None


def unrefract(pressure, temperature, altitude):
    """The true altitude whose refracted value is `altitude`."""
    global _unrefract_entry
    if _unrefract_entry is None:
        _unrefract_entry = _fn("mpe_unrefract")
    _RAD[0] = 0.0
    _unrefract_entry(pressure, temperature, altitude, _RAD_ADDR)
    return Angle(_RAD[0], _DEG_UNIT)


_precess_entry = None


def precess(epoch1, epoch2, ra, dec):
    """Correct ra and dec (radians) for precession from one epoch to another."""
    global _precess_entry
    if _precess_entry is None:
        _precess_entry = _fn("mpe_precess")
    _PAIR[0] = ra
    _PAIR[1] = dec
    _precess_entry(float(epoch1), float(epoch2), _PAIR_ADDR)
    return Angle(_PAIR[0], _HOUR_UNIT), Angle(_PAIR[1], _DEG_UNIT)


def ecl_eq(epoch, lon, lat):
    """Ecliptic (lon, lat) to equatorial (ra, dec), as Angles."""
    # upstream's ecl_eq takes the latitude first, like libastro's.
    _fn("mpe_ecl_eq")(float(epoch), float(lat), float(lon), _RAD_ADDR)
    return Angle(_RAD[0], _HOUR_UNIT), Angle(_RAD[1], _DEG_UNIT)


def eq_ecl(epoch, ra, dec):
    """Equatorial (ra, dec) to ecliptic (lon, lat), as Angles."""
    _fn("mpe_eq_ecl")(float(epoch), float(ra), float(dec), _RAD_ADDR)
    return Angle(_RAD[0], _DEG_UNIT), Angle(_RAD[1], _DEG_UNIT)


def obliquity(epoch):
    _fn("mpe_obliquity")(float(epoch), _RAD_ADDR)
    return _RAD[0]


def nutation(epoch):
    _fn("mpe_nutation")(float(epoch), _reg(), _RAD_ADDR)
    return _RAD[0], _RAD[1]


def now_lst(date_value, longitude):
    _fn("mpe_now_lst")(float(date_value), float(longitude), _reg(), _RAD_ADDR)
    return _RAD[0]


def utc_gst(date_value, utc_hours):
    _fn("mpe_utc_gst")(float(date_value), float(utc_hours), _RAD_ADDR)
    return _RAD[0]


_ZERO_DEGREES = Angle(0.0, _DEG_UNIT)


def separation(p, q):
    """The angular separation between two objects or positions."""
    try:
        plng, plat = p
        qlng, qlat = q
        plng = float(plng)
        plat = float(plat)
        qlng = float(qlng)
        qlat = float(qlat)
    except (TypeError, ValueError):
        plng, plat = _separation_arg(p)
        qlng, qlat = _separation_arg(q)
    # rounding errors in the trigonometry might return >0 for this case
    if plat == qlat and plng == qlng:
        return _ZERO_DEGREES
    cosine = _sin(plat) * _sin(qlat) + _cos(plat) * _cos(qlat) * _cos(plng - qlng)
    if cosine >= 1.0:  # rounding sometimes makes this >1
        return _ZERO_DEGREES
    return Angle(_acos(cosine), _DEG_UNIT)


def _separation_arg(arg):
    if isinstance(arg, Observer):
        return arg._lon_radians(), arg._lat_radians()
    if isinstance(arg, Body):
        return float(arg.ra), float(arg.dec)
    if hasattr(arg, "__len__") and len(arg) == 2:
        return float(arg[0]), float(arg[1])
    raise TypeError(
        "each separation argument must be an Observer, a Body, or a pair "
        "of numeric coordinates, not an %r" % (type(arg).__name__,)
    )


# ------------------------------------------------------------------
# Observer
class Observer:
    """A location and time: the geocentric circumstances of one moment.

    Mirrors the Observer_getset table upstream: `date` and `epoch` are
    Dates, `lat`, `lon` and `horizon` are Angles in degrees (a bare number
    is read as degrees, a "d:m:s" string likewise), and the rest are
    plain floats.
    """

    def __init__(self):
        self._date = Date(0.0)
        self._lat = degrees(0.0)
        self._lon = degrees(0.0)
        self.elevation = 0.0  # meters
        self._horizon = degrees(0.0)
        self.temp = 15.0
        self.pressure = 1010.0
        self._epoch = Date(J2000)

    def copy(self):
        other = Observer()
        other.__dict__.update(self.__dict__)
        return other

    def __repr__(self):
        return "<ephem.Observer date=%r lon=%s lat=%s elevation=%sm>" % (
            self.date,
            self.lon,
            self.lat,
            self.elevation,
        )

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value):
        self._date = Date(value)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = Date(value)

    @property
    def lat(self):
        return self._lat

    @lat.setter
    def lat(self, value):
        self._lat = degrees(value)

    @property
    def lon(self):
        return self._lon

    @lon.setter
    def lon(self, value):
        self._lon = degrees(value)

    @property
    def long(self):
        """Alias for lon, for compatibility with older scripts."""
        return self._lon

    @long.setter
    def long(self, value):
        self._lon = degrees(value)

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        self._horizon = degrees(value)

    def _lat_radians(self):
        return float(self._lat)

    def _lon_radians(self):
        return float(self._lon)

    def _elev_radians(self):
        """Elevation as a fraction of the earth's equatorial radius."""
        return self.elevation / earth_radius

    def _epoch_mjd(self):
        return float(self._epoch)

    def sidereal_time(self):
        """The local apparent sidereal time, as an Angle in hours."""
        lst = now_lst(float(self.date), self._lon_radians())
        return Angle(lst * 15.0 * pi / 180.0, 12.0 / pi)

    def compute_pressure(self):
        """Set the atmospheric pressure for the current elevation."""
        self.pressure = 1010.0 * pow(1.0 - 2.2557e-5 * self.elevation, 5.2568)


# ------------------------------------------------------------------
# The Obj field vector the kernel writes, in the order obj_planet fills it.
# Field 4 is the elongation in DEGREES, as upstream's s_elong is; every other
# angular field is in radians. Distances are au, magnitude is quantised to
# hundredths by the kernel's set_smag.
_F_HLON = 0
_F_HLAT = 1
_F_SDIST = 2
_F_EDIST = 3
_F_ELONG = 4
_F_PHASE = 5
_F_MAG = 6
_F_SIZE = 7
_F_GAERA = 8
_F_GAEDEC = 9
_F_ASTRO_RA = 10
_F_ASTRO_DEC = 11
_F_HA = 12
_F_ALT = 13
_F_AZ = 14
_F_RHO = 15
_F_MJ = 16
_NFIELDS = 17

_obj_planet_entry = None


def _compute(planet: int, when, buf: np.ndarray, addr: int) -> np.ndarray:
    """Fill `buf` with the Obj fields for one body at one moment."""
    global _obj_planet_entry
    if _obj_planet_entry is None:
        _obj_planet_entry = (_fn("mpe_obj_planet"), _reg())
    entry, reg = _obj_planet_entry
    if isinstance(when, Observer):
        entry(
            when._date,
            planet,
            float(when._lat),
            float(when._lon),
            when.elevation / earth_radius,
            float(when._epoch),
            1,
            when.pressure,
            when.temp,
            reg,
            addr,
        )
        buf[_F_MJ] = when._date
    else:
        entry(
            _parse_mjd(when), planet, 0.0, 0.0, 0.0, J2000, 0, 0.0, 15.0,
            reg, addr,
        )
    return buf


def compute_series(planet: int, dates, when: Observer | None = None):
    """Positions of one body across many dates, in one Mojo call.

    The scalar path is the same obj_planet() upstream runs once per date;
    this is that call repeated over a buffer, which is the shape the
    kernel is fastest in.
    """
    if not 0 <= planet <= 9:
        raise ValueError("planet must be a PLCode in 0..9, not %r" % (planet,))
    mjs = np.ascontiguousarray(dates, dtype=np.float64)
    n = mjs.size
    out = np.zeros((n, _NFIELDS))
    if n == 0:
        return out
    if when is None:
        lat = lng = elev = 0.0
        epoch, topocentric, pressure, temp = J2000, 0, 0.0, 15.0
    else:
        lat = when._lat_radians()
        lng = when._lon_radians()
        elev = when._elev_radians()
        epoch, topocentric, pressure, temp = (
            when._epoch_mjd(),
            1,
            when.pressure,
            when.temp,
        )
    _fn("mpe_obj_planet_series")(
        mjs.ctypes.data,
        out.ctypes.data,
        n,
        planet,
        lat,
        lng,
        elev,
        epoch,
        topocentric,
        pressure,
        temp,
        _reg(),
    )
    return out


class Body:
    """A celestial object; the base of every body class here."""

    def __init__(self, *args, **kw):
        self.name = kw.pop("name", None)
        self._computed = False
        self._topocentric = False
        self._fields = np.zeros(_NFIELDS)
        # the field vector never moves, so its address is paid for once
        self._addr = self._fields.ctypes.data
        if args or kw:
            self.compute(*args, **kw)

    def compute(self, when):
        """(Re)compute this body's circumstances for a Date or Observer."""
        self._fields = _compute(self.__planet__, when, self._fields, self._addr)
        self._topocentric = isinstance(when, Observer)
        self._computed = True
        return self

    def _require(self, field, topocentric=False):
        if not self._computed:
            raise RuntimeError(
                "field %s undefined until first compute()" % field
            )
        if topocentric and not self._topocentric:
            raise RuntimeError(
                "field %s undefined because the most recent compute() was "
                "supplied a date rather than an Observer" % field
            )
        return self._fields

    # Body_getset, in upstream's order.
    @property
    def ha(self):
        return hours(self._require("ha")[_F_HA])

    @property
    def ra(self):
        return hours(self._require("ra")[_F_GAERA])

    @property
    def dec(self):
        return degrees(self._require("dec")[_F_GAERA + 1])

    @property
    def g_ra(self):
        return hours(self._require("g_ra")[_F_GAERA])

    @property
    def g_dec(self):
        return degrees(self._require("g_dec")[_F_GAERA + 1])

    @property
    def a_ra(self):
        return hours(self._require("a_ra")[_F_ASTRO_RA])

    @property
    def a_dec(self):
        return degrees(self._require("a_dec")[_F_ASTRO_RA + 1])

    @property
    def a_epoch(self):
        return Date(J2000)

    @property
    def elong(self):
        return degrees(self._require("elong")[_F_ELONG] * pi / 180.0)

    @property
    def mag(self):
        return self._require("mag")[_F_MAG]

    @property
    def size(self):
        return self._require("size")[_F_SIZE]

    @property
    def radius(self):
        return degrees(self._require("radius")[_F_SIZE] * pi / 360.0 / 3600.0)

    @property
    def az(self):
        return degrees(self._require("az", True)[_F_AZ])

    @property
    def alt(self):
        return degrees(self._require("alt", True)[_F_ALT])

    # Planet_getset
    @property
    def hlon(self):
        return degrees(self._require("hlon")[_F_HLON])

    hlong = hlon

    @property
    def hlat(self):
        return degrees(self._require("hlat")[_F_HLON + 1])

    @property
    def sun_distance(self):
        return self._require("sun_distance")[_F_SDIST]

    @property
    def earth_distance(self):
        return self._require("earth_distance")[_F_EDIST]

    @property
    def phase(self):
        return self._require("phase")[_F_PHASE]

    def __repr__(self):
        if not self._computed:
            return "<%s>" % type(self).__name__
        return "<%s %s at %r>" % (
            type(self).__name__,
            self.ra,
            self._fields[_F_MJ],
        )


class Planet(Body):
    """One of the built-in solar-system bodies."""

    __planet__ = -1

    def __init__(self, *args, **kw):
        Body.__init__(self, *args, **kw)


class Mercury(Planet):
    __planet__ = 0


class Venus(Planet):
    __planet__ = 1


class Mars(Planet):
    __planet__ = 2


class Jupiter(Planet):
    __planet__ = 3


class Saturn(Planet):
    __planet__ = 4


class Uranus(Planet):
    __planet__ = 5


class Neptune(Planet):
    __planet__ = 6


class Pluto(Planet):
    __planet__ = 7


class Sun(Planet):
    __planet__ = 8


class Moon(Planet):
    __planet__ = 9


# ------------------------------------------------------------------
# Rise, set and culmination. Upstream implements this search in Python
# (ephem/__init__.py _find_rise_or_set), on top of body.ha and body.dec;
# it is reproduced here unchanged apart from using our own unrefract.
class CircumpolarError(ValueError):
    pass


class NeverUpError(CircumpolarError):
    pass


class AlwaysUpError(CircumpolarError):
    pass


def _plusminus_pi(angle):
    return (angle - pi) % tau - pi


def newton(f, x0, x1, precision=default_newton_precision):
    """Return an x-value at which the given function reaches zero.

    Stops and declares victory once the x-value is within ``precision``
    of the solution, which defaults to a half-second of clock time.

    """
    f0, f1 = f(x0), f(x1)
    while f1 and abs(x1 - x0) > precision and f1 != f0:
        x0, x1 = x1, x1 + (x1 - x0) / (f0 / f1 - 1)
        f0, f1 = f1, f(x1)
    return x1


def _previous_transit(self, body, start=None):
    return self._compute_transit(body, start, -1, 0.0)


def _next_transit(self, body, start=None):
    return self._compute_transit(body, start, +1, 0.0)


def _previous_antitransit(self, body, start=None):
    return self._compute_transit(body, start, -1, pi)


def _next_antitransit(self, body, start=None):
    return self._compute_transit(body, start, +1, pi)


def _body(body):
    if isinstance(body, type):
        return body()
    return body


def disallow_circumpolar(self, declination):
    """Raise if an object at this declination is circumpolar here."""
    if abs(self._lat_radians() - declination) >= halfpi:
        raise NeverUpError(
            "The declination %s never rises above the horizon at latitude %s"
            % (declination, self.lat)
        )
    if abs(self._lat_radians() + declination) >= halfpi:
        raise AlwaysUpError(
            "The declination %s is always above the horizon at latitude %s"
            % (declination, self.lat)
        )


def _target_hour_angle(self, body, alt):
    lat = self._lat_radians()
    dec = float(body.dec)
    arg = (_sin(alt) - _sin(lat) * _sin(dec)) / (_cos(lat) * _cos(dec))

    if arg < -1.0:
        return _slightly_more_than_pi
    elif arg > 1.0:
        return _slightly_less_than_zero

    return _acos(arg)


def _find_rise_or_set(self, body, start, use_center, direction, do_rising):
    body = _body(body)
    original_pressure = self.pressure
    original_date = self.date
    try:
        self.pressure = 0.0  # otherwise geometry doesn't work
        if start is not None:
            self.date = start
        prev_ha = None
        abs_target_ha = None
        for _ in rise_set_iterations:
            if _isnan(self.date):
                raise ValueError(
                    "cannot find a next rising or setting if the date is NaN"
                )
            body.compute(self)
            horizon = float(self.horizon)
            if not use_center:
                horizon -= float(body.radius)
            if original_pressure:
                horizon = float(unrefract(original_pressure, self.temp, horizon))
            abs_target_ha = self._target_hour_angle(body, horizon)
            if do_rising:
                target_ha = -abs_target_ha  # rises in east (az 0-180)
            else:
                target_ha = abs_target_ha  # sets in west (az 180-360)
            ha = float(body.ha)
            difference = target_ha - ha
            if prev_ha is None:
                difference %= tau  # force angle to be positive
                if direction < 0:
                    difference -= tau
                bump = difference / tau
                if abs(bump) < default_newton_precision:
                    # Already at target event: move forward to next one.
                    bump += direction
            else:
                difference = _plusminus_pi(difference)
                bump = difference / tau
            if abs(bump) < default_newton_precision:
                break
            self.date = self.date + bump
            prev_ha = ha

        if abs_target_ha == _slightly_more_than_pi:
            raise AlwaysUpError(
                "%r is above the horizon at %s" % (body.name, self.date)
            )

        if abs_target_ha == _slightly_less_than_zero:
            raise NeverUpError(
                "%r is below the horizon at %s" % (body.name, self.date)
            )

        return Date(self.date)
    finally:
        if self.pressure != original_pressure:
            self.pressure = original_pressure
            body.compute(self)
        self.date = original_date


def previous_transit(self, body, start=None):
    """Search for the given body's previous culmination."""
    return self._compute_transit(body, start, -1, 0.0)


def next_transit(self, body, start=None):
    """Search for the given body's next culmination."""
    return self._compute_transit(body, start, +1, 0.0)


def previous_antitransit(self, body, start=None):
    """Search for the given body's previous lower culmination."""
    return self._compute_transit(body, start, -1, pi)


def next_antitransit(self, body, start=None):
    """Search for the given body's next lower culmination."""
    return self._compute_transit(body, start, +1, pi)


def _compute_transit(self, body, start, sign, offset):
    """Internal function used to compute transits, as upstream does it."""

    def f(d):
        self.date = d
        body.compute(self)
        return float(degrees(offset - self.sidereal_time() + body.g_ra).znorm)

    if start is not None:
        self.date = Date(start)
    sidereal_time = self.sidereal_time
    body.compute(self)
    ha = sidereal_time() - float(body.g_ra)
    ha_to_move = (offset - ha) % (sign * twopi)
    if abs(ha_to_move) < tiny:
        ha_to_move = sign * twopi
    d = self.date + ha_to_move / twopi
    return Date(newton(f, d, d + minute))


Observer._compute_transit = _compute_transit
Observer.previous_transit = previous_transit
Observer.next_transit = next_transit
Observer.previous_antitransit = previous_antitransit
Observer.next_antitransit = next_antitransit
Observer.disallow_circumpolar = disallow_circumpolar
Observer._target_hour_angle = _target_hour_angle
Observer._find_rise_or_set = _find_rise_or_set


def previous_rising(self, body, start=None, use_center=False):
    """Search for the given body's previous rising."""
    return self._find_rise_or_set(body, start, use_center, -1, True)


def previous_setting(self, body, start=None, use_center=False):
    """Search for the given body's previous setting."""
    return self._find_rise_or_set(body, start, use_center, -1, False)


def next_rising(self, body, start=None, use_center=False):
    """Search for the given body's next rising."""
    return self._find_rise_or_set(body, start, use_center, +1, True)


def next_setting(self, body, start=None, use_center=False):
    """Search for the given body's next setting."""
    return self._find_rise_or_set(body, start, use_center, +1, False)


Observer.previous_rising = previous_rising
Observer.previous_setting = previous_setting
Observer.next_rising = next_rising
Observer.next_setting = next_setting


# ------------------------------------------------------------------
# Solstices, equinoxes and moon phases. Upstream's own Newton searches.
_sun = Sun()  # used for computing equinoxes


def holiday(d0, motion, offset):
    """Function that assists the finding of equinoxes and solstices."""

    def f(d):
        _sun.compute(d)
        return (_sun.ra + eighthpi) % quarterpi - eighthpi

    d0 = Date(d0)
    _sun.compute(d0)
    angle_to_cover = motion - (_sun.ra + offset) % motion
    if abs(angle_to_cover) < tiny:
        angle_to_cover = motion
    d = d0 + 365.25 * angle_to_cover / twopi
    return Date(newton(f, d, d + hour))


def previous_vernal_equinox(date):
    """Return the date of the previous vernal equinox."""
    return holiday(date, -twopi, 0)


def next_vernal_equinox(date):
    """Return the date of the next vernal equinox."""
    return holiday(date, twopi, 0)


def previous_summer_solstice(date):
    """Return the date of the previous summer solstice."""
    return holiday(date, -twopi, pi + halfpi)


def next_summer_solstice(date):
    """Return the date of the next summer solstice."""
    return holiday(date, twopi, pi + halfpi)


def previous_autumnal_equinox(date):
    """Return the date of the previous autumnal equinox."""
    return holiday(date, -twopi, pi)


def next_autumnal_equinox(date):
    """Return the date of the next autumnal equinox."""
    return holiday(date, twopi, pi)


def previous_winter_solstice(date):
    """Return the date of the previous winter solstice."""
    return holiday(date, -twopi, halfpi)


def next_winter_solstice(date):
    """Return the date of the next winter solstice."""
    return holiday(date, twopi, halfpi)


# Common synonyms.
next_spring_equinox = next_vernal_equinox
previous_spring_equinox = previous_vernal_equinox
next_fall_equinox = next_autumnal_equinox = next_autumnal_equinox
previous_fall_equinox = previous_autumnal_equinox = previous_autumnal_equinox


# More-general functions that find any equinox or solstice.
def previous_equinox(date):
    """Return the date of the previous equinox."""
    return holiday(date, -pi, 0)


def next_equinox(date):
    """Return the date of the next equinox."""
    return holiday(date, pi, 0)


def previous_solstice(date):
    """Return the date of the previous solstice."""
    return holiday(date, -pi, halfpi)


def next_solstice(date):
    """Return the date of the next solstice."""
    return holiday(date, pi, halfpi)


_moon = Moon()  # used for computing Moon phases


def _find_moon_phase(d0, motion, target):
    """Function that assists the finding of moon phases."""

    def f(d):
        _sun.compute(d)
        _moon.compute(d)
        slon = eq_ecl(d, _sun.g_ra, _sun.g_dec)[0]
        mlon = eq_ecl(d, _moon.g_ra, _moon.g_dec)[0]
        return (mlon - slon - antitarget) % twopi - pi

    antitarget = target + pi
    d0 = Date(d0)
    f0 = f(d0)
    angle_to_cover = (-f0) % motion
    if abs(angle_to_cover) < tiny:
        angle_to_cover = motion
    d = d0 + 29.53 * angle_to_cover / twopi
    return Date(newton(f, d, d + hour))


def previous_new_moon(date):
    """Return the date of the previous New Moon."""
    return _find_moon_phase(date, -twopi, 0)


def next_new_moon(date):
    """Return the date of the next New Moon."""
    return _find_moon_phase(date, twopi, 0)


def previous_first_quarter_moon(date):
    """Return the date of the previous First Quarter Moon."""
    return _find_moon_phase(date, -twopi, halfpi)


def next_first_quarter_moon(date):
    """Return the date of the next First Quarter Moon."""
    return _find_moon_phase(date, twopi, halfpi)


def previous_full_moon(date):
    """Return the date of the previous Full Moon."""
    return _find_moon_phase(date, -twopi, pi)


def next_full_moon(date):
    """Return the date of the next Full Moon."""
    return _find_moon_phase(date, twopi, pi)


def previous_last_quarter_moon(date):
    """Return the date of the previous Last Quarter Moon."""
    return _find_moon_phase(date, -twopi, pi + halfpi)


def next_last_quarter_moon(date):
    """Return the date of the next Last Quarter Moon."""
    return _find_moon_phase(date, twopi, pi + halfpi)


def next_pass(self, body, singlepass=True):
    raise NotImplementedError(
        "next_pass() is part of upstream's EarthSatellite support, which "
        "this port does not cover; see README.md"
    )


Observer.next_pass = next_pass


# ------------------------------------------------------------------
# Time conversion
def _convert_to_seconds_and_microseconds(value):
    microseconds = int(round(24 * 60 * 60 * 1000000 * _parse_mjd(value)))
    seconds, microseconds = divmod(microseconds, 1000000)
    seconds -= 2209032000  # difference between epoch 1900 and epoch 1970
    return seconds, microseconds


def localtime(value):
    """Convert a Date into naive local time, as a Python datetime."""
    seconds, microseconds = _convert_to_seconds_and_microseconds(value)
    y, m, d, H, M, S, wday, yday, isdst = _localtime(seconds)
    return _datetime(y, m, d, H, M, S, microseconds)


def _convert_to_seconds(value):
    microseconds = int(round(24 * 60 * 60 * 1000000 * _parse_mjd(value)))
    seconds, microseconds = divmod(microseconds, 1000000)
    seconds -= 2209032000
    return seconds


class _UTC(_tzinfo):
    """The UTC timezone, as a tzinfo."""

    ZERO = _timedelta(0)

    def utcoffset(self, dt):
        return self.ZERO

    def dst(self, dt):
        return self.ZERO

    def tzname(self, dt):
        return "UTC"

    def __repr__(self):
        return "<ephem.UTC>"


UTC = _UTC()  # upstream exposes the instance, not the class


def to_timezone(value, tzinfo):
    """Convert a Date into a timezone-aware Python datetime."""
    seconds = _convert_to_seconds(value)
    return _datetime.fromtimestamp(seconds, tzinfo)


def now():
    """The current date, as a Date."""
    return Date()


# ------------------------------------------------------------------
# Coordinate transformations
class Coordinate:
    def __init__(self, *args, **kw):
        epoch = kw.pop("epoch", None)
        if epoch is not None:
            self.epoch = Date(epoch)
        if kw:
            raise TypeError(
                '"epoch" is the only keyword argument you can use during '
                "%s instantiation" % type(self).__name__
            )

        if len(args) == 1:
            a = args[0]
            if isinstance(a, Body):
                a = Equatorial(a.a_ra, a.a_dec, epoch=a.a_epoch)
            for cls in (Equatorial, Ecliptic):
                if isinstance(a, cls):
                    if epoch is None:
                        self.epoch = epoch = a.epoch
                    if isinstance(self, cls) and epoch == a.epoch:
                        self.set(*a.get())
                        return
                    ra, dec = a.to_radec()
                    if epoch != a.epoch:
                        ra, dec = precess(a.epoch, epoch, ra, dec)
                    self.from_radec(ra, dec)
                    return
            raise TypeError(
                "a single argument used to initialize %s() must be either a "
                "coordinate or a Body, not an %r" % (type(self).__name__,)
            )
        elif len(args) == 2:
            self.set(*args)
            if epoch is None:
                self.epoch = Date(J2000)
        else:
            raise TypeError(
                "to initialize %s you must pass either a Body, another "
                "coordinate, or two coordinate values, but not: %r"
                % (type(self).__name__, args)
            )


class Equatorial(Coordinate):
    """An equatorial sky coordinate in right ascension and declination."""

    def get(self):
        return self.ra, self.dec

    def set(self, ra, dec):
        self.ra, self.dec = hours(ra), degrees(dec)

    to_radec = get
    from_radec = set

    def __repr__(self):
        return "<Equatorial %s %s at epoch %s>" % (self.ra, self.dec, self.epoch)


class LonLatCoordinate(Coordinate):
    """A coordinate that is measured with a longitude and latitude."""

    def set(self, lon, lat):
        self.lon, self.lat = degrees(lon), degrees(lat)

    def get(self):
        return self.lon, self.lat

    @property
    def long(self):
        return self.lon

    @long.setter
    def long(self, value):
        self.lon = value

    def __repr__(self):
        return "<%s %s %s at epoch %s>" % (
            type(self).__name__, self.lon, self.lat, self.epoch
        )


class Ecliptic(LonLatCoordinate):
    """An ecliptic latitude and longitude."""

    def to_radec(self):
        return ecl_eq(self.epoch, self.lon, self.lat)

    def from_radec(self, ra, dec):
        self.lon, self.lat = eq_ecl(self.epoch, ra, dec)


LongLatCoordinate = LonLatCoordinate
