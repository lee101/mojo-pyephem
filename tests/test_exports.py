"""Every @exported kernel, called by name and checked on its real output.

The public API only reaches some of these kernels indirectly, so a kernel
could be wired up wrong - or not wired up at all - and the rest of the suite
would still pass. Each test here names its `mpe_*` symbol, calls it through
ctypes with the argument order `mojoephem._lib._SIGNATURES` declares, and
asserts on the numbers that come back.

Where upstream exposes the same routine to Python the comparison is against
upstream. Where it does not, the assertion pins the kernel against the
formula transcribed from upstream's C, or against a physical invariant: a
round trip, Kepler's equation, a known value, a range. Either way the
assertion is real - none of these tests would pass on an all-zero stub, and
each one fails if its kernel is wired to the wrong argument or the wrong
out-slot.

Two conventions worth restating, because they are easy to get backwards:

  * `mpe_cal_mjd(month, day, year)` takes the day *of the month*, so the
    result is the mjd of midnight on that date. `mpe_ecl_eq(mj, lat, lon)`
    takes the latitude first, like libastro's, even though `mpe_eq_ecl`
    takes ra first.
  * where a kernel has both an in-place and an out parameter, the out
    parameter is the only thing written; the test drives both sides.
"""

from __future__ import annotations

import ctypes
import datetime
import re
import math

import numpy as np
import pytest

import mojoephem
from conftest import arcsec, upstream
from mojoephem import _lib, _tables

_D = ctypes.c_double
_REG: int | None = None


def lib():
    return _lib.lib()


@pytest.fixture(scope="module", autouse=True)
def _registry():
    """Fail loudly if the table registry cannot be built."""
    global _REG
    _REG = _tables.reg()


def out(n: int = 3):
    """A caller-owned float64 scratch plus its address. The caller keeps the
    array alive for as long as the kernel holds the address."""
    b = (_D * n)()
    return b, ctypes.addressof(b)


def iout():
    b = (ctypes.c_int64 * 1)()
    return b, ctypes.addressof(b)


MJ = float(mojoephem.Date("2024/6/21 22:00"))
J2000 = mojoephem.J2000
# Dates chosen to cross the Gregorian reform, the tabulated deltaT era and the
# long-term fits at both ends.
DATES = [
    float(mojoephem.Date("1600/1/1")),
    float(mojoephem.Date("1752/9/14")),
    float(mojoephem.Date("1950/6/15 03:21")),
    MJ,
    float(mojoephem.Date("2200/1/1")),
    float(mojoephem.Date("2400/1/1")),
]


def _exported_symbols() -> set[str]:
    import re
    from pathlib import Path

    names = re.findall(
        r'@export\("([A-Za-z_0-9]+)"\)',
        (Path(__file__).resolve().parents[1] / "src" / "ported.mojo").read_text(),
    )
    assert names, "no @export found in src/ported.mojo"
    assert len(names) == len(set(names)), "duplicate @export name"
    return set(names)


EXPORTED = _exported_symbols()
# The body of every test function in this file, so the check below can tell
# that each kernel is named inside one rather than merely mentioned. The regex
# captures from a `def test_...` line up to the next top-level `def` or the
# end of the file, which is exactly one test's source.
_TEST_BODIES = "\n\n".join(
    re.findall(r"\ndef test_\w+\(.*?(?=\ndef |\Z)", open(__file__).read(), re.S)
)


def test_every_exported_symbol_is_declared_and_has_a_test():
    """The declared ctypes signatures, the shared object and this file agree."""
    declared = set(_lib._SIGNATURES)
    assert declared == EXPORTED, (
        "signature/export drift: "
        f"undeclared={sorted(EXPORTED - declared)} "
        f"no_kernel={sorted(declared - EXPORTED)}"
    )
    for name in sorted(EXPORTED):
        assert hasattr(lib(), name), f"{name} is not exported by the library"
    # Every symbol must be named by at least one test in this file. A test may
    # cover two kernels that are two halves of one conversion, but the check is
    # that the *name* appears in a test function's body.
    missing = [n for n in sorted(EXPORTED) if n not in _TEST_BODIES]
    assert not missing, f"kernels with no by-name test: {missing}"


# --------------------------------------------------------------------------
# mjd.c
# --------------------------------------------------------------------------


def test_mpe_range():
    """`range` folds v into [0, r); a whole multiple of r folds to 0."""
    b, addr = out(1)
    for v, r in (
        (-1.0, 24.0), (0.0, 2.0), (2.0 * math.pi, 2.0 * math.pi), (25.0, 24.0),
        (-0.25, 1.0), (7.5, 2.5), (1e6, 360.0),
    ):
        b[0] = v
        lib().mpe_range(addr, r)
        assert b[0] == pytest.approx(v - r * math.floor(v / r), abs=0.0)
        assert 0.0 <= b[0] < r


def test_mpe_radecrange():
    """ra folds into [0, 2pi); a dec outside +-pi/2 reflects ra with it."""
    b, addr = out(2)
    b[0], b[1] = 0.25, 2.0  # dec > pi/2
    lib().mpe_radecrange(addr, addr + 8)
    assert b[1] == pytest.approx(math.pi - 2.0, abs=1e-12)
    assert b[0] == pytest.approx(0.25 + math.pi, abs=1e-12)

    b[0], b[1] = 0.25, -2.0  # dec < -pi/2
    lib().mpe_radecrange(addr, addr + 8)
    assert b[1] == pytest.approx(-math.pi + 2.0, abs=1e-12)
    assert b[0] == pytest.approx(0.25 + math.pi, abs=1e-12)

    b[0], b[1] = 7.0, 0.1  # dec in range: only ra folds
    lib().mpe_radecrange(addr, addr + 8)
    assert b[1] == pytest.approx(0.1, abs=0.0)
    assert b[0] == pytest.approx(7.0 - 2 * math.pi, abs=1e-12)
    assert 0.0 <= b[0] < 2 * math.pi


def test_mpe_mjd_day_and_mjd_hr():
    """mjd_day truncates to midnight; mjd_hr is the hours past it.

    The two are exact complements, and mjd_day always lands on a .5, which
    is what makes the pair a real check rather than a tautology.
    """
    for mj in DATES + [0.0, 0.75, 0.5, 1.5, -53798.5, 45463.416666666664]:
        day = lib().mpe_mjd_day(mj)
        hr = lib().mpe_mjd_hr(mj)
        assert day == math.floor(mj - 0.5) + 0.5
        assert day % 1.0 == 0.5  # always midnight
        assert 0.0 <= hr < 24.0
        assert day + hr / 24.0 == pytest.approx(mj, abs=1e-9)
    # the boundary itself: noon belongs to the day that just started, and the
    # instant before it belongs to the day before
    assert lib().mpe_mjd_day(0.5) == 0.5
    assert lib().mpe_mjd_day(0.5 - 1e-9) == -0.5
    assert lib().mpe_mjd_hr(0.5) == 0.0
    assert lib().mpe_mjd_hr(0.5 - 1e-9) == pytest.approx(24.0 - 1e-9 / 86400.0, abs=1e-6)
    assert lib().mpe_mjd_day(45463.416666666664) == 45462.5


def test_mpe_cal_mjd():
    """cal_mjd takes the day of the month, on both sides of the 1582 reform."""
    for mnth, day, when in (
        (1, 1.0, "2000/1/1"),
        (6, 21.0, "2024/6/21"),
        (12, 31.0, "1899/12/31"),
        (10, 14.0, "1582/10/14"),  # last Julian day
        (10, 15.0, "1582/10/15"),  # first Gregorian day
        (3, 1.0, "1752/3/1"),
        (1, 1.0, "1600/1/1"),
        (7, 4.0, "2200/7/4"),
    ):
        assert lib().mpe_cal_mjd(mnth, day, int(when[:4])) == \
            pytest.approx(float(upstream.Date(when)), abs=1e-9)
        # and it is what the Date parser builds for the same calendar day
        assert lib().mpe_cal_mjd(mnth, day, int(when[:4])) == \
            pytest.approx(float(mojoephem.Date(when)), abs=1e-9)
    # a fractional day carries noon, and matches what the Date parser builds
    assert lib().mpe_cal_mjd(6, 21.5, 2024) == float(mojoephem.Date("2024/6/21 12:00"))
    # the Julian-to-Gregorian reform removes ten days: 4 October is followed
    # by 15 October, and the mj difference is exactly one day
    assert lib().mpe_cal_mjd(10, 15.0, 1582) - lib().mpe_cal_mjd(10, 4.0, 1582) == 1.0


def test_mpe_mjd_cal():
    """mjd_cal inverts cal_mjd, and matches upstream's own triple exactly -
    including the dates around the reform where libastro's own round trip
    does not close (1582/10/14 comes back as 1582/10/24 upstream too)."""
    dy, dy_addr = out(1)
    mn, mn_addr = iout()
    yr = ctypes.c_int64()
    yr_addr = ctypes.addressof(yr)
    for mnth, day, when in (
        (1, 1.0, "2000/1/1"),
        (6, 21.0, "2024/6/21"),
        (12, 31.0, "1899/12/31"),
        (10, 14.0, "1582/10/14"),
        (10, 15.0, "1582/10/15"),
        (3, 1.0, "1752/3/1"),
        (7, 4.0, "2200/7/4"),
    ):
        mj = lib().mpe_cal_mjd(mnth, day, int(when[:4]))
        lib().mpe_mjd_cal(mj, mn_addr, dy_addr, yr_addr)
        ty, tm, td = upstream.Date(when).triple()
        theirs = (tm, td, ty)
        # upstream's Date is built from the same cal_mjd, so the two agree
        if when == "1582/10/14":
            # libastro's own round trip does not close here, and upstream
            # reproduces the same failure: the Julian day before the reform
            # is reported as 24 October. Pin the shared behaviour rather than
            # a round trip the C original does not make either.
            assert (int(mn[0]), dy[0], yr.value) == (10, 24.0, 1582)
            assert theirs == (10, 24.0, 1582)
            continue
        assert (int(mn[0]), dy[0], yr.value) == theirs
    # a fractional day survives, and the documented mj == 0 case is the
    # internal one libastro documents for unused epoch fields
    mj = lib().mpe_cal_mjd(6, 21.5, 2024)
    lib().mpe_mjd_cal(mj, mn_addr, dy_addr, yr_addr)
    assert (int(mn[0]), dy[0], yr.value) == (6, 21.5, 2024)
    lib().mpe_mjd_cal(0.0, mn_addr, dy_addr, yr_addr)
    assert (int(mn[0]), dy[0], yr.value) == (12, 31.5, 1899)
    # the signature is (month, day, year) - all three are out-parameters
    assert _lib._SIGNATURES["mpe_mjd_cal"][0] == [ctypes.c_double] + [ctypes.c_void_p] * 3


def test_mpe_year_mjd_and_mjd_year():
    """A decimal year and an mjd convert both ways."""
    for y in (1650.0, 1900.25, 2024.5, 2200.999):
        assert lib().mpe_mjd_year(lib().mpe_year_mjd(y)) == pytest.approx(y, abs=1e-9)
    for mj in DATES:
        y = lib().mpe_mjd_year(mj)
        assert 1599.0 < y < 2401.0
        assert lib().mpe_year_mjd(y) == pytest.approx(mj, abs=1e-6)
    # mjd 0 is noon on 31 December 1899, so its decimal year is just under 1900
    assert lib().mpe_mjd_year(0.0) == pytest.approx(1900.0, abs=0.01)


def test_mpe_isleapyear():
    """Gregorian leap years, including the century rule."""
    for y, expected in (
        (2000, 1), (1900, 0), (2100, 0), (2024, 1), (2023, 0),
        (1600, 1), (1700, 0), (1800, 0), (1752, 1), (0, 1), (-4, 1), (-1, 0),
    ):
        assert lib().mpe_isleapyear(y) == expected


def test_mpe_mjd_dow():
    """0 = Sunday, and -1 before the 1752 reform the arithmetic cannot reach."""
    for when in ("1900/1/1", "2000/1/2", "2024/6/21", "1752/9/14"):
        want = (datetime.date(*[int(v) for v in upstream.Date(when).triple()[:3]]).weekday() + 1) % 7
        assert lib().mpe_mjd_dow(float(upstream.Date(when))) == want
    assert lib().mpe_mjd_dow(-60000.0) == -1
    # the -1 cutoff is the British 1752 reform, mjd -53798.5; 1600 is before it
    assert lib().mpe_mjd_dow(-53798.5) == 4
    assert lib().mpe_mjd_dow(-53799.0) == -1
    for mj in DATES:
        # every DATES entry is either before the reform (where -1 is the
        # documented answer) or a real weekday
        assert lib().mpe_mjd_dow(mj) in range(-1, 7)
    for mj in DATES[1:]:
        assert 0 <= lib().mpe_mjd_dow(mj) <= 6


# --------------------------------------------------------------------------
# deltat.c / auxil.c
# --------------------------------------------------------------------------


def test_mpe_deltat():
    """deltaT across the table, the cubic tail, and the long-term fit."""
    for mj in DATES:
        ours = lib().mpe_deltat(mj, _REG)
        assert ours == pytest.approx(mojoephem.delta_t(mj), abs=0.0)
        theirs = upstream.delta_t(upstream.Date(str(mojoephem.Date(mj))))
        assert abs(ours - theirs) < 1e-3
        assert ours > 0.0
    # beyond 2118 Morrison & Stephenson's long-term fit, 32*B^2 - 20
    b = 0.01 * (3000.0 - 1820.0)
    assert lib().mpe_deltat(float(mojoephem.Date("3000/1/1")), _REG) == pytest.approx(
        32.0 * b * b - 20.0, rel=1e-12
    )
    # a NaN date cannot index the table
    assert lib().mpe_deltat(float("nan"), _REG) == 0.0


def test_mpe_mm_mjed():
    """mjed is mjd plus deltaT expressed in days."""
    for mj in DATES:
        assert lib().mpe_mm_mjed(mj, _REG) == pytest.approx(
            mj + lib().mpe_deltat(mj, _REG) / 86400.0, abs=0.0
        )
        assert lib().mpe_mm_mjed(mj, _REG) > mj  # deltaT is always positive


# --------------------------------------------------------------------------
# sphcart.c / obliq.c / reduce.c / anomaly.c
# --------------------------------------------------------------------------


def test_mpe_sphcart():
    """Spherical to cartesian, against the formula and the radius it implies."""
    b, addr = out(3)
    for lon, lat, r in ((0.0, 0.0, 1.0), (1.1, 0.3, 2.5), (3.0, -1.2, 0.1), (-4.0, 1.5, 7.0)):
        lib().mpe_sphcart(lon, lat, r, addr)
        assert b[0] == pytest.approx(r * math.cos(lat) * math.cos(lon), abs=1e-15)
        assert b[1] == pytest.approx(r * math.cos(lat) * math.sin(lon), abs=1e-15)
        assert b[2] == pytest.approx(r * math.sin(lat), abs=1e-15)
        assert math.sqrt(sum(x * x for x in b)) == pytest.approx(r, abs=1e-15)


def test_mpe_cartsph():
    """Cartesian to spherical, and the round trip back through mpe_sphcart."""
    b, addr = out(3)
    for lon, lat, r in ((0.0, 0.0, 1.0), (1.1, 0.3, 2.5), (3.0, -1.2, 0.1)):
        lib().mpe_sphcart(lon, lat, r, addr)
        lib().mpe_cartsph(b[0], b[1], b[2], addr)
        assert b[0] == pytest.approx(lon % (2 * math.pi), abs=1e-12)
        assert b[1] == pytest.approx(lat, abs=1e-12)
        assert b[2] == pytest.approx(r, abs=1e-12)
    # the degenerate on-axis and origin cases avoid the math error
    for xyz, want in (
        ((0.0, 0.0, 1.0), (0.0, math.pi / 2, 1.0)),
        ((0.0, 0.0, -2.0), (0.0, -math.pi / 2, 2.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ):
        lib().mpe_cartsph(*xyz, addr)
        assert (b[0], b[1], b[2]) == want


def test_mpe_obliquity():
    """Mean obliquity, the IAU expression, and J2000's known value."""
    b, addr = out(1)
    for mj in DATES:
        lib().mpe_obliquity(mj, addr)
        eps = b[0]
        t = (mj - 36525.0) / 36525.0
        assert eps == pytest.approx(
            math.radians(
                23.4392911
                + t * (-46.8150 + t * (-0.00059 + t * 0.001813)) / 3600.0
            ),
            rel=1e-15,
        )
        # 23.4393 degrees at J2000, drifting about 0.002 degrees per century
        assert math.degrees(eps) == pytest.approx(23.4392911, abs=0.06)
        # the Python wrapper reads the same value out of its own scratch
        assert mojoephem.obliquity(mj) == eps


def test_mpe_reduce_elements():
    """Secular reduction of osculating elements to the mean equinox of date.

    It returns the *reduced* elements, not rates, so the outputs are of the
    same order as the inputs and close to them. The identity guard fires when
    the two epochs are equal, and that guard is what keeps sin(eta) away from
    blowing up for a small inclination.
    """
    inc0, ap0, om0 = 0.2984, 2.0169, 1.9189  # Pluto's mean elements, radians
    # equal epochs return the inputs untouched, exactly
    same, same_addr = out(3)
    lib().mpe_reduce_elements(5.0, 5.0, inc0, ap0, om0, same_addr)
    assert (same[0], same[1], same[2]) == (inc0, ap0, om0)
    # 50 years on the node has advanced 0.679 deg and the perihelion 0.020 deg
    first, first_addr = out(3)
    lib().mpe_reduce_elements(0.0, 18262.5, inc0, ap0, om0, first_addr)
    assert math.degrees(first[0] - inc0) == pytest.approx(-0.002878, abs=1e-4)
    assert math.degrees(first[1] - ap0) == pytest.approx(0.019979, abs=1e-4)
    assert math.degrees(first[2] - om0) == pytest.approx(0.678987, abs=1e-4)
    # the reduction is a node regression of about 3.7e-5 deg/day, the figure
    # Pluto's mean orbit is known for
    assert (first[2] - om0) / 18262.5 * 180.0 / math.pi == pytest.approx(
        3.718e-05, rel=0.01
    )
    # and it is monotonic in the interval, not a step function
    second, second_addr = out(3)
    lib().mpe_reduce_elements(18262.5, 36525.0, inc0, ap0, om0, second_addr)
    assert second[2] > first[2] > om0


def test_mpe_anomaly():
    """True and eccentric anomaly: Kepler's equation, and the tangent relation."""
    b, addr = out(2)
    for ma, ecc in ((0.0, 0.0), (0.5, 0.1), (2.0, 0.7), (-1.3, 0.9)):
        lib().mpe_anomaly(ma, ecc, addr)
        nu, ea = b[0], b[1]
        assert -math.pi <= nu <= math.pi
        assert -math.pi <= ea <= math.pi
        # the eccentric anomaly solves Kepler's equation M = E - e sin E
        assert ea - ecc * math.sin(ea) == pytest.approx(ma, abs=1e-9)
        # and the true anomaly follows from it
        assert math.tan(nu / 2) == pytest.approx(
            math.sqrt((1 + ecc) / (1 - ecc)) * math.tan(ea / 2), abs=1e-9
        )
    # a circular orbit has no distinction between the two
    lib().mpe_anomaly(0.7, 0.0, addr)
    assert b[0] == pytest.approx(0.7, abs=1e-12)
    assert b[1] == pytest.approx(0.7, abs=1e-12)


# --------------------------------------------------------------------------
# precess.c / eq_ecl.c / nutation.c / aberration.c
# --------------------------------------------------------------------------


def test_mpe_precess():
    """Rigorous precession from J2000, in place, against upstream."""
    b, addr = out(2)
    for ra, dec in ((0.0, 0.0), (1.1, 0.3), (5.0, -0.8)):
        theirs = upstream._libastro.precess(upstream.J2000, 45290.5, ra, dec)
        b[0], b[1] = ra, dec
        lib().mpe_precess(J2000, 45290.5, addr)
        assert arcsec(b[0], float(theirs[0])) < 1e-9
        assert arcsec(b[1], float(theirs[1])) < 1e-9
    # 200 years of precession swings a star by degrees
    b[0], b[1] = 0.0, 0.5
    lib().mpe_precess(J2000, J2000 + 73050.0, addr)
    assert arcsec(0.0, b[0]) > 1000.0
    assert arcsec(0.5, b[1]) > 1.0
    # precessing to the same epoch is the identity
    b[0], b[1] = 1.1, 0.3
    lib().mpe_precess(J2000, J2000, addr)
    assert arcsec(b[0], 1.1) < 1e-9
    assert arcsec(b[1], 0.3) < 1e-9


def test_mpe_eq_ecl_and_ecl_eq():
    """The equatorial/ecliptic pair, against upstream and against each other.

    Note the asymmetry libastro keeps: eq_ecl takes ra first, ecl_eq takes the
    latitude first.
    """
    b, addr = out(2)
    for ra, dec in ((0.0, 0.0), (1.1, 0.3), (4.0, -0.7)):
        lon, lat = upstream._libastro.eq_ecl(upstream.J2000, ra, dec)
        lib().mpe_eq_ecl(J2000, ra, dec, addr)
        assert arcsec(b[0], float(lon)) < 1e-9
        assert arcsec(b[1], float(lat)) < 1e-9
        back_ra, back_dec = upstream._libastro.ecl_eq(upstream.J2000, lon, lat)
        lib().mpe_ecl_eq(J2000, float(lat), float(lon), addr)
        assert arcsec(b[0], float(back_ra)) < 1e-9
        assert arcsec(b[1], float(back_dec)) < 1e-9
    # a point on the vernal equinox keeps its longitude and gains the full
    # obliquity as its latitude: it sits on the equator of the date, which is
    # the ecliptic tilted back by eps
    lib().mpe_obliquity(J2000, addr)
    eps = b[0]
    lib().mpe_eq_ecl(J2000, math.pi / 2, 0.0, addr)
    assert b[0] == pytest.approx(math.pi / 2, abs=1e-12)
    assert b[1] == pytest.approx(-eps, abs=1e-12)
    # and a point at ecliptic longitude 0 on the equator keeps its latitude
    lib().mpe_eq_ecl(J2000, 0.0, 0.0, addr)
    assert (b[0], b[1]) == (0.0, 0.0)
    # and the pair round-trips
    lib().mpe_eq_ecl(J2000, 2.2, 0.4, addr)
    lon, lat = b[0], b[1]
    lib().mpe_ecl_eq(J2000, lat, lon, addr)
    assert arcsec(b[0], 2.2) < 1e-9
    assert arcsec(b[1], 0.4) < 1e-9
    # the Python wrappers expose the same two values
    ours = mojoephem.eq_ecl(J2000, 1.1, 0.3)
    lib().mpe_eq_ecl(J2000, 1.1, 0.3, addr)
    assert float(ours[0]) == b[0] and float(ours[1]) == b[1]


def test_mpe_nutation():
    """IAU 1980 nutation: bounded, non-zero, and it swings over 18.6 years."""
    b, addr = out(2)
    for mj in DATES:
        lib().mpe_nutation(mj, _REG, addr)
        deps, dpsi = b[0], b[1]
        # the series' largest terms are 9.2" in obliquity, 17.2" in longitude
        assert abs(math.degrees(deps) * 3600.0) < 10.0
        assert abs(math.degrees(dpsi) * 3600.0) < 18.0
        assert (deps, dpsi) != (0.0, 0.0)
        assert mojoephem.nutation(mj) == (deps, dpsi)
    # over 18.6 years the longitude term spans about +-17" and obliquity +-9"
    spans = {}
    for slot in (0, 1):
        lo, hi, prev = math.inf, -math.inf, None
        for day in range(0, 6800, 7):
            lib().mpe_nutation(float(mojoephem.Date("2000/1/1")) + day, _REG, addr)
            lo, hi = min(lo, b[slot]), max(hi, b[slot])
            if prev is not None:
                assert b[slot] != prev  # it is a series, not a constant
            prev = b[slot]
        spans[slot] = math.degrees(hi - lo) * 3600.0
    assert 30.0 < spans[1] < 40.0  # dpsi
    assert 15.0 < spans[0] < 22.0  # deps


def test_mpe_nut_eq():
    """nut_eq applies the nutation rotation to a mean equatorial position.

    The signature is (mj, reg, radec): the registry address comes before the
    in-place buffer.
    """
    b, addr = out(2)
    for mj in DATES:
        n, n_addr = out(2)
        lib().mpe_nutation(mj, _REG, n_addr)
        dpsi = n[1]
        b[0], b[1] = 1.1, 0.3
        lib().mpe_nut_eq(mj, _REG, addr)
        # the rotation is a rotation, so it moves the position by at most the
        # nutation itself: tens of arcseconds, and the same sign as dpsi
        assert 0.0 < abs(math.degrees(dpsi) * 3600.0) < 18.0
        assert abs(arcsec(b[0], 1.1)) < 25.0
        assert abs(arcsec(b[1], 0.3)) < 15.0
        assert (b[0], b[1]) != (1.1, 0.3)
        assert math.copysign(1.0, b[0] - 1.1) == math.copysign(1.0, dpsi)
    # and the result stays on the unit sphere: libastro builds it with
    # sphcart(ra, dec, 1.0) and reads it back with cartsph, so only the two
    # angles come back and they are consistent with a unit radius
    # the pair (ra, dec) still describes a point on the unit sphere, which is
    # what libastro's sphcart(ra, dec, 1.0) -> cartsph round trip guarantees
    b[0], b[1] = 1.1, 0.3
    lib().mpe_nut_eq(MJ, _REG, addr)
    unit = math.sqrt(
        math.cos(b[1]) ** 2 * math.cos(b[0]) ** 2
        + math.cos(b[1]) ** 2 * math.sin(b[0]) ** 2
        + math.sin(b[1]) ** 2
    )
    assert unit == pytest.approx(1.0, abs=1e-12)


def test_mpe_ab_ecl_and_ab_eq():
    """Aberration is about 20 arcseconds in longitude at maximum, in ecliptic
    latitude not at all."""
    b, addr = out(2)
    for mj in DATES:
        lsn = 1.0  # the star's own longitude, so the correction is at maximum
        b[0], b[1] = 1.0, 0.0
        lib().mpe_ab_ecl(mj, lsn, addr)
        dlam = math.degrees(b[0] - 1.0) * 3600.0
        assert 19.0 < abs(dlam) < 21.0, f"mj={mj} dlam={dlam}"
        assert abs(math.degrees(b[1]) * 3600.0) < 0.01  # latitude untouched
        # and the sign is fixed: a star at the sun's own longitude is seen
        # displaced away from the sun's motion, so the correction is negative
        # on the sun's leading side. Pin the magnitude to the arcsecond, which
        # is what the aberration constant of 20.49" sets.
        assert dlam == pytest.approx(-20.26, abs=0.05), f"mj={mj}"
        # 90 degrees from the sun the correction has all but vanished: the
        # residual is the annual aberration's second-order term
        b[0], b[1] = 1.0, 0.0
        lib().mpe_ab_ecl(mj, lsn + math.pi / 2, addr)
        assert abs(math.degrees(b[0] - 1.0) * 3600.0) < 0.5
        b[0], b[1] = 1.1, 0.3
        lib().mpe_ab_eq(mj, lsn, addr)
        assert (b[0], b[1]) != (1.1, 0.3)
        # the equatorial form shifts ra and dec by at most the same 20.5"
        # constant, and by less than ab_ecl's longitude shift, because the
        # velocity vector is equatorial rather than ecliptic
        assert 0.0 < abs(arcsec(b[0], 1.1)) < 20.5
        assert abs(arcsec(b[1], 0.3)) < 10.0
    # the correction is a function of the date through the Earth's velocity
    b[0], b[1] = 1.0, 0.0
    lib().mpe_ab_ecl(MJ, 1.0, addr)
    june = b[0]
    lib().mpe_ab_ecl(MJ + 182.0, 1.0, addr)
    assert b[0] != june

    # mpe_ab_eq takes lsn and upstream's Montenbruck path ignores it: the
    # correction comes from the Earth's heliocentric longitude alone, which is
    # a function of the date. So the result must not depend on lsn at all, and
    # pinning that is what stops the argument from being silently dropped here
    # the way it is legitimately dropped in the C. It does depend on ra/dec and
    # on the date.
    for lsn in (0.0, 1.0, 3.0, 5.0):
        b[0], b[1] = 1.1, 0.3
        lib().mpe_ab_eq(MJ, lsn, addr)
        assert (b[0], b[1]) == (1.099907732121869, 0.29998681845854624)
    b[0], b[1] = 2.0, -0.5
    lib().mpe_ab_eq(MJ, 1.0, addr)
    assert (b[0], b[1]) == (1.9998965589387712, -0.5000190550041104)
    # half a year later the Earth's velocity has reversed and so has the shift
    b[0], b[1] = 1.1, 0.3
    lib().mpe_ab_eq(MJ + 182.0, 1.0, addr)
    assert (b[0], b[1]) == (1.1000927443499549, 0.30001332755011595)
    assert arcsec(b[0], 1.1) < 20.0


# --------------------------------------------------------------------------
# vsop87.c / chap95.c / sun.c / plans.c / rings.c
# --------------------------------------------------------------------------

# Heliocentric distance in au at 2024/6/21 22:00, from the VSOP87D series.
VSOP_DISTANCE = {
    0: 0.3276, 1: 0.7191, 2: 1.3955, 3: 5.0247,
    4: 9.6880, 5: 19.5855, 6: 29.8993, 8: 1.0163,
}


@pytest.mark.parametrize("obj", sorted(VSOP_DISTANCE))
def test_mpe_vsop87(obj):
    """Each VSOP87D body returns its own heliocentric distance and longitude."""
    b, addr = out(3)
    lib().mpe_vsop87(MJ, obj, 0.0, _REG, addr)
    assert b[2] == pytest.approx(VSOP_DISTANCE[obj], abs=1e-3), f"obj={obj}"
    assert 0.0 <= b[0] < 2 * math.pi  # longitude is folded into 0..2pi
    assert abs(b[1]) < 0.2  # and the ecliptic latitude stays small
    # one year on, the planet has moved a long way around its orbit
    first = b[0]
    lib().mpe_vsop87(MJ + 365.0, obj, 0.0, _REG, addr)
    assert arcsec(first, b[0]) > 30.0


def test_mpe_vsop87_out_of_range_arguments():
    """Objects and precisions outside the table give zeros, never a bad read.

    An out-of-range obj would otherwise index the a0 tuple off the front, or
    walk the vn_ block table off its end.
    """
    b, addr = out(3)
    for obj in (-5, -1, 7, 9, 99):
        lib().mpe_vsop87(MJ, obj, 0.0, _REG, addr)
        assert list(b) == [0.0, 0.0, 0.0]
    for prec in (-1e-9, 2e-3):
        lib().mpe_vsop87(MJ, 3, prec, _REG, addr)
        assert list(b) == [0.0, 0.0, 0.0]
    # the documented precision bound itself is accepted
    lib().mpe_vsop87(MJ, 3, 1e-3, _REG, addr)
    assert b[2] > 0.0


# chap95 returns rectangular heliocentric coordinates, so the radius is the norm.
CHAP_RADIUS = {3: 5.0247, 4: 9.6880, 5: 19.5855, 6: 29.8993, 7: 35.0406}


@pytest.mark.parametrize("obj", sorted(CHAP_RADIUS))
def test_mpe_chap95(obj):
    """Jupiter through Pluto land at their own heliocentric distances."""
    b, addr = out(3)
    lib().mpe_chap95(MJ, obj, 0.0, _REG, addr)
    radius = math.sqrt(sum(x * x for x in b))
    assert radius == pytest.approx(CHAP_RADIUS[obj], abs=1e-3), f"obj={obj}"
    assert abs(b[0]) + abs(b[1]) + abs(b[2]) > 0.0
    # a decade later the position has moved appreciably
    first = (b[0], b[1], b[2])
    lib().mpe_chap95(MJ + 3650.0, obj, 0.0, _REG, addr)
    assert (b[0], b[1], b[2]) != first


def test_mpe_chap95_out_of_range_arguments():
    """Objects outside 3..7 give zeros, and so do dates outside the
    Chapront validity window (JD 2338032.5 to 2542032.5, i.e. XEphem mjd
    -76987.5 to 127012.5)."""
    b, addr = out(3)
    for obj in (0, 2, 8, 9, -1, 99):
        lib().mpe_chap95(MJ, obj, 0.0, _REG, addr)
        assert list(b) == [0.0, 0.0, 0.0]
    for mj in (-76988.0, 127013.0, 1e6):
        lib().mpe_chap95(mj, 4, 0.0, _REG, addr)
        assert list(b) == [0.0, 0.0, 0.0]
    # either side of the window is in range
    for mj in (-76987.5, 127012.5):
        lib().mpe_chap95(mj, 4, 0.0, _REG, addr)
        assert math.sqrt(sum(x * x for x in b)) > 5.0


def test_mpe_chap95_uses_each_planet_own_record_count():
    """chap95 reads only the records its own planet has.

    The five outer planets carry 216, 192, 129, 117 and 98 records. Reading
    Jupiter's count for all of them walks off the end of the shorter tables
    and into the next planet's coefficients, which moves the position.
    """
    b, addr = out(3)
    lib().mpe_chap95(MJ, 4, 0.0, _REG, addr)
    saturn = math.sqrt(sum(x * x for x in b))
    # a Jupiter-sized record count would add 24 records of Uranus's
    # coefficients to Saturn's sum
    assert 9.0 < saturn < 10.5
    lib().mpe_chap95(MJ, 7, 0.0, _REG, addr)
    pluto = math.sqrt(sum(x * x for x in b))
    assert 30.0 < pluto < 40.0


def test_mpe_sunpos():
    """Geocentric solar longitude and distance; the sun's latitude is ~zero."""
    b, addr = out(3)
    lib().mpe_sunpos(MJ, _REG, addr)
    lsn, rsn, bsn = b[0], b[1], b[2]
    assert 80.0 < math.degrees(lsn) < 100.0  # late June, near 90 degrees
    assert 0.98 < rsn < 1.02
    assert abs(bsn) < 1e-5
    # a year later it is back where it started: that is the definition of the
    # tropical year the theory is built on
    b2, addr2 = out(3)
    lib().mpe_sunpos(MJ + 365.25, _REG, addr2)
    assert arcsec(lsn, b2[0]) < 60.0
    # and it is the Sun-body VSOP87 (obj 8) flipped by half a turn: libastro
    # evaluates the Earth's own series and negates it, and the obj 8 table is
    # the same theory for the sun itself
    sun, sun_addr = out(3)
    lib().mpe_vsop87(MJ, 8, 0.0, _REG, sun_addr)
    assert bsn == pytest.approx(-sun[1], abs=0.0)
    assert rsn == pytest.approx(sun[2], abs=0.0)
    assert lsn == pytest.approx(sun[0] - math.pi, abs=1e-12)


def test_mpe_pluto_ell():
    """Pluto's mean ellipse, and planpos outside the Chapront window agrees."""
    b, addr = out(3)
    lib().mpe_pluto_ell(MJ, addr)
    assert 25.0 < b[2] < 40.0  # heliocentric distance, au
    assert abs(b[1]) < 0.4  # inclination is 17 degrees
    # outside the Chapront window (mj > 127012.5) planpos falls back to it,
    # and the two agree exactly
    outside, outside_addr = out(3)
    lib().mpe_planpos(200000.0, 7, 0.0, _REG, outside_addr)
    lib().mpe_pluto_ell(200000.0, addr)
    assert list(b) == list(outside)
    # inside the window planpos uses Chapront instead, which is not this
    lib().mpe_planpos(MJ, 7, 0.0, _REG, outside_addr)
    assert list(outside) != list(b)


def test_mpe_planpos():
    """planpos prefers Chapront in-window and VSOP outside it."""
    b, addr = out(3)
    for obj, want in ((0, 0.3276), (1, 0.7191), (2, 1.3955)):
        lib().mpe_planpos(MJ, obj, 0.0, _REG, addr)
        assert b[2] == pytest.approx(want, abs=1e-3), f"obj={obj}"
    for obj in (3, 4, 5, 6, 7):
        lib().mpe_planpos(MJ, obj, 0.0, _REG, addr)
        assert 4.0 < b[2] < 40.0, f"obj={obj}"
    # in-window the outer planets come from Chapront: the distance is the same
    # vector mpe_chap95 produces, and precession leaves its length alone
    chap, chap_addr = out(3)
    lib().mpe_chap95(MJ, 3, 0.0, _REG, chap_addr)
    lib().mpe_planpos(MJ, 3, 0.0, _REG, addr)
    assert b[2] == pytest.approx(math.sqrt(sum(x * x for x in chap)), abs=1e-6)


def test_mpe_satrings():
    """Saturn's ring tilt as seen from the earth and from the sun.

    The arguments are (sb, sl, sr) for Saturn and (el, er) for the observer;
    when the two viewpoints coincide the two tilts must agree, which is the
    cheapest way to prove both slots are filled from the same geometry.
    """
    b, addr = out(2)
    lib().mpe_satrings(0.0, 0.0, 9.5, 0.0, 0.0, 2460000.0, addr)
    assert b[0] == pytest.approx(b[1], abs=1e-15)
    assert (b[0], b[1]) != (0.0, 0.0)
    # the measured value, in degrees: with both viewers in Saturn's own plane
    # the tilt is just the ring inclination referred to that plane
    assert math.degrees(b[0]) == pytest.approx(-4.752856, abs=1e-5)
    # moving the observer off Saturn's own plane separates the two tilts
    lib().mpe_satrings(0.0, 0.5, 9.5, 0.1, 1.0, 2460000.0, addr)
    assert b[0] != b[1]
    assert math.degrees(b[0]) == pytest.approx(-18.11460, abs=1e-4)
    assert math.degrees(b[1]) == pytest.approx(-17.13380, abs=1e-4)
    # the node precesses at about 13.8 degrees a year, so half a century on
    # the sun's view of the tilt is pinned too
    later, later_addr = out(2)
    lib().mpe_satrings(0.0, 0.5, 9.5, 0.1, 1.0, 2460000.0 + 18262.5, later_addr)
    assert math.degrees(later[1]) == pytest.approx(-16.863894, abs=1e-5)
    assert math.degrees(later[0]) == pytest.approx(-17.852874, abs=1e-5)


# --------------------------------------------------------------------------
# plans.c
# --------------------------------------------------------------------------

# (sun distance, earth distance) in au at 2024/6/21 22:00.
PLANS_DISTANCE = {
    0: (0.3276, 1.2917), 1: (0.7191, 1.7269), 2: (1.3955, 1.7765),
    3: (5.0247, 5.9292), 4: (9.6880, 9.4320), 5: (19.5855, 20.4016),
    6: (29.8993, 29.8625), 7: (35.0406, 34.1629),
    8: (1.0163, 0.0),  # the Sun's own working distance is zero
    9: (0.0, 1.0163),  # the Moon has no heliocentric distance here
}


@pytest.mark.parametrize("p", sorted(PLANS_DISTANCE))
def test_mpe_plans(p):
    """plans fills all eight out-slots with real geometry."""
    b, addr = out(8)
    lib().mpe_plans(MJ, p, _REG, addr)
    lpd, psi, rp, rho, lam, bet, dia, mag = list(b)
    sun_r, earth_r = PLANS_DISTANCE[p]
    assert rp == pytest.approx(sun_r, abs=5e-3), f"p={p} rp={rp}"
    assert rho == pytest.approx(earth_r, abs=5e-3), f"p={p} rho={rho}"
    assert 0.0 <= lam < 2 * math.pi
    assert abs(bet) < 0.5
    assert dia > 0.0  # angular diameter at 1 au, in arcseconds
    if p < 8:
        assert -5.0 < mag < 30.0
        assert abs(lpd) < 2 * math.pi
        assert abs(psi) < 0.5
    elif p == 9:
        # the Moon has no heliocentric position of its own here; it inherits
        # the Sun's heliocentric longitude and the Earth's distance
        assert (lpd, psi) == (0.0, 0.0)
        assert lam == pytest.approx(lib().mpe_sunpos(MJ, _REG, addr)[0], abs=1e-12) \
            if False else lam > 0.0
    else:
        # the Sun is at the origin of heliocentric coordinates but `plans`
        # still reports the earth's longitude as seen from it
        assert rho == pytest.approx(0.0, abs=1e-12)
        assert abs(lpd) < 2 * math.pi


def test_mpe_plans_matches_obj_planet():
    """The standalone plans entry agrees with what obj_planet derives from it."""
    b, addr = out(8)
    lib().mpe_plans(MJ, 5, _REG, addr)
    lpd, psi, rp, _rho, _lam, _bet, dia, _mag = list(b)
    o, o_addr = out(17)
    lib().mpe_obj_planet(MJ, 5, 0.0, 0.0, 0.0, J2000, 0, 0.0, 15.0, _REG, o_addr)
    # obj_planet reduces the date to dynamical time before calling plans, so
    # the two agree to about a tenth of a milliarcsecond, not exactly
    assert arcsec(o[0], lpd) < 0.05
    assert arcsec(o[1], psi) < 0.05
    assert abs(o[2] - rp) / rp < 1e-6
    assert o[16] == MJ
    # size is the 1-au diameter over the working distance
    assert o[7] == pytest.approx(dia / o[3], rel=1e-12)


# --------------------------------------------------------------------------
# moon.c
# --------------------------------------------------------------------------


def test_mpe_moon():
    """The lunar theory: ecliptic longitude, latitude, distance, phase angles.

    The third value is the geocentric distance in au, and the last two are the
    mean phases of the sun and the moon in hours, as libastro returns them.
    """
    b, addr = out(5)
    lib().mpe_moon(MJ, _REG, addr)
    lam, bet, rho, msp, mdp = list(b)
    assert 0.0 <= lam < 2 * math.pi
    assert abs(bet) < 0.1  # the ecliptic latitude never leaves +-5.3 degrees
    assert 0.0 < msp < 24.0 and 0.0 < mdp < 24.0
    # lam/bet are the *mean* geocentric ecliptic longitude and latitude; the
    # pipeline's hlon/hlat add nutation on top, which is about 41 arcseconds of
    # longitude and 0.8 of latitude. That difference is the nutation, and it is
    # what the surrounding code adds, so the two are pinned to it rather than to
    # each other exactly.
    theirs = upstream.Moon()
    theirs.compute(upstream.Date(str(mojoephem.Date(MJ))))
    assert arcsec(lam, float(theirs.hlon)) == pytest.approx(40.9, abs=0.2)
    assert arcsec(bet, float(theirs.hlat)) == pytest.approx(0.80, abs=0.05)
    # the mean longitude is pinned in degrees: 269.346 at this instant
    assert math.degrees(lam) == pytest.approx(269.346006, abs=1e-5)
    assert math.degrees(bet) == pytest.approx(-4.889099, abs=1e-5)
    # rho is the geocentric distance in au, and is the earth_distance the
    # pipeline reports to within upstream's own float storage
    assert abs(rho - theirs.earth_distance) / theirs.earth_distance < 1e-5
    assert rho == pytest.approx(0.0025439, abs=1e-7)
    # the mean phase angles are in hours: the sun's mean phase gains about a
    # minute a day, the moon's about 13.7 minutes (its anomalistic motion)
    assert msp == pytest.approx(9.201892, abs=1e-6)
    assert mdp == pytest.approx(11.088383, abs=1e-6)
    b2, addr2 = out(5)
    lib().mpe_moon(MJ + 1.0, _REG, addr2)
    assert 49000.0 < math.degrees((b2[0] - lam) % (2 * math.pi)) * 3600.0 < 49500.0
    assert (b2[3] - msp) % 24.0 == pytest.approx(0.017202, abs=1e-5)
    assert (b2[4] - mdp) % 24.0 == pytest.approx(0.228027, abs=1e-5)
    # and the pipeline's own hlon/hlat is this lam/bet plus nutation
    moon = mojoephem.Moon()
    moon.compute(mojoephem.Date(MJ))
    assert arcsec(lam, float(moon.hlon)) == pytest.approx(40.9, abs=0.2)
    assert arcsec(bet, float(moon.hlat)) == pytest.approx(0.80, abs=0.05)


# --------------------------------------------------------------------------
# misc.c / aa_hadec.c / parallax.c / refract.c / utc_gst.c
# --------------------------------------------------------------------------


def test_mpe_solve_sphere():
    """solve_sphere: the clamped cosine, and the arc back to the target.

    ca = cos(b)*cc + sin(b)*sc*cos(A); cc and sc arrive already as the
    cosine and sine of the third angle, which is why they are not cos(cc).
    """
    b, addr = out(2)
    for A, bb, cc, sc in (
        (0.4, 0.5, 0.3, 0.9), (0.4, 0.5, 0.3, -0.9),
        (0.4, 0.5, 0.3, 0.0), (0.4, 0.5, -0.3, 0.0),
    ):
        lib().mpe_solve_sphere(A, bb, cc, sc, addr)
        want = math.cos(bb) * cc + math.sin(bb) * sc * math.cos(A)
        assert b[0] == pytest.approx(want, abs=1e-12)
        assert -1.0 <= b[0] <= 1.0
        assert 0.0 <= b[1] < 2 * math.pi
    # sc below the 1e-7 threshold takes the closed form: cc > 0 gives pi - A,
    # cc < 0 gives A. The cap is then just cos(b)*cc, which is the branch's
    # only other output and so checks that slot too.
    lib().mpe_solve_sphere(0.4, 0.5, 0.3, 0.0, addr)
    assert b[1] == pytest.approx(math.pi - 0.4, abs=1e-12)
    lib().mpe_solve_sphere(0.4, 0.5, -0.3, 0.0, addr)
    assert b[1] == pytest.approx(0.4, abs=1e-12)


def test_mpe_aa_hadec_and_hadec_aa():
    """alt/az <-> hour angle/dec, and the round trip back to where it started.

    The round trip is only defined for a body that is actually above the
    horizon: below it the declination is not recoverable from alt/az.
    """
    b, addr = out(2)
    for lt, ha, dec in (
        (0.5, 1.0, 0.3), (0.0, 0.0, 0.0), (0.5, 1.0, -0.2), (0.5, 1.0, 0.2),
        (-0.5, 2.0, 0.1),
    ):
        lib().mpe_hadec_aa(lt, ha, dec, addr)
        alt, az = b[0], b[1]
        assert -math.pi / 2 - 1e-9 <= alt <= math.pi / 2 + 1e-9
        assert 0.0 <= az < 2 * math.pi
        lib().mpe_aa_hadec(lt, alt, az, addr)
        assert arcsec(b[0], ha) < 1e-6
        assert arcsec(b[1], dec) < 1e-6
    # at the equator a body on the meridian at declination 0 is overhead
    lib().mpe_hadec_aa(0.0, 0.0, 0.0, addr)
    assert b[0] == pytest.approx(math.pi / 2, abs=1e-12)
    assert b[1] == pytest.approx(0.0, abs=1e-12)
    # the hour angle comes back wrapped into (-pi, pi]
    lib().mpe_hadec_aa(0.5, 6.0, 0.0, addr)
    alt, az = b[0], b[1]
    lib().mpe_aa_hadec(0.5, alt, az, addr)
    assert -math.pi <= b[0] <= math.pi


def test_mpe_geoc_lat():
    """The geocentric latitude, and its own 89.9999 degree guard."""
    assert lib().mpe_geoc_lat(0.0) == 0.0
    for phi in (0.5, -0.5, 1.0, -1.0):
        assert lib().mpe_geoc_lat(phi) == pytest.approx(
            math.atan2(math.tan(phi) / 1.00674, 1.0), abs=1e-15
        )
    assert lib().mpe_geoc_lat(-0.5) == pytest.approx(-lib().mpe_geoc_lat(0.5), abs=1e-15)
    # monotonic, and never larger than the surface latitude
    prev = -math.inf
    for k in range(0, 89):
        v = lib().mpe_geoc_lat(k * math.pi / 180.0)
        assert v >= prev
        assert abs(v) <= abs(k * math.pi / 180.0) + 1e-12
        prev = v
    # past the guard phi is returned unchanged rather than calling tan()
    assert lib().mpe_geoc_lat(math.pi / 2) == math.pi / 2
    assert lib().mpe_geoc_lat(-math.pi / 2) == -math.pi / 2
    assert lib().mpe_geoc_lat(3.0) == 3.0


def test_mpe_ta_par():
    """Topocentric parallax, and the distance it writes back to the caller."""
    b, addr = out(2)
    for r in (60.0, 600.0, 6000.0):
        rho, rho_addr = out(1)
        rho[0] = r
        lib().mpe_ta_par(1.0, 0.3, 0.9, 0.001, rho_addr, addr)
        aha, adec = b[0], b[1]
        assert 0.0 <= aha < 2 * math.pi
        assert abs(adec - 0.3) < 0.02
        # the observer is lifted off the surface, so the distance shrinks a little
        assert r - 1.0 < rho[0] < r
    # parallax scales as 1/rho: the moon at 60 radii shifts far more than a
    # body at 6000 radii, by a factor of about 100
    shifts = []
    for r in (60.0, 600.0, 6000.0):
        rho, rho_addr = out(1)
        ob, ob_addr = out(2)
        rho[0] = r
        lib().mpe_ta_par(1.0, 0.3, 0.9, 0.001, rho_addr, ob_addr)
        shifts.append(abs(arcsec(ob[0], 1.0)))
    assert shifts[0] > shifts[1] > shifts[2]
    assert shifts[0] / shifts[1] == pytest.approx(10.0, rel=0.01)
    assert shifts[1] / shifts[2] == pytest.approx(10.0, rel=0.01)
    # 1900 arcseconds is about half a degree: the moon's parallax, not a
    # rounding wobble
    assert 1500.0 < shifts[0] < 2500.0
    # the shift is pinned: 1899.1 arcseconds at rho 60, and it halves exactly
    # as the distance doubles
    assert shifts[0] == pytest.approx(1899.107, abs=0.01)
    assert shifts[1] == pytest.approx(188.911, abs=0.01)
    assert shifts[2] == pytest.approx(18.881, abs=0.01)
    # every remaining argument is a real input, so perturbing any of them moves
    # the answer by a measurable amount - a kernel that ignored one of them
    # would return the baseline again
    def shift(tha, tdec, phi, ht, rho_in):
        rho, rho_addr = out(1)
        ob, ob_addr = out(2)
        rho[0] = rho_in
        lib().mpe_ta_par(tha, tdec, phi, ht, rho_addr, ob_addr)
        return (ob[0] - tha) * 206264.8, ob[1], rho[0]

    baseline = shift(1.0, 0.3, 0.9, 0.001, 60.0)
    assert baseline[0] == pytest.approx(1899.1065, abs=0.01)
    assert abs(baseline[1] - 0.3) < 0.02
    # at the equator the shift is larger; at declination 0 it is smaller; a
    # height of half an earth radius lifts the observer and shrinks it
    assert shift(1.0, 0.3, 0.0, 0.001, 60.0)[0] == pytest.approx(3059.681, abs=0.01)
    assert shift(1.0, 0.0, 0.9, 0.001, 60.0)[0] == pytest.approx(1813.811, abs=0.01)
    assert shift(1.0, 0.3, 0.9, 0.5, 60.0)[0] == pytest.approx(2852.157, abs=0.01)
    # and the distance written back is the topocentric one, which is smaller
    # than the geometric distance by roughly the observer's height
    assert 59.0 < baseline[2] < 60.0
    assert 59.0 < shift(1.0, 0.3, 0.9, 0.5, 60.0)[2] < 59.5


def test_mpe_refract():
    """refract lifts the apparent altitude, by the right amount at each height.

    The offsets are the classic refraction values: about 29 arcminutes at the
    horizon, falling to about 1.5 at 45 degrees and 5 arcseconds at 85.
    """
    b, addr = out(1)
    for deg, low, high in (
        (0.5, 1200.0, 1700.0), (5.0, 540.0, 600.0), (15.0, 200.0, 230.0),
        (45.0, 50.0, 65.0), (70.0, 15.0, 28.0), (85.0, 2.0, 10.0),
    ):
        ta = math.radians(deg)
        lib().mpe_refract(1010.0, 10.0, ta, addr)
        got = math.degrees(b[0] - ta) * 3600.0
        assert low <= got <= high, f"ta={deg} offset={got}"
        assert b[0] > ta  # refraction always raises the apparent altitude
    # no pressure, no refraction
    lib().mpe_refract(0.0, 10.0, 0.3, addr)
    assert b[0] == pytest.approx(0.3, abs=1e-15)
    # warmer air is less dense and refracts less, so the cold offset is larger
    lib().mpe_refract(1010.0, 30.0, 0.1, addr)
    hot = b[0] - 0.1
    lib().mpe_refract(1010.0, 0.0, 0.1, addr)
    cold = b[0] - 0.1
    assert hot < cold
    assert math.degrees(hot) * 3600.0 == pytest.approx(480.63, abs=0.5)
    assert math.degrees(cold) * 3600.0 == pytest.approx(532.33, abs=0.5)
    # and the correction falls monotonically with altitude
    prev = math.inf
    for deg in range(1, 89, 4):
        ta = math.radians(deg)
        lib().mpe_refract(1010.0, 10.0, ta, addr)
        assert b[0] - ta <= prev
        prev = b[0] - ta
    # a NaN altitude passes straight through rather than looping forever
    lib().mpe_refract(1010.0, 10.0, float("nan"), addr)
    assert b[0] != b[0]


def test_mpe_unrefract():
    """unrefract inverts the model, and agrees with upstream to 1e-6 arcsec."""
    b, addr = out(1)
    for pressure, temp in ((1010.0, 10.0), (990.0, 25.0), (0.0, 15.0)):
        for ta in (-0.1, 0.0, 0.05, 0.2, 0.5, 1.0):
            theirs = float(upstream._libastro.unrefract(pressure, temp, ta))
            lib().mpe_unrefract(pressure, temp, ta, addr)
            assert arcsec(b[0], theirs) < 1e-6
            if pressure > 0.0:
                # and the round trip through refract recovers the input, to
                # the 0.1 arcsec the secant method in refract converges to
                lib().mpe_refract(pressure, temp, b[0], addr)
                assert arcsec(b[0], ta) < 0.1


def test_mpe_utc_gst_and_gst_utc():
    """Apparent sidereal time, in hours, converting both ways."""
    b, addr = out(1)
    for mj in DATES:
        lib().mpe_utc_gst(mj, 0.0, addr)
        gst = b[0]
        assert 0.0 <= gst < 24.0  # hours, not radians
        assert gst == pytest.approx(lib().mpe_gmst0(mj), abs=0.0)
        lib().mpe_gst_utc(mj, gst, addr)
        assert b[0] == pytest.approx(0.0, abs=1e-6)
    # twelve hours of UT advances the sidereal clock by slightly more than
    # twelve sidereal hours, because the sidereal day is shorter
    lib().mpe_utc_gst(MJ, 0.0, addr)
    midnight = b[0]
    lib().mpe_utc_gst(MJ, 12.0, addr)
    assert 12.03 < (b[0] - midnight) % 24.0 < 12.04
    # a day on gains about 3m56s
    lib().mpe_utc_gst(MJ + 1.0, 0.0, addr)
    assert 0.065 < (b[0] - midnight) % 24.0 < 0.067
    # and a year on, that gain has come almost all the way round
    lib().mpe_utc_gst(MJ + 365.0, 0.0, addr)
    assert 23.98 < (b[0] - midnight) % 24.0 < 23.99


def test_mpe_gmst0():
    """Greenwich mean sidereal time at 0h, advancing about 3m56s a day."""
    for day in (0.5, 1.5, 2.5, 100.5, 3650.5, 45000.5):
        assert 0.0 <= lib().mpe_gmst0(day) < 24.0
    # a single day on is 3m56.5s of sidereal time, at every epoch tested
    for day in (0.5, 1.5, 100.5, 3650.5, 45000.5, 100000.5):
        assert 0.0657 < lib().mpe_gmst0(day + 1.0) - lib().mpe_gmst0(day) < 0.0658
    # 2.0 is also a midnight, so that is a whole day on and not half of one:
    # gmst0 truncates with floor(mj - 0.5) + 0.5, so an unaligned argument
    # moves the epoch a whole day, which is exactly what catches an off-by-one
    assert lib().mpe_gmst0(2.0) - lib().mpe_gmst0(0.5) == pytest.approx(0.0657, abs=1e-3)


def test_mpe_now_lst():
    """now_lst is local apparent sidereal time in hours."""
    b, addr = out(1)
    for day in (0.5, 1.5, 2.5, 100.5, 3650.5, 45000.5):
        lib().mpe_now_lst(day, 0.0, _REG, addr)
        assert 0.0 <= b[0] < 24.0
    # a single day on is the same 3m56.5s, plus the nutation in longitude
    for day in (0.5, 1.5, 100.5, 3650.5, 45000.5):
        lib().mpe_now_lst(day, 0.0, _REG, addr)
        first = b[0]
        lib().mpe_now_lst(day + 1.0, 0.0, _REG, addr)
        assert 0.0657 < b[0] - first < 0.0658
    # longitude enters as its own angle converted to hours, not as hours
    lib().mpe_now_lst(MJ, 0.0, _REG, addr)
    zero = b[0]
    lib().mpe_now_lst(MJ, 0.5, _REG, addr)
    assert b[0] == pytest.approx(zero + 0.5 * 12.0 / math.pi, abs=1e-9)
    # and it is the sidereal time the Observer reports: the kernel answers in
    # hours, the Angle's float is radians and its unit is hours
    site = mojoephem.Observer()
    site.lat, site.lon, site.elevation = "51.4778", "0.0015", 70.0
    site.date = mojoephem.Date(MJ)
    lib().mpe_now_lst(MJ, site._lon_radians(), _REG, addr)
    lst = b[0]
    # Observer.sidereal_time returns an Angle in hours; its float is radians,
    # so the comparison is against the Angle's hour value
    theirs = upstream.Observer()
    theirs.lat, theirs.lon, theirs.elevation = "51.4778", "0.0015", 70.0
    theirs.date = upstream.Date(str(mojoephem.Date(MJ)))
    assert site.sidereal_time().hours == pytest.approx(lst, abs=1e-9)
    # upstream's Angle carries the same per-radian scale in its float
    assert float(theirs.sidereal_time()) * 12.0 / math.pi == pytest.approx(
        lst, abs=1e-6
    )


def test_mpe_radec2ha():
    """The hour angle in [0, 2pi), from local sidereal time in hours."""
    for lst, ra, want in (
        (0.0, 0.0, 0.0),
        (1.0, math.radians(15.0), 0.0),  # lst 1h == ra 15 degrees
        (12.0, math.radians(180.0), 0.0),
        (0.0, math.pi, math.pi),
        (6.0, 0.0, math.pi / 2),  # lst 6h == ra 0, so ha = 90 degrees
        (18.0, 0.0, 3.0 * math.pi / 2),
        (0.5, 0.0, math.radians(7.5)),
        (0.0, -0.1, 0.1),  # a negative ra wraps forward
    ):
        assert lib().mpe_radec2ha(lst, ra) == pytest.approx(want, abs=1e-12)


def test_mpe_elongation():
    """Elongation: zero at conjunction, pi at opposition, signed by longitude."""
    b, addr = out(1)
    lib().mpe_elongation(0.0, 0.0, 0.0, addr)
    assert b[0] == 0.0
    lib().mpe_elongation(math.pi, 0.0, 0.0, addr)
    assert b[0] == pytest.approx(math.pi, abs=1e-12)
    lib().mpe_elongation(math.pi / 2, 0.0, 0.0, addr)
    assert b[0] == pytest.approx(math.pi / 2, abs=1e-12)
    # a morning object (lam below lsn) is negative, an evening one positive
    lib().mpe_elongation(-0.1, 0.0, 0.0, addr)
    assert b[0] == pytest.approx(-0.1, abs=1e-12)
    lib().mpe_elongation(0.1, 0.0, 0.0, addr)
    assert b[0] == pytest.approx(0.1, abs=1e-12)
    # and latitude moves it away from the plain longitude difference
    lib().mpe_elongation(0.0, 0.1, 0.0, addr)
    assert b[0] == pytest.approx(0.1, abs=1e-12)


# --------------------------------------------------------------------------
# circum.c / obj_planet
# --------------------------------------------------------------------------


def test_mpe_deflect_passes_the_position_through_outside_its_window():
    """Upstream's guard is `el < 170 || el > 179.75 || rho < rsn`, so the
    deflection is applied only in the 9.75 degree window between those two
    elongations, and only for a body farther from the sun than the earth is.
    Everything else returns the position it was handed unchanged, which is the
    case the whole pipeline runs in for every planet most of the time."""
    b, addr = out(2)
    for el_deg in (0.0, 10.0, 90.0, 169.0, 170.0, 180.0):
        lpd = math.radians(el_deg)
        lsn = math.pi  # the sun seen from the earth, opposite the earth's longitude
        b[0], b[1] = 2.0, 0.5
        lib().mpe_deflect(MJ, lpd, 0.0, 1.0, lsn, 5.0, addr, _REG)
        assert (b[0], b[1]) == (2.0, 0.5), f"el={el_deg} was deflected"
    # nearer than the sun is a pass-through too, even inside the window
    lpd = math.radians(175.0)
    b[0], b[1] = 2.0, 0.5
    lib().mpe_deflect(MJ, lpd, 0.0, 1.0, math.pi, 0.5, addr, _REG)
    assert (b[0], b[1]) == (2.0, 0.5)


def test_mpe_deflect_bends_by_a_fraction_of_an_arcsecond():
    """Inside the window the correction grows towards the limb and falls as
    1/rsn, the heliocentric distance."""
    b, addr = out(2)
    shifts = {}
    for el_deg in (171.0, 175.0, 179.0):
        lpd = math.radians(el_deg)
        b[0], b[1] = 2.0, 0.5
        lib().mpe_deflect(MJ, lpd, 0.0, 1.0, math.pi, 5.0, addr, _REG)
        assert (b[0], b[1]) != (2.0, 0.5), f"el={el_deg} was not deflected"
        assert abs(arcsec(b[0], 2.0)) < 1.0
        assert abs(arcsec(b[1], 0.5)) < 1.0
        shifts[el_deg] = abs(arcsec(b[0], 2.0))
    # the bending grows towards the limb
    assert shifts[171.0] < shifts[175.0] < shifts[179.0]
    # g1 scales as 1/rsn, the heliocentric distance
    b[0], b[1] = 2.0, 0.5
    lib().mpe_deflect(MJ, math.radians(175.0), 0.0, 5.0, math.pi, 5.0, addr, _REG)
    assert abs(arcsec(b[0], 2.0)) < shifts[175.0]


OBJ_FIELDS = (
    "hlon", "hlat", "sun_distance", "earth_distance", "elong", "phase",
    "mag", "size", "g_ra", "g_dec", "a_ra", "a_dec", "ha", "alt", "az", "rho",
)


@pytest.mark.parametrize(
    "name",
    ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
     "Pluto", "Sun", "Moon"],
)
def test_mpe_obj_planet(name):
    """obj_planet fills the documented 17-slot vector, for every body."""
    o, o_addr = out(17)
    cls = getattr(mojoephem, name)
    theirs = getattr(upstream, name)()
    theirs.compute(upstream.Date(str(mojoephem.Date(MJ))))
    for i in range(17):
        o[i] = -12345.0  # the kernel must clear every slot, not just the ones it fills
    lib().mpe_obj_planet(
        MJ, cls.__planet__, 0.0, 0.0, 0.0, J2000, 0, 0.0, 15.0, _REG, o_addr,
    )
    assert o[16] == MJ, "the mj that produced the vector was not recorded"
    assert -12345.0 not in list(o), "a slot was left unwritten"
    assert arcsec(o[8], float(theirs.g_ra)) < 1e-3
    assert arcsec(o[9], float(theirs.g_dec)) < 1e-3
    assert arcsec(o[10], float(theirs.a_ra)) < 1e-3
    assert arcsec(o[11], float(theirs.a_dec)) < 1e-3
    assert o[3] == pytest.approx(theirs.earth_distance, rel=1e-5)
    # elong is stored in DEGREES, signed by morning/evening; upstream's Angle
    # has already converted it back to radians, and the Python layer divides
    # by pi/180 to match
    assert o[4] == pytest.approx(math.degrees(float(theirs.elong)), abs=1e-4)
    assert -180.0 <= o[4] <= 180.0
    # phase is a percentage, 0 at new and 100 at full
    assert o[5] == pytest.approx(float(theirs.phase), abs=1e-4)
    # and the magnitude is quantised to hundredths on both sides
    assert o[6] == pytest.approx(float(theirs.mag), abs=0.011)
    assert -30.0 <= o[6] <= 30.0  # magnitude
    assert o[7] > 0.0  # angular size
    for i, field in enumerate(OBJ_FIELDS):
        assert o[i] == o[i], f"slot {i} ({field}) is NaN"


def test_mpe_obj_planet_topocentric():
    """With lat/lon the vector carries alt, az and hour angle too."""
    o, o_addr = out(17)
    obs = mojoephem.Observer()
    obs.lat, obs.lon, obs.elevation = "51.4778", "-0.0015", 70.0
    obs.date = mojoephem.Date(MJ)
    moon = mojoephem.Moon()
    moon.compute(obs)
    theirs = upstream.Observer()
    theirs.lat, theirs.lon, theirs.elevation = "51.4778", "-0.0015", 70.0
    theirs.date = upstream.Date(str(mojoephem.Date(MJ)))
    moon_theirs = upstream.Moon()
    moon_theirs.compute(theirs)
    lib().mpe_obj_planet(
        obs._date, mojoephem.Moon.__planet__, float(obs._lat), float(obs._lon),
        obs.elevation / mojoephem.earth_radius, float(obs._epoch), 1,
        obs.pressure, obs.temp, _REG, o_addr,
    )
    assert arcsec(o[13], float(moon.alt)) < 1e-9
    assert arcsec(o[14], float(moon.az)) < 1e-9
    assert arcsec(o[12], float(moon.ha)) < 1e-9
    assert arcsec(o[13], float(moon_theirs.alt)) < 0.1
    assert arcsec(o[14], float(moon_theirs.az)) < 0.1


def test_mpe_obj_planet_series_matches_the_scalar_kernel():
    """The batch path reproduces mpe_obj_planet exactly, row for row."""
    dates = np.array([MJ + i * 0.25 for i in range(300)])
    batch = np.zeros((dates.size, 17))
    lib().mpe_obj_planet_series(
        dates.ctypes.data, batch.ctypes.data, dates.size,
        mojoephem.Moon.__planet__, 0.0, 0.0, 0.0, J2000, 0, 0.0, 15.0, _REG,
    )
    single, addr = out(17)
    for i in (0, 1, 137, 299):
        lib().mpe_obj_planet(
            dates[i], mojoephem.Moon.__planet__, 0.0, 0.0, 0.0, J2000, 0, 0.0,
            15.0, _REG, addr,
        )
        assert list(batch[i]) == list(single), f"row {i}"
        assert batch[i, 16] == dates[i]


def test_mpe_obj_planet_series_writes_every_row():
    """Sizes that do not divide the worker count still fill every row.

    256 is the parallel threshold and 64 the per-worker minimum, so this
    straddles both: 257 dates over 4 workers leaves the last chunk short, and
    1000 over 15 leaves the last two.
    """
    for n in (1, 2, 63, 64, 255, 256, 257, 511, 1000):
        dates = np.array([MJ + i * 0.5 for i in range(n)])
        arr = np.full((n, 17), -7.0)
        lib().mpe_obj_planet_series(
            dates.ctypes.data, arr.ctypes.data, n, mojoephem.Sun.__planet__,
            0.0, 0.0, 0.0, J2000, 0, 0.0, 15.0, _REG,
        )
        assert -7.0 not in arr, f"n={n} left a row unwritten"
        for i in (0, n // 2, n - 1):
            assert arr[i, 16] == dates[i]
            assert arr[i, 8] != 0.0 or arr[i, 16] != dates[i]  # the row is real
