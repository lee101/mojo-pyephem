"""Parity of the individual libastro kernels against upstream's exports.

Where upstream exposes a routine to Python (precess, eq_ecl, ecl_eq,
unrefract, julian_date, delta_t) this compares the Mojo kernel directly.
For the rest it drives the whole pipeline, which is the only place
upstream makes them visible.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import mojoephem
from conftest import arcsec, upstream

ANGLES = [
    (0.0, 0.0),
    (0.3, 0.1),
    (1.5, -0.4),
    (3.0, 0.0),
    (4.5, 0.6),
    (5.9, -0.9),
]


@pytest.mark.parametrize("ra,dec", ANGLES)
@pytest.mark.parametrize("epoch", [mojoephem.J2000, 0.0, 30000.0, 45290.5])
def test_precess(ra, dec, epoch):
    """Rigorous precession from J2000 to the given epoch matches exactly."""
    theirs = upstream._libastro.precess(upstream.J2000, epoch, ra, dec)
    ours = mojoephem.precess(mojoephem.J2000, epoch, ra, dec)
    assert ours[0] == pytest.approx(float(theirs[0]), abs=1e-12)
    assert ours[1] == pytest.approx(float(theirs[1]), abs=1e-12)


@pytest.mark.parametrize("ra,dec", ANGLES)
def test_eq_ecl(ra, dec):
    """Equatorial to ecliptic matches upstream to 1e-10 arcsec."""
    theirs = upstream._libastro.eq_ecl(upstream.J2000, ra, dec)
    ours = mojoephem.eq_ecl(mojoephem.J2000, ra, dec)
    assert arcsec(ours[0], float(theirs[0])) < 1e-10
    assert arcsec(ours[1], float(theirs[1])) < 1e-10


@pytest.mark.parametrize("lon,lat", ANGLES)
def test_ecl_eq(lon, lat):
    """Ecliptic to equatorial matches upstream to 1e-10 arcsec."""
    theirs = upstream._libastro.ecl_eq(upstream.J2000, lon, lat)
    ours = mojoephem.ecl_eq(mojoephem.J2000, lon, lat)
    assert arcsec(ours[0], float(theirs[0])) < 1e-10
    assert arcsec(ours[1], float(theirs[1])) < 1e-10


@pytest.mark.parametrize("alt", [-0.3, 0.0, 0.05, 0.2, 0.26, 0.5])
@pytest.mark.parametrize("pressure,temp", [(0.0, 15.0), (1010.0, 10.0), (990.0, 25.0)])
def test_unrefract(alt, pressure, temp):
    """Inverting the refraction model matches upstream exactly."""
    theirs = float(upstream._libastro.unrefract(pressure, temp, alt))
    ours = float(mojoephem.unrefract(pressure, temp, alt))
    assert arcsec(ours, theirs) < 1e-6


@pytest.mark.parametrize(
    "when",
    [
        "1600/1/1",
        "1700/6/1 12:00",
        "1800/1/1",
        "1900/1/1 00:00:30",
        "1955/10/1",
        "2000/1/1 12:00",
        "2024/7/4 23:59:59.5",
        "2200/1/1",
        "-0500/1/1",
        "1752/9/14",
    ],
)
def test_julian_date(when):
    """Date parsing and julian_date agree to the double's last bit."""
    ours = mojoephem.julian_date(mojoephem.Date(when))
    theirs = upstream.julian_date(upstream.Date(when))
    assert ours == pytest.approx(theirs, abs=1e-9)


@pytest.mark.parametrize(
    "when",
    ["1600/1/1", "1800/1/1", "1900/1/1", "1950/1/1", "2000/1/1", "2018/6/1", "2024/1/1", "2200/1/1"],
)
def test_delta_t(when):
    """deltaT agrees to a millisecond, from tabulated values to the
    long-term polynomial beyond 2118."""
    ours = mojoephem.delta_t(mojoephem.Date(when))
    theirs = upstream.delta_t(upstream.Date(when))
    assert ours == pytest.approx(theirs, abs=1e-3)


def test_date_roundtrip():
    for when in ("1600/1/1", "1800/3/1 06:07:08", "2024/7/4 23:59:59", "2200/1/1"):
        assert str(mojoephem.Date(when)) == str(upstream.Date(when))
        assert mojoephem.Date(when).triple() == upstream.Date(when).triple()


def test_separation():
    """Angular separation between bodies and between raw angles."""
    d = mojoephem.Date("2024/1/1")
    for name_a, name_b in (("Sun", "Moon"), ("Venus", "Jupiter"), ("Mars", "Saturn")):
        ours = getattr(mojoephem, name_a)()
        ours.compute(d)
        theirs = getattr(upstream, name_a)()
        theirs.compute(upstream.Date("2024/1/1"))
        b_ours = getattr(mojoephem, name_b)()
        b_ours.compute(d)
        b_theirs = getattr(upstream, name_b)()
        b_theirs.compute(upstream.Date("2024/1/1"))
        a = float(mojoephem.separation(ours, b_ours))
        b = float(upstream.separation(theirs, b_theirs))
        assert arcsec(a, b) < 1e-3



def test_separation_argument_forms():
    """The pair fast path and the general Observer/Body path agree."""
    a = (1.1, 0.4)
    b = (1.3, 0.45)
    fast = float(mojoephem.separation(a, b))
    # lists and arrays are not tuples, so they take the general path
    assert fast == float(mojoephem.separation(list(a), list(b)))
    assert fast == float(mojoephem.separation(np.array(a), np.array(b)))
    assert fast == pytest.approx(float(upstream.separation(a, b)), abs=1e-12)
    # a position against itself is exactly zero
    assert float(mojoephem.separation(a, a)) == 0.0

    # Observer coordinates are radians, as upstream's are
    site = mojoephem.Observer()
    site.lat, site.lon, site.elevation = 0.9, -0.0023, 70.0
    origin = (0.0, 0.0)
    plng, plat = mojoephem._separation_arg(site)
    cosine = math.cos(plat) * math.cos(plng)
    assert float(mojoephem.separation(site, origin)) == pytest.approx(
        math.acos(cosine), abs=1e-12
    )
    with pytest.raises(TypeError):
        mojoephem.separation(1.0, b)


@pytest.mark.parametrize("n", [1, 2, 3, 255, 256, 257, 400, 1000, 4001])
def test_series_sizes_cover_every_date(n):
    """Every date is computed, below and above the parallel threshold.

    The worker count does not divide most of these, so the last chunk runs
    short; a kernel that skipped its tail would show up here.
    """
    start = float(mojoephem.Date("2024/1/1"))
    dates = start + np.arange(n) * 0.25
    for name in ("Sun", "Moon", "Saturn"):
        out = mojoephem.compute_series(
            getattr(mojoephem, name).__planet__, dates
        )
        body = getattr(mojoephem, name)()
        theirs = getattr(upstream, name)()
        for i in (0, n // 2, n - 1):
            body.compute(mojoephem.Date(dates[i]))
            theirs.compute(upstream.Date("2024/1/1") + i * 0.25)
            assert arcsec(out[i, 8], float(theirs.g_ra)) < 1e-3
            assert arcsec(out[i, 9], float(theirs.g_dec)) < 1e-3
            assert abs(out[i, 16] - dates[i]) == 0.0


def test_series_parallel_matches_serial():
    """The worker-split series is the same numbers as the serial one."""
    start = float(mojoephem.Date("2024/1/1"))
    dates = start + np.arange(400) * 0.25
    obs = mojoephem.Observer()
    obs.lat, obs.lon, obs.elevation = 51.5, -0.13, 70.0
    for name in ("Sun", "Moon", "Venus", "Saturn"):
        planet = getattr(mojoephem, name).__planet__
        parallel = mojoephem.compute_series(planet, dates, obs)
        serial = np.concatenate(
            [mojoephem.compute_series(planet, dates[i : i + 7], obs) for i in range(0, 400, 7)]
        )
        assert np.array_equal(parallel, serial)


def test_topocentric_series_matches_scalar_body():
    """The topocentric Sun and Moon batches equal one body at a time.

    `obj_planet` computes the solar ecliptic longitude and distance once and
    threads them into `plans` and `cir_pos` instead of letting each recompute
    `sunpos`. Sun and Moon reach `cir_pos` through `sun_cir`/`moon_cir` rather
    than `cir_sky`, so they are the two paths where the threaded arguments
    differ from the planet path; this pins that the batched and scalar forms
    agree element for element on both.
    """
    dates = np.array([float(mojoephem.Date("2024/1/1")) + i * 0.25 for i in range(37)])
    obs = mojoephem.Observer()
    obs.lat, obs.lon, obs.elevation = 51.5, -0.13, 70.0
    for name in ("Sun", "Moon"):
        body = getattr(mojoephem, name)()
        out = mojoephem.compute_series(body.__planet__, dates, obs)
        for i in (0, 1, 18, 36):
            obs.date = mojoephem.Date(dates[i])
            body.compute(obs)
            # a_ra, a_dec (10, 11), alt, az (13, 14) and the distances (2, 3)
            assert arcsec(out[i, 10], float(body.a_ra)) < 1e-9
            assert arcsec(out[i, 11], float(body.a_dec)) < 1e-9
            assert arcsec(out[i, 13], float(body.alt)) < 1e-9
            assert arcsec(out[i, 14], float(body.az)) < 1e-9
            assert abs(out[i, 2] - body.sun_distance) < 1e-12
            assert abs(out[i, 3] - body.earth_distance) < 1e-12


def test_sidereal_time():
    """The local apparent sidereal time matches upstream's Observer method."""
    site = ("51.4778", "0.0015", 70.0)
    ours = mojoephem.Observer()
    ours.lat, ours.lon, ours.elevation = site
    ours.date = mojoephem.Date("2024/6/21 22:00")
    theirs = upstream.Observer()
    theirs.lat, theirs.lon, theirs.elevation = site
    theirs.date = upstream.Date("2024/6/21 22:00")
    assert arcsec(
        float(ours.sidereal_time()), float(theirs.sidereal_time())
    ) < 1e-6


@pytest.mark.parametrize(
    "fn",
    [
        "next_vernal_equinox",
        "next_summer_solstice",
        "next_autumnal_equinox",
        "next_winter_solstice",
        "previous_vernal_equinox",
        "previous_summer_solstice",
        "previous_autumnal_equinox",
        "previous_winter_solstice",
        "next_equinox",
        "previous_equinox",
        "next_solstice",
        "previous_solstice",
        "next_new_moon",
        "next_full_moon",
        "next_first_quarter_moon",
        "next_last_quarter_moon",
        "previous_new_moon",
        "previous_full_moon",
        "previous_first_quarter_moon",
        "previous_last_quarter_moon",
    ],
)
@pytest.mark.parametrize("when", ["2024/1/1", "2000/3/1", "1950/9/9", "2024/12/25"])
def test_solar_and_lunar_events(fn, when):
    """Equinoxes, solstices and moon phases agree to a tenth of a second."""
    ours = float(getattr(mojoephem, fn)(mojoephem.Date(when)))
    theirs = float(getattr(upstream, fn)(upstream.Date(when)))
    assert abs(ours - theirs) * 86400.0 < 0.1


def test_batch_series_matches_scalar():
    """compute_series agrees with the one-at-a-time path and with upstream."""
    dates = mojoephem.Date("2024/1/1") + np.arange(400) * 0.25
    for name in ("Sun", "Moon", "Venus", "Saturn"):
        out = mojoephem.compute_series(
            getattr(mojoephem, name).__planet__, dates
        )
        theirs = getattr(upstream, name)()
        for i in (0, 1, 137, 399):
            theirs.compute(upstream.Date("2024/1/1") + i * 0.25)
            assert arcsec(out[i, 8], float(theirs.g_ra)) < 1e-3
            assert arcsec(out[i, 9], float(theirs.g_dec)) < 1e-3
            assert abs(out[i, 3] - theirs.earth_distance) < 1e-6


def test_compute_series_rejects_an_out_of_range_planet():
    """A PLCode outside 0..9 raises instead of indexing the kernel's tables."""
    dates = np.array([float(mojoephem.Date("2024/1/1"))])
    for planet in (-1, 10, 99):
        with pytest.raises(ValueError):
            mojoephem.compute_series(planet, dates)


def test_vsop87_rejects_an_out_of_range_object():
    """vsop87 returns zeros for a PLCode it has no table for.

    A negative obj would otherwise index `a0[obj]` off the front of the
    tuple and read whatever is below it in the shared object's data.
    """
    lib = mojoephem._lib.lib()
    out = (ctypes.c_double * 3)()
    addr = ctypes.addressof(out)
    reg = mojoephem._tables.reg()
    for obj in (-5, -1, 7, 9, 99):
        lib.mpe_vsop87(50000.0, obj, 0.0, reg, addr)
        assert list(out) == [0.0, 0.0, 0.0]
    # and a real one is not zero, so the loop above is not vacuous
    lib.mpe_vsop87(50000.0, 0, 0.0, reg, addr)
    assert out[2] > 0.0


def test_chap95_uses_each_planet_own_record_count():
    """chap95 reads only the records its own planet has.

    The five outer planets have 216, 192, 129, 117 and 98 records. Reading
    Jupiter's count for all of them walks off the end of the shorter tables
    and into the next planet's coefficients.
    """
    lib = mojoephem._lib.lib()
    out = (ctypes.c_double * 3)()
    addr = ctypes.addressof(out)
    reg = mojoephem._tables.reg()
    mj = float(mojoephem.Date("2024/1/1"))
    # Saturn has 192 records, so a Jupiter-sized loop picks up 24 records of
    # Uranus's coefficients and moves the position.
    lib.mpe_chap95(mj, 3, 0.0, reg, addr)
    jupiter = list(out)
    lib.mpe_chap95(mj, 4, 0.0, reg, addr)
    saturn = list(out)
    # each planet must land near its own semimajor axis, in au
    assert 4.5 < math.sqrt(sum(x * x for x in jupiter)) < 6.0
    assert 8.0 < math.sqrt(sum(x * x for x in saturn)) < 11.0


def _series_via(rows: int, budget) -> np.ndarray:
    """compute_series with the thread budget forced to `budget`."""
    start = float(mojoephem.Date("2023/1/1"))
    dates = np.linspace(start, start + rows, rows)
    saved = mojoephem._lib._BUDGET
    mojoephem._lib._BUDGET = budget
    try:
        return mojoephem.compute_series(mojoephem.Moon.__planet__, dates)
    finally:
        mojoephem._lib._BUDGET = saved


def _budget_without_pool(spare: int) -> "mojoephem._lib.ThreadBudget":
    """A budget for a host on which MAX's worker pool cannot spawn.

    `pool_fits` is the host's own verdict on two measured numbers,
    `spare >= needed`, so the way to pin the fan-out is to report a pool
    requirement the host cannot meet. Overriding the verdict instead would
    build a budget that contradicts itself, and `as_kernel_arg()` would then
    answer from a premise its own fields deny.
    """
    from mojoephem._lib import ThreadBudget

    return ThreadBudget(spare=spare, needed=spare + 1)


def test_compute_series_fanout_matches_the_scalar_kernel():
    """The Python fan-out splits the batch; every row must still be the same.

    Forcing `pool_fits` False pins the path that chunks one batch across
    Python threads instead of MAX's worker pool. The chunk boundaries land
    mid-batch, so a row computed from the wrong slice would show up here.
    """
    serial = _series_via(1000, _budget_without_pool(0))
    for spare in (2, 3, 5, 8):
        fanned = _series_via(1000, _budget_without_pool(spare))
        assert np.array_equal(fanned, serial), f"spare={spare} changed a row"

    # and against the scalar kernel the batch is supposed to repeat
    start = float(mojoephem.Date("2023/1/1"))
    dates = np.linspace(start, start + 1000, 1000)
    for i in (0, 1, 2, 17, 511, 998, 999):
        body = mojoephem.Moon(dates[i])
        body.compute(dates[i])
        assert body._fields[mojoephem._F_GAERA] == serial[i, 8]
        assert body._fields[mojoephem._F_EDIST] == serial[i, 3]


def test_thread_budget_reports_a_usable_headroom():
    """The measurement must hand the kernel a worker count it can honour."""
    budget = mojoephem._lib.thread_budget()
    assert budget.spare >= 0
    assert budget.as_kernel_arg() >= 1
    # pool_fits means "MAX's whole pool will spawn", which is a property of the
    # host, so it must be stable across calls.
    assert budget.pool_fits == mojoephem._lib.thread_budget().pool_fits
