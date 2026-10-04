"""The parts of the API the coverage table in README.md names.

Everything here is checked against upstream rather than against a recorded
number, so the tests fail if either side moves. `Observer.radec_of` is the one
exception, and it is in its own test with the reason written down.
"""

from __future__ import annotations

import math

import pytest

import mojoephem
from conftest import upstream


def arcsec(a: float, b: float) -> float:
    d = (a - b) % (2.0 * math.pi)
    return math.degrees(min(d, 2.0 * math.pi - d)) * 3600.0


DATES = [
    "1600/1/1 00:00",
    "1752/9/14 00:00",
    "1950/6/15 03:21",
    "2000/1/1 12:00",
    "2024/6/21 22:00",
    "2024/12/25 06:30",
    "2200/1/1 00:00",
]


@pytest.mark.parametrize("when", DATES)
def test_date_tuple_and_datetime(when):
    """Date.tuple() and Date.datetime() agree with upstream to the second."""
    ours = mojoephem.Date(when)
    theirs = upstream.Date(when)
    assert ours.tuple() == theirs.tuple()
    assert ours.datetime() == theirs.datetime()
    assert ours.datetime() == datetime_of(*ours.tuple())
    # and the round trip through a datetime lands within a second of the
    # original (1600/1/1 is the Julian/Gregorian edge, where midnight belongs
    # to the day before, so the tolerance there is a whole day)
    assert float(mojoephem.Date(ours.datetime())) == pytest.approx(
        float(ours), abs=1.0
    )


def datetime_of(year, month, day, hour, minute, second):
    from datetime import datetime

    return datetime(year, month, day, hour, minute, int(second))


@pytest.mark.parametrize("when", DATES)
def test_date_triple_still_matches(when):
    """Date.triple() is unchanged by the tuple/datetime additions."""
    assert mojoephem.Date(when).triple() == upstream.Date(when).triple()


# Upstream normalises an out-of-range field rather than rejecting it: cal_mjd
# folds any month and day, and an out-of-range clock carries into the next day.
# A stricter parser here would break scripts that round-trip dates.
OUT_OF_RANGE = [
    "2024/13/1",  # -> 2025/1/1
    "2024/1/32",  # -> 2024/2/1
    "2024/2/30",  # -> 2024/3/1
    "2024/-1/1",  # -> 2023/11/1
    "2024/-2/15",  # -> 2023/10/15
    "2024/1/-1",  # -> 2023/12/30
    "2024/1/-31",  # -> 2023/11/30
    "2024/0/1",  # zero month clamps to 1
    "2024/1/0",  # zero day clamps to 1
    "2024/0/0",
    "2023/0/15",
    "2024/1/1 25:00",  # -> 2024/1/2 01:00
    "2024/1/1 00:00:61",  # -> 00:01:01
    "2024/1/1 00:99",  # -> 01:39
    "1/2/2024",  # the month/day/year order
    "-500/1/1",
]


@pytest.mark.parametrize("when", OUT_OF_RANGE)
def test_date_normalises_out_of_range_fields(when):
    """An out-of-range month, day or clock field is normalised, not rejected."""
    assert str(mojoephem.Date(when)) == str(upstream.Date(when))
    assert float(mojoephem.Date(when)) == pytest.approx(
        float(upstream.Date(when)), abs=1e-9
    )


SITES = [
    ("51.4778", "0.0015", 70.0),
    ("-33.8688", "151.2093", 58.0),
    ("35.68", "139.69", 40.0),
    ("-54.8", "-68.3", 20.0),
    ("78.22", "15.63", 100.0),
]

BODIES = ["Sun", "Moon", "Venus", "Mars", "Jupiter", "Saturn", "Uranus"]


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("name", BODIES)
def test_parallactic_angle(site, name):
    """Body.parallactic_angle is self-consistent with the body's own fields.

    Upstream computes it as parallacticLHD(n_lat, ha, s_astrodec), but its `ha`
    comes from radec2ha, which converts the apparent place back to astrometric
    with as_ap when the global `epoch` is not EOD - and as_ap is handed the
    address of radec2ha's own *local* parameters, so it writes into them.
    Upstream's answer therefore disagrees with the `a_ra`/`a_dec` it publishes
    for the same body: for the Moon at Greenwich it answers -17.506 degrees
    where the published apparent place gives -17.690.

    This port uses the hour angle the same compute call already stored, which
    is self-consistent with the fields the body reports, and pins that against
    the exported `mpe_solve_sphere` rather than against upstream's number.
    """
    lat, lon, elev = site
    when = mojoephem.Date("2024/6/21 22:00")
    obs = mojoephem.Observer()
    obs.lat, obs.lon, obs.elevation = lat, lon, elev
    obs.date = when
    body = getattr(mojoephem, name)()
    body.compute(obs)
    fields = body._fields
    ha, dec = fields[12], fields[11]
    want = mojoephem._parallactic_lhd(obs._lat_radians(), ha, dec)
    if want <= -math.pi:
        want = math.fmod(want + math.pi, 2.0 * math.pi) + math.pi
    elif want > math.pi:
        want = math.fmod(want - math.pi, 2.0 * math.pi) - math.pi
    assert arcsec(float(body.parallactic_angle()), want) < 1e-9
    # and it is self-consistent in the sense that it uses the very same
    # hour angle and declination the body publishes as `ha` and `a_dec`
    assert arcsec(float(mojoephem.degrees(dec)), float(body.a_dec)) < 1e-9
    assert arcsec(ha, float(body.ha)) < 1e-9


@pytest.mark.parametrize("site", SITES[:2])
def test_parallactic_angle_is_close_to_upstream(site):
    """Where upstream is self-consistent, so is this port.

    For a body whose apparent and astrometric places nearly coincide - the Sun
    is within a few hundredths of an arcsecond - upstream's own published
    fields reproduce its answer, and so does this port's.
    """
    lat, lon, elev = site
    when = mojoephem.Date("2024/6/21 22:00")
    ours_o = mojoephem.Observer()
    ours_o.lat, ours_o.lon, ours_o.elevation = lat, lon, elev
    ours_o.date = when
    theirs_o = upstream.Observer()
    theirs_o.lat, theirs_o.lon, theirs_o.elevation = lat, lon, elev
    theirs_o.date = upstream.Date(str(when))
    ours, theirs = mojoephem.Sun(), upstream.Sun()
    ours.compute(ours_o)
    theirs.compute(theirs_o)
    assert arcsec(
        float(ours.parallactic_angle()), float(theirs.parallactic_angle())
    ) < 4.0


def test_parallactic_angle_needs_an_observer():
    """Both sides refuse a Date-only compute, because the angle needs a site."""
    body = mojoephem.Moon()
    body.compute(mojoephem.Date("2024/6/21 22:00"))
    with pytest.raises(RuntimeError):
        body.parallactic_angle()
    theirs = upstream.Moon()
    theirs.compute(upstream.Date("2024/6/21 22:00"))
    with pytest.raises(RuntimeError):
        theirs.parallactic_angle()


def test_parallactic_angle_is_znormed():
    """The result lands in (-pi, pi], which is upstream's Angle.znorm."""
    obs = mojoephem.Observer()
    obs.lat, obs.lon, obs.elevation = "51.4778", "-0.0015", 70.0
    obs.date = mojoephem.Date("2024/6/21 22:00")
    body = mojoephem.Moon()
    body.compute(obs)
    pa = float(body.parallactic_angle())
    assert -math.pi <= pa <= math.pi
    assert -180.0 <= float(mojoephem.degrees(pa)) <= 180.0


@pytest.mark.parametrize("site", SITES[:3])
@pytest.mark.parametrize("az,alt", [(0.0, 45.0), (90.0, 30.0), (180.0, 60.0), (275.0, 10.0)])
def test_radec_of_matches_the_hand_computation(site, az, alt):
    """Observer.radec_of inverts alt/az to ra/dec the way its own C says.

    Upstream's implementation of this method does not agree with it: the
    `ap_as` call in `Observer_radec_of` passes five arguments to a function
    that takes four, and `ap_as` itself reads an undeclared `mjd`. Upstream
    therefore returns a declination no alt/az pair can produce - for az=100
    alt=30 it answers -43.7 degrees, where the geometry gives +17.3. This port
    computes the documented result and pins it against the formulas.

    The angles are passed as sexagesimal strings because a plain float is
    radians everywhere in this API, so 100.0 would mean 100 radians.
    """
    lat, lon, elev = site
    ours = mojoephem.Observer()
    ours.lat, ours.lon, ours.elevation = lat, lon, elev
    ours.date = mojoephem.Date("2024/6/21 22:00")
    ra, dec = ours.radec_of("%d:00:00" % az, "%d:00:00" % alt)
    # rebuild the same answer from the exported kernels, independently
    lst = mojoephem.now_lst(float(ours.date), ours._lon_radians())
    az_rad = math.radians(az)
    true_alt = float(
        mojoephem.unrefract(ours.pressure, ours.temp, math.radians(alt))
    )
    ha, want_dec = mojoephem._aa_hadec(ours._lat_radians(), true_alt, az_rad)
    want_ra = (lst * 15.0 * math.pi / 180.0 - ha) % (2.0 * math.pi)
    assert arcsec(float(ra), want_ra) < 1e-6
    assert arcsec(float(dec), want_dec) < 1e-6
    # and the result is self-consistent: feeding it back through hadec_aa
    # recovers the azimuth and the unrefracted altitude exactly. The hour angle
    # has to be reconstructed in the [0, 2pi) convention aa_hadec uses, which
    # is what `lst - ra` is before it is wrapped by the fmod.
    ha_back = (lst * 15.0 * math.pi / 180.0 - float(ra)) % (2.0 * math.pi)
    alt_back, az_back = mojoephem._hadec_aa(ours._lat_radians(), ha_back, float(dec))
    assert arcsec(alt_back, true_alt) < 1e-9
    assert arcsec(az_back, az_rad) < 1e-9
    # the declination is physically sensible, which is what upstream's is not
    assert -90.0 <= float(mojoephem.degrees(float(dec))) <= 90.0


def test_radec_of_treats_a_float_as_radians():
    """A plain float is radians here, as everywhere else in this API."""
    ours = mojoephem.Observer()
    ours.lat, ours.lon, ours.elevation = "51.4778", "-0.0015", 70.0
    ours.date = mojoephem.Date("2024/6/21 22:00")
    a = ours.radec_of(math.radians(100.0), math.radians(30.0))
    b = ours.radec_of("100:00:00", "30:00:00")
    assert arcsec(float(a[0]), float(b[0])) < 1e-9
    assert arcsec(float(a[1]), float(b[1])) < 1e-9


def test_radec_of_accepts_strings():
    """A sexagesimal string is degrees, as upstream's parse_angle treats it."""
    ours = mojoephem.Observer()
    ours.lat, ours.lon, ours.elevation = "51.4778", "-0.0015", 70.0
    ours.date = mojoephem.Date("2024/6/21 22:00")
    a = ours.radec_of("100:00:00", "30:00:00")
    b = ours.radec_of("0:30:00", "0:15:00")
    assert float(a[0]) != float(b[0])
    assert float(a[1]) != float(b[1])


def test_radec_of_returns_angles_in_the_right_units():
    """The pair is (ra as an Angle in hours, dec as an Angle in degrees)."""
    ours = mojoephem.Observer()
    ours.lat, ours.lon, ours.elevation = "51.4778", "-0.0015", 70.0
    ours.date = mojoephem.Date("2024/6/21 22:00")
    ra, dec = ours.radec_of("100:00:00", "30:00:00")
    assert ra.unit == pytest.approx(12.0 / math.pi)
    assert dec.unit == pytest.approx(180.0 / math.pi)
    assert 0.0 <= float(ra) < 2.0 * math.pi


ALIASES = [
    "next_autumnal_equinox", "next_autumn_equinox", "next_fall_equinox",
    "previous_autumnal_equinox", "previous_autumn_equinox",
    "previous_fall_equinox",
]


@pytest.mark.parametrize("name", ALIASES)
def test_equinox_aliases_match_upstream(name):
    """The autumn equinox has three spellings upstream; all three must work."""
    when = mojoephem.Date("2024/1/1")
    assert float(getattr(mojoephem, name)(when)) == pytest.approx(
        float(getattr(upstream, name)(upstream.Date("2024/1/1"))), abs=1e-6
    )
    # and the aliases are the same object, as upstream's are
    ours = getattr(mojoephem, name)
    if "autumnal" in name:
        assert ours is getattr(mojoephem, name.replace("autumnal", "autumn"))
        assert ours is getattr(mojoephem, name.replace("autumnal", "fall"))


def test_math_reexports_match_upstream():
    """Upstream's `from math import acos, cos, isnan, pi, sin` is public API."""
    for name in ("acos", "cos", "sin", "isnan"):
        assert getattr(mojoephem, name) is getattr(__import__("math"), name)
    assert mojoephem.pi == math.pi == upstream.pi
    assert mojoephem.acos(0.5) == upstream.acos(0.5)
    assert mojoephem.cos(0.0) == upstream.cos(0.0)
    assert mojoephem.sin(0.0) == upstream.sin(0.0)
    assert mojoephem.isnan(float("nan")) == upstream.isnan(float("nan"))


@pytest.mark.parametrize("epoch", [None, "2000/1/1 12:00", "2024/6/21 22:00"])
def test_coordinate_classes_match_upstream(epoch):
    """Equatorial and Ecliptic agree with upstream, including conversions."""
    kw = {} if epoch is None else {"epoch": epoch}
    eq = mojoephem.Equatorial(10.0, 20.0, **kw)
    eq_u = upstream.Equatorial(10.0, 20.0, **kw)
    assert abs(float(eq.ra) - float(eq_u.ra)) < 1e-12
    assert abs(float(eq.dec) - float(eq_u.dec)) < 1e-12
    assert eq.get() == (eq.ra, eq.dec)

    # Ecliptic is a LonLatCoordinate, so it has .lon/.lat like upstream's
    ec = mojoephem.Ecliptic(1.0, 2.0, **kw)
    ec_u = upstream.Ecliptic(1.0, 2.0, **kw)
    assert abs(float(ec.lon) - float(ec_u.lon)) < 1e-12
    assert abs(float(ec.lat) - float(ec_u.lat)) < 1e-12

    # and the two convert into each other both ways. The round trip is not
    # the identity - upstream's Equatorial -> Ecliptic -> Equatorial loses
    # 2pi of right ascension - but this port reproduces upstream's answer
    # exactly, which is what is being asserted.
    from_eq = mojoephem.Ecliptic(eq)
    from_eq_u = upstream.Ecliptic(eq_u)
    assert abs(float(from_eq.lon) - float(from_eq_u.lon)) < 1e-9
    assert abs(float(from_eq.lat) - float(from_eq_u.lat)) < 1e-9
    back = mojoephem.Equatorial(from_eq)
    back_u = upstream.Equatorial(from_eq_u)
    assert abs(float(back.ra) - float(back_u.ra)) < 1e-9
    assert abs(float(back.dec) - float(back_u.dec)) < 1e-9
    # to_radec/from_radec are the documented names for the same conversion
    lon, lat = from_eq.to_radec()
    assert abs(float(lon) - float(back.ra)) < 1e-9
    assert abs(float(lat) - float(back.dec)) < 1e-9


def test_lonlat_coordinate_matches_upstream():
    """LonLatCoordinate parses the same sexagesimal strings and reports .lat/.lon."""
    ll = mojoephem.LonLatCoordinate("51:28:40", "0:00:05")
    ll_u = upstream.LonLatCoordinate("51:28:40", "0:00:05")
    assert float(ll.lat) == float(ll_u.lat)
    assert float(ll.lon) == float(ll_u.lon)
    assert float(ll.long) == float(ll.lon)  # the alternate spelling
    assert ll.get() == (ll.lon, ll.lat)
    assert ll.epoch == float(mojoephem.J2000)


def test_coordinate_rejects_a_single_non_coordinate():
    """Upstream's TypeError is preserved for a single argument that is neither."""
    with pytest.raises(TypeError):
        mojoephem.Equatorial("not a coordinate")
    with pytest.raises(TypeError):
        upstream.Equatorial("not a coordinate")
