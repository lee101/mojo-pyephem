"""Parity of the topocentric pipeline: Observer, alt/az, rise/set, transit."""

from __future__ import annotations

import pytest

import mojoephem
from conftest import arcsec, upstream

SITES = [
    ("51.4778", "0.0015", 70.0, 1010.0, 10.0),  # Greenwich
    ("-33.8688", "151.2093", 58.0, 1010.0, 20.0),  # Sydney
    ("35.68", "139.69", 40.0, 1010.0, 15.0),  # Tokyo
    ("-54.8", "-68.3", 20.0, 990.0, 5.0),  # Ushuaia
    ("78.22", "15.63", 100.0, 1000.0, -5.0),  # Longyearbyen
]

BODIES = ["Sun", "Moon", "Venus", "Mars", "Jupiter", "Saturn", "Uranus"]


def observers(site, when):
    lat, lon, elev, pressure, temp = site
    ours = mojoephem.Observer()
    ours.lat, ours.lon = lat, lon
    ours.elevation, ours.pressure, ours.temp = elev, pressure, temp
    ours.date = when
    theirs = upstream.Observer()
    theirs.lat, theirs.lon = lat, lon
    theirs.elevation, theirs.pressure, theirs.temp = elev, pressure, temp
    theirs.date = when
    return ours, theirs


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("name", BODIES)
def test_alt_az(site, name):
    """Altitude and azimuth agree.

    Upstream stores both as C floats in radians, whose resolution near 2pi
    is about 0.08 arcsecond, so that is the floor here; the underlying
    altitude/azimuth computation is far tighter than that.
    """
    when = mojoephem.Date("2024/6/21 22:00")
    ours_o, theirs_o = observers(site, when)
    ours, theirs = getattr(mojoephem, name)(), getattr(upstream, name)()
    ours.compute(ours_o)
    theirs.compute(theirs_o)
    assert arcsec(float(ours.alt), float(theirs.alt)) < 0.1
    assert arcsec(float(ours.az), float(theirs.az)) < 0.1


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("name", BODIES)
def test_hour_angle_and_distance(site, name):
    """Hour angle and the topocentric distance agree."""
    when = mojoephem.Date("2024/6/21 22:00")
    ours_o, theirs_o = observers(site, when)
    ours, theirs = getattr(mojoephem, name)(), getattr(upstream, name)()
    ours.compute(ours_o)
    theirs.compute(theirs_o)
    assert arcsec(float(ours.ha), float(theirs.ha)) < 1e-2
    # s_edist is a C float upstream
    assert abs(ours.earth_distance - theirs.earth_distance) < 1e-9 + 1e-6 * abs(
        theirs.earth_distance
    )


SEARCHES = [
    "next_rising",
    "next_setting",
    "next_transit",
    "previous_rising",
    "previous_setting",
    "previous_transit",
    "next_antitransit",
    "previous_antitransit",
]


@pytest.mark.parametrize("site", SITES[:3])
@pytest.mark.parametrize("name", BODIES)
@pytest.mark.parametrize("method", SEARCHES)
def test_rise_set_search(site, name, method):
    """Rise, set and transit searches agree to a tenth of a second."""
    start = mojoephem.Date("2024/6/21 00:00")
    ours_o, theirs_o = observers(site, start)
    ours, theirs = getattr(mojoephem, name)(), getattr(upstream, name)()
    try:
        a = float(getattr(ours_o, method)(ours, start=start))
    except mojoephem.CircumpolarError as exc:
        # upstream must refuse for the same reason at the same moment
        with pytest.raises(upstream.CircumpolarError) as other:
            getattr(theirs_o, method)(theirs, start=start)
        assert type(other.value) is type(exc)
        return
    b = float(getattr(theirs_o, method)(theirs, start=start))
    assert abs(a - b) * 86400.0 < 0.1


def test_circumpolar_errors_match():
    """A body that never rises raises the same exception as upstream."""
    ours_o, theirs_o = observers(("80.0", "0.0", 0.0, 1010.0, 15.0), mojoephem.Date("2024/1/1"))
    ours, theirs = mojoephem.Sun(), upstream.Sun()
    with pytest.raises(mojoephem.NeverUpError):
        ours_o.next_rising(ours, start=mojoephem.Date("2024/1/1"))
    with pytest.raises(upstream.NeverUpError):
        theirs_o.next_rising(theirs, start=upstream.Date("2024/1/1"))


def test_disallow_circumpolar():
    """The declination argument is a plain angle in radians, as upstream.

    At a latitude of one radian, a declination of -1 radian is never up
    and +1 radian is always up.
    """
    ours = mojoephem.Observer()
    ours.lat = mojoephem.degrees(1.0)
    theirs = upstream.Observer()
    theirs.lat = 1.0
    with pytest.raises(mojoephem.NeverUpError):
        ours.disallow_circumpolar(-1.0)
    with pytest.raises(mojoephem.AlwaysUpError):
        ours.disallow_circumpolar(1.0)
    with pytest.raises(upstream.NeverUpError):
        theirs.disallow_circumpolar(-1.0)
    with pytest.raises(upstream.AlwaysUpError):
        theirs.disallow_circumpolar(1.0)
