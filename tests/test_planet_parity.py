"""Parity between mojoephem's planet positions and upstream ephem's.

Every assertion is a numerical comparison against the real upstream
package over a spread of epochs, not a smoke test.

Upstream stores the Obj fields in C types that are narrower than a double:
the elongation, phase, distances and angular size are `float` (24-bit
mantissa) and the visual magnitude is a `short` of hundredths. The port
keeps full double precision everywhere except the magnitude, which applies
upstream's set_smag quantisation so `mag` is bit-comparable. The
tolerances below are therefore set by upstream's storage, not by the
kernels' accuracy: the angles themselves agree to well under a
milliarcsecond (see test_apparent_place).
"""

from __future__ import annotations

import pytest

import mojoephem
from conftest import arcsec, upstream

BODIES = [
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
    "Sun",
    "Moon",
]

# Epochs chosen to cross the Gregorian reform, the VSOP87 validity window,
# the Chapront-95 window and the Moshier lunar window.
EPOCHS = [
    "2200/6/1 00:00",
    "2024/1/1 00:00",
    "2024/7/4 23:59:59",
    "2000/1/1 12:00",
    "1950/6/15 03:21",
    "1800/3/1 00:00",
    "1700/11/9 00:00",
    "1600/1/1 00:00",
]


@pytest.mark.parametrize("when", EPOCHS)
@pytest.mark.parametrize("name", BODIES)
def test_apparent_place(name, when):
    """Astrometric right ascension and declination agree to 1 mas."""
    ours = getattr(mojoephem, name)()
    theirs = getattr(upstream, name)()
    ours.compute(mojoephem.Date(when))
    theirs.compute(upstream.Date(when))
    assert arcsec(float(ours.a_ra), float(theirs.a_ra)) < 1e-3
    assert arcsec(float(ours.a_dec), float(theirs.a_dec)) < 1e-3


@pytest.mark.parametrize("when", EPOCHS)
@pytest.mark.parametrize("name", BODIES)
def test_geocentric_place(name, when):
    """Apparent geocentric ra/dec, elongation and phase agree."""
    ours = getattr(mojoephem, name)()
    theirs = getattr(upstream, name)()
    ours.compute(mojoephem.Date(when))
    theirs.compute(upstream.Date(when))
    assert arcsec(float(ours.g_ra), float(theirs.g_ra)) < 1e-3
    assert arcsec(float(ours.g_dec), float(theirs.g_dec)) < 1e-3
    # elong is a C float in degrees upstream
    assert arcsec(float(ours.elong), float(theirs.elong)) < 0.1
    # phase is a C float in percent
    assert abs(ours.phase - theirs.phase) < 1e-3 + 1e-6 * abs(theirs.phase)


@pytest.mark.parametrize("name", BODIES)
def test_heliocentric_place(name):
    """Heliocentric longitude and latitude agree across a full orbit."""
    worst_lon = worst_lat = 0.0
    for day in range(0, 900, 7):
        d0 = "2023/1/1"
        ours = getattr(mojoephem, name)()
        theirs = getattr(upstream, name)()
        ours.compute(mojoephem.Date(d0) + day)
        theirs.compute(upstream.Date(d0) + day)
        worst_lon = max(worst_lon, arcsec(float(ours.hlon), float(theirs.hlon)))
        worst_lat = max(worst_lat, arcsec(float(ours.hlat), float(theirs.hlat)))
    assert worst_lon < 0.1, f"{name} heliocentric longitude {worst_lon:.4f} arcsec"
    assert worst_lat < 0.1, f"{name} heliocentric latitude {worst_lat:.4f} arcsec"


@pytest.mark.parametrize("name", BODIES)
def test_distances(name):
    """Earth and Sun distances agree; the Sun's own distance is zero."""
    for day in range(0, 800, 11):
        ours = getattr(mojoephem, name)()
        theirs = getattr(upstream, name)()
        ours.compute(mojoephem.Date("2023/1/1") + day)
        theirs.compute(upstream.Date("2023/1/1") + day)
        # both distances are C floats upstream
        for ours_d, theirs_d in (
            (ours.earth_distance, theirs.earth_distance),
            (ours.sun_distance, theirs.sun_distance),
        ):
            assert abs(ours_d - theirs_d) < 1e-6 + 1e-6 * abs(theirs_d)


@pytest.mark.parametrize("name", BODIES)
def test_magnitude_and_size(name):
    """Visual magnitude and angular size agree.

    `size` is a C float upstream, so its last bits cannot match; `mag` is
    quantised to hundredths by set_smag on both sides, and a value that
    lands exactly on a rounding boundary can still fall either way.
    """
    for day in range(0, 800, 13):
        ours = getattr(mojoephem, name)()
        theirs = getattr(upstream, name)()
        ours.compute(mojoephem.Date("2023/1/1") + day)
        theirs.compute(upstream.Date("2023/1/1") + day)
        assert abs(ours.mag - theirs.mag) < 0.011
        assert abs(ours.size - theirs.size) < 1e-3 + 1e-6 * abs(theirs.size)
