"""mojo-pyephem against upstream ephem on the same inputs.

Always run this through `pixi run bench`, never as `python bench/bench.py`:
the pixi task holds a machine-wide flock so concurrent factory jobs cannot
distort the numbers. Every number printed here is measured; nothing is
estimated.

One trap worth naming: upstream's `Body.compute()` is a statement that
returns None. It records the date and marks the body dirty; the ephemeris
itself is evaluated on the first attribute read. A benchmark that calls
`compute()` in a loop without reading a field is timing Python's `__init__`,
not the astronomy. Every case below reads the fields it is asking for.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import ephem  # noqa: E402  upstream
import mojoephem as msk  # noqa: E402

PLCODE = {name: getattr(msk, name).__planet__ for name in
          ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus",
           "Neptune", "Pluto", "Sun", "Moon")}


def timeit(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _dates(n: int) -> np.ndarray:
    start = float(msk.Date("2023/1/1"))
    return np.linspace(start, start + n, n)


def _read(body) -> float:
    """Force the ephemeris, and touch the fields we are timing."""
    return float(body.a_ra) + float(body.earth_distance)


def _upstream(name: str):
    return getattr(ephem, name)()


def _theirs(name: str, dates) -> float:
    body = _upstream(name)
    total = 0.0
    for x in dates:
        body.compute(ephem.Date(x))
        total += _read(body)
    return total


def _theirs_observer(name: str, dates, site) -> float:
    body = _upstream(name)
    observer = ephem.Observer()
    observer.lat, observer.lon, observer.elevation = site
    total = 0.0
    for x in dates:
        observer.date = ephem.Date(x)
        body.compute(observer)
        total += _read(body)
    return total


def _ours(name: str, dates, site=None) -> float:
    out = msk.compute_series(PLCODE[name], dates, site)
    return float(out[:, 8].sum() + out[:, 3].sum())


def _site_mine(site):
    o = msk.Observer()
    o.lat, o.lon, o.elevation = site
    return o


CASES = []


def case(name):
    """Register a benchmark; the decorated function returns
    (theirs_lambda, ours_lambda) in that order."""

    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


BATCH = 20_000
SITE = (51.5, -0.13, 70.0)


@case("Sun.compute, 20000 dates (batched)")
def _():
    d = _dates(BATCH)
    return (lambda: _theirs("Sun", d), lambda: _ours("Sun", d))


@case("Moon.compute, 20000 dates (batched)")
def _():
    d = _dates(BATCH)
    return (lambda: _theirs("Moon", d), lambda: _ours("Moon", d))


@case("Saturn.compute, 20000 dates (batched)")
def _():
    d = _dates(BATCH)
    return (lambda: _theirs("Saturn", d), lambda: _ours("Saturn", d))


@case("Mars.compute, 20000 dates (batched)")
def _():
    d = _dates(BATCH)
    return (lambda: _theirs("Mars", d), lambda: _ours("Mars", d))


@case("Sun.compute, 20000 dates, topocentric (batched)")
def _():
    d = _dates(BATCH)
    mine = _site_mine(SITE)
    return (
        lambda: _theirs_observer("Sun", d, SITE),
        lambda: _ours("Sun", d, mine),
    )


@case("Sun.compute, 20000 calls, one date at a time")
def _():
    dates = _dates(20_000)
    body, sun = _upstream("Sun"), msk.Sun()
    theirs_dates = [ephem.Date(x) for x in dates]
    mine_dates = [msk.Date(x) for x in dates]
    return (
        lambda: [body.compute(d) or _read(body) for d in theirs_dates],
        lambda: [_read(sun.compute(d)) for d in mine_dates],
    )


@case("Sun.compute, 2000 calls, one date at a time")
def _():
    dates = _dates(2_000)
    body, sun = _upstream("Sun"), msk.Sun()
    theirs_dates = [ephem.Date(x) for x in dates]
    mine_dates = [msk.Date(x) for x in dates]
    return (
        lambda: [body.compute(d) or _read(body) for d in theirs_dates],
        lambda: [_read(sun.compute(d)) for d in mine_dates],
    )


@case("Observer.next_rising(Sun), 2000 searches")
def _():
    mine = _site_mine(SITE)
    theirs = ephem.Observer()
    theirs.lat, theirs.lon, theirs.elevation = SITE
    start = msk.Date("2024/3/15 12:00")
    start_theirs = ephem.Date("2024/3/15 12:00")
    sun, sun_theirs = msk.Sun(), ephem.Sun()
    return (
        lambda: [theirs.next_rising(sun_theirs, start=start_theirs) for _ in range(2_000)],
        lambda: [mine.next_rising(sun, start=start) for _ in range(2_000)],
    )


@case("delta_t, 200000 dates")
def _():
    d = _dates(200_000)
    return (
        lambda: [ephem.delta_t(ephem.Date(x)) for x in d],
        lambda: [msk.delta_t(x) for x in d],
    )


@case("precess, 200000 pairs")
def _():
    ra = np.linspace(0.0, 6.28, 200_000)
    dec = np.linspace(-1.5, 1.5, 200_000)
    pairs = list(zip(ra, dec))
    return (
        lambda: [ephem.precess(ephem.J2000, 45290.5, a, b) for a, b in pairs],
        lambda: [msk.precess(msk.J2000, 45290.5, a, b) for a, b in pairs],
    )


@case("separation, 200000 pairs")
def _():
    ra = np.linspace(0.0, 6.28, 200_000)
    dec = np.linspace(-1.5, 1.5, 200_000)
    pairs = list(zip(ra, dec))
    return (
        lambda: [ephem.separation((a, b), (a + 0.1, b)) for a, b in pairs],
        lambda: [msk.separation((a, b), (a + 0.1, b)) for a, b in pairs],
    )


def main() -> None:
    print(f"machine: {platform.platform()}")
    print(f"python:  {platform.python_version()}  numpy: {np.__version__}")
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                print(
                    f"cpu:     {line.split(':', 1)[1].strip()}"
                    f" ({os.cpu_count()} threads)"
                )
                break
    print(f"upstream ephem {ephem.__version__}; mojo-pyephem {msk.__version__}")
    print()
    print("| case | mojo-pyephem | upstream ephem | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, build in CASES:
        theirs_fn, ours_fn = build()
        ours = timeit(ours_fn)
        upstream = timeit(theirs_fn)
        note = "" if ours <= upstream else " (mojo slower)"
        print(
            f"| {name} | {ours * 1e3:.1f} ms | {upstream * 1e3:.1f} ms |"
            f" {upstream / ours:.2f}x{note} |"
        )


if __name__ == "__main__":
    main()
