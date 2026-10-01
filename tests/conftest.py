"""Shared fixtures and the upstream-parity skip."""

from __future__ import annotations

import math
import os

# numpy's OpenBLAS spawns a thread per core the first time it is imported,
# which overflows the pids.max of a container/cgroup sandbox and kills the
# run mid-import with KeyboardInterrupt. Bound it before anything pulls
# numpy in; an explicit value in the environment still wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import pytest

import mojoephem

# Upstream pyephem is installed as the `ephem` package (the `pyephem`
# distribution on PyPI is only a stub that depends on it).
upstream = pytest.importorskip("ephem", reason="upstream ephem not installed")

ARCSEC = 180.0 * 3600.0 / math.pi


def arcsec(a: float, b: float) -> float:
    """The absolute separation of two angles, in arcseconds."""
    d = (a - b) % (2.0 * math.pi)
    return math.degrees(min(d, 2.0 * math.pi - d)) * 3600.0


def both(cls_name: str, when, when_upstream=None):
    """A computed (ours, theirs) pair for one body class."""
    ours = getattr(mojoephem, cls_name)()
    theirs = getattr(upstream, cls_name)()
    ours.compute(when)
    theirs.compute(when if when_upstream is not None else _as_upstream(when))
    return ours, theirs


def _as_upstream(when):
    if isinstance(when, mojoephem.Observer):
        return when
    return upstream.Date(str(when))


@pytest.fixture(scope="session", autouse=True)
def _library():
    """Fail loudly if the shared library cannot be loaded."""
    mojoephem._lib.lib()
