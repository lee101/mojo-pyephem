"""Builds the table registry the Mojo kernels index by slot.

`mojo build --emit shared-lib` cannot hold read-only comptime tables in a
shared object, so the VSOP87D, Chapront-95, IAU-1980 nutation, Moshier
lunar and deltaT tables ship as one flat blob and cross the ABI as `Int`
addresses. `registry()` returns the `int64` vector whose slot i is the base
address of table i; the slot order is the manifest order of
`tools/gen_tables.py` and is asserted below against `data/tables.json`.

Regenerate the blob from an ephem sdist with:
    tools/gen_tables.py /path/to/ephem-4.2.1
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

# Slot order, mirrored from the comment block at the top of
# src/ported.mojo. The assert below fails loudly if the two drift.
SLOTS = (
    "vsop_vx_mercury",
    "vsop_vn_mercury",
    "vsop_vx_venus",
    "vsop_vn_venus",
    "vsop_vx_earth",
    "vsop_vn_earth",
    "vsop_vx_mars",
    "vsop_vn_mars",
    "vsop_vx_jupiter",
    "vsop_vn_jupiter",
    "vsop_vx_saturn",
    "vsop_vn_saturn",
    "vsop_vx_uranus",
    "vsop_vn_uranus",
    "vsop_vx_neptune",
    "vsop_vn_neptune",
    "nut_delaunay",
    "nut_multarg",
    "nut_ampsecul",
    "nut_ampconst",
    "moon_moonlr_harmonic",
    "moon_moonlr_power",
    "moon_moonlr_args",
    "moon_moonlr_tabl",
    "moon_moonlr_tabb",
    "moon_moonlr_tabr",
    "moon_moonlr_scalars",
    "moon_moonlat_harmonic",
    "moon_moonlat_power",
    "moon_moonlat_args",
    "moon_moonlat_tabl",
    "moon_moonlat_tabb",
    "moon_moonlat_tabr",
    "moon_moonlat_scalars",
    "chap95_n_jupiter",
    "chap95_amp_jupiter",
    "chap95_nu_jupiter",
    "chap95_n_saturn",
    "chap95_amp_saturn",
    "chap95_nu_saturn",
    "chap95_n_uranus",
    "chap95_amp_uranus",
    "chap95_nu_uranus",
    "chap95_n_neptune",
    "chap95_amp_neptune",
    "chap95_nu_neptune",
    "chap95_n_pluto",
    "chap95_amp_pluto",
    "chap95_nu_pluto",
    "deltat_ms",
    "deltat_dt",
)


class TableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def tables() -> dict[str, np.ndarray]:
    manifest = json.loads((DATA / "tables.json").read_text())
    names = tuple(entry["name"] for entry in manifest)
    if names != SLOTS:
        raise TableError(
            "table manifest does not match the slot order compiled into "
            "src/ported.mojo; regenerate both with tools/gen_tables.py"
        )
    blob = np.frombuffer((DATA / "tables.bin").read_bytes(), dtype=np.uint8)
    out: dict[str, np.ndarray] = {}
    for entry in manifest:
        dtype = np.float64 if entry["dtype"] == "f8" else np.int32
        item = np.dtype(dtype).itemsize
        start = entry["offset"]
        out[entry["name"]] = blob[start : start + entry["count"] * item].view(dtype)
    return out


@lru_cache(maxsize=1)
def registry() -> np.ndarray:
    """An int64 vector whose slot i holds the base address of table i."""
    reg = np.zeros(len(SLOTS), dtype=np.int64)
    for i, name in enumerate(SLOTS):
        reg[i] = tables()[name].ctypes.data
    return reg


@lru_cache(maxsize=1)
def reg() -> int:
    """The registry address, as the C ABI wants it.

    Cached: numpy builds a fresh ctypes object for `.ctypes.data` on every
    access, which costs more than the kernel call it feeds.
    """
    return int(registry().ctypes.data)
