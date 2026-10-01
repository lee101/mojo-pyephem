"""Loads (and if necessary builds) the compiled Mojo library.

Every entry point takes buffers as `Int` addresses and returns plain
scalars, so the whole surface is `ctypes` + numpy views of caller-owned
arrays.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LIB = Path(os.environ.get("MOJOPHYPHEM_LIB") or ROOT / "dist" / "libmojo-pyephem.so")

I = ctypes.c_int64
F = ctypes.c_double
P = ctypes.c_void_p

# name -> (argtypes, restype)
_SIGNATURES: dict[str, tuple[list, object]] = {
    "mpe_range": ([P, F], None),
    "mpe_radecrange": ([P, P], None),
    "mpe_mjd_day": ([F], F),
    "mpe_mjd_hr": ([F], F),
    "mpe_cal_mjd": ([I, F, I], F),
    "mpe_mjd_cal": ([F, P, P, P], None),
    "mpe_year_mjd": ([F], F),
    "mpe_mjd_year": ([F], F),
    "mpe_isleapyear": ([I], I),
    "mpe_mjd_dow": ([F], I),
    "mpe_deltat": ([F, P], F),
    "mpe_mm_mjed": ([F, P], F),
    "mpe_sphcart": ([F, F, F, P], None),
    "mpe_cartsph": ([F, F, F, P], None),
    "mpe_obliquity": ([F, P], None),
    "mpe_reduce_elements": ([F, F, F, F, F, P], None),
    "mpe_anomaly": ([F, F, P], None),
    "mpe_precess": ([F, F, P], None),
    "mpe_eq_ecl": ([F, F, F, P], None),
    "mpe_ecl_eq": ([F, F, F, P], None),
    "mpe_nutation": ([F, P, P], None),
    "mpe_nut_eq": ([F, P, P], None),
    "mpe_ab_ecl": ([F, F, P], None),
    "mpe_ab_eq": ([F, F, P], None),
    "mpe_vsop87": ([F, I, F, P, P], None),
    "mpe_chap95": ([F, I, F, P, P], None),
    "mpe_sunpos": ([F, P, P], None),
    "mpe_pluto_ell": ([F, P], None),
    "mpe_planpos": ([F, I, F, P, P], None),
    "mpe_satrings": ([F, F, F, F, F, F, P], None),
    "mpe_plans": ([F, I, P, P], None),
    "mpe_moon": ([F, P, P], None),
    "mpe_solve_sphere": ([F, F, F, F, P], None),
    "mpe_aa_hadec": ([F, F, F, P], None),
    "mpe_hadec_aa": ([F, F, F, P], None),
    "mpe_geoc_lat": ([F], F),
    "mpe_ta_par": ([F, F, F, F, P, P], None),
    "mpe_refract": ([F, F, F, P], None),
    "mpe_unrefract": ([F, F, F, P], None),
    "mpe_utc_gst": ([F, F, P], None),
    "mpe_gst_utc": ([F, F, P], None),
    "mpe_gmst0": ([F], F),
    "mpe_now_lst": ([F, F, P, P], None),
    "mpe_radec2ha": ([F, F], F),
    "mpe_elongation": ([F, F, F, P], None),
    "mpe_deflect": ([F, F, F, F, F, F, P, P], None),
    "mpe_obj_planet": ([F, I, F, F, F, F, I, F, F, P, P], None),
    "mpe_obj_planet_series": ([P, P, I, I, F, F, F, F, I, F, F, P], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOPHYPHEM_MOJO")
    if override:
        return override.split()
    for candidate in (ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo",):
        if candidate.exists():
            return [str(candidate)]
    found = os.environ.get("PATH", "").split(os.pathsep)
    for d in found:
        candidate = Path(d) / "mojo"
        if candidate.exists():
            return [str(candidate)]
    raise BuildError("mojo not found; set MOJOPHYPHEM_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/ported.mojo` into `dist/libmojo-pyephem.so` if it is stale."""
    target = LIB
    entry = SRC / "ported.mojo"
    if target.exists() and not force and target.stat().st_mtime >= entry.stat().st_mtime:
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        str(entry),
        "-o",
        str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(proc.stdout + proc.stderr)
    return str(target)


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib



def main() -> int:
    """`python -m mojoephem._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
