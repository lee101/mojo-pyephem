#!/usr/bin/env python3
"""Transcribe upstream libastro's numeric tables into one flat data blob.

The Mojo shared library cannot hold read-only comptime tables (they land in
`.text` with PC-32 relocations, which `ld` refuses for a shared object), so
the coefficient tables travel over the ABI as `Int` addresses. This script
packs them into `python/mojoephem/data/tables.bin` plus a JSON manifest that
`mojoephem._tables` maps onto addresses at import time.

Usage:
    tools/gen_tables.py <path-to-ephem-sdist> [out.bin out.json]

Every number is copied from the C source verbatim; nothing is refitted.
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

VSOP_BODIES = [
    "mercury",
    "venus",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]

NUT_SERIES = 106


def strip_comments(src: str) -> str:
    return re.sub(r"/\*.*?\*/", "", src, flags=re.S)

def block(src: str, name: str) -> str:
    # Strip first: upstream comments name the very tables we are looking for
    # (e.g. "appended to the dt[] table"), and a raw regex would land there.
    m = re.search(
        rf"\b{name}\s*(\[[^=]*\])?\s*=\s*\{{(.*?)\n?\}};", strip_comments(src), re.S
    )
    if m is None:
        raise SystemExit(f"table {name} not found")
    return m.group(2)


def rows(src: str, name: str, width: int) -> list[list[int]]:
    """Rows of exactly `width` integers; narrower rows are C terminators."""
    return [
        [int(v) for v in row.group(1).split(",") if v.strip()]
        for row in re.finditer(r"\{([^{}]*)\}", block(src, name))
        if len([v for v in row.group(1).split(",") if v.strip()]) == width
    ]


def numbers(src: str, name: str) -> list[float]:
    body_ = block(src, name)
    return [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body_)]


def vsop_tables(src: str) -> dict:
    out: dict[str, tuple[str, list]] = {}
    for b in VSOP_BODIES:
        tr = []
        for row in re.finditer(r"\{([^{}]*)\}", block(src, f"vx_{b}")):
            tr += [float(v) for v in row.group(1).split(",") if v.strip()]
        out[f"vsop_vx_{b}"] = ("f8", tr)
        bn = []
        for row in rows(src, f"vn_{b}", 3):
            bn += row
        # The C terminator row { 0, } is what stops the alpha loop; a zero
        # row after the "end" row reproduces it exactly.
        bn += [0, 0, 0]
        out[f"vsop_vn_{b}"] = ("i4", bn)
    return out


def nutation_tables(src: str) -> dict:
    mult = [v for r in rows(src, "multarg", 5) for v in r]
    sec = [v for r in rows(src, "ampsecul", 5) for v in r]
    con = [v for r in rows(src, "ampconst", 2) for v in r]
    for name, table, want in (
        ("nut_multarg", mult, NUT_SERIES * 5),
        ("nut_ampsecul", sec, 15 * 5),
        ("nut_ampconst", con, NUT_SERIES * 2),
    ):
        if len(table) != want:
            raise SystemExit(f"{name}: {len(table)} values, expected {want}")
    return {
        "nut_delaunay": ("f8", numbers(src, "delaunay")),
        "nut_multarg": ("i4", mult),
        "nut_ampsecul": ("i4", sec),
        "nut_ampconst": ("i4", con),
    }


def moon_tables(src: str) -> dict:
    out: dict[str, tuple[str, list]] = {}
    raw = {"moonlr": "lr", "moonlat": "b"}
    for name, pre in raw.items():
        m = re.search(rf"struct plantbl {name}\s*=\s*\{{(.*?)\n\}};", strip_comments(src), re.S)
        if m is None:
            raise SystemExit(f"struct plantbl {name} not found")
        head = re.match(r"\s*\{([^}]*)\},\s*(\d+),", m.group(1))
        if head is None:
            raise SystemExit(f"{name}: unexpected plantbl header")
        out[f"moon_{name}_harmonic"] = ("i4", [int(v) for v in head.group(1).split(",") if v.strip()])
        out[f"moon_{name}_power"] = ("i4", [int(head.group(2))])
        for suffix in ("args", "tabl", "tabb", "tabr"):
            vals = [int(v) for v in re.findall(r"[-+]?\d+", block(src, pre + suffix))]
            out[f"moon_{name}_{suffix}"] = ("i4", vals)
        tail = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1)[head.end() :])
        if len(tail) != 3:
            raise SystemExit(f"{name}: expected 3 scalars, got {tail}")
        out[f"moon_{name}_scalars"] = ("f8", [float(v) for v in tail])
    return out


def deltat_tables(src: str) -> dict:
    ms = [int(v) for v in re.findall(r"[-+]?\d+", block(src, "m_s"))]
    dt = [int(v) for v in re.findall(r"[-+]?\d+", block(src, "dt"))]
    if (len(ms), len(dt)) != (28, 399):
        raise SystemExit(f"deltat sizes: m_s={len(ms)} dt={len(dt)}")
    return {"deltat_ms": ("i4", ms), "deltat_dt": ("i4", dt)}



CHAP_BODIES = ["jupiter", "saturn", "uranus", "neptune", "pluto"]


def chap95_tables(src: str) -> dict:
    """Flat out of `chap95_rec[]`: n, six amplitudes, and Nu (n == 0 only)."""
    out: dict[str, tuple[str, list]] = {}
    for b in CHAP_BODIES:
        body_ = block(src, f"chap95_{b}")
        ns: list[int] = []
        amp: list[float] = []
        nu: list[float] = []
        for row in re.finditer(r"\{\s*(-?\d+)\s*,\s*\{([^{}]*)\}\s*,?\s*([^{}]*?)\s*\}", body_):
            ns.append(int(row.group(1)))
            amp += [float(v) for v in row.group(2).split(",") if v.strip()]
            nu.append(float(row.group(3).strip().rstrip(",") or 0.0))
        if len(amp) != 6 * len(ns):
            raise SystemExit(f"chap95_{b}: {len(ns)} records, {len(amp)} amplitudes")
        out[f"chap95_n_{b}"] = ("i4", ns)
        out[f"chap95_amp_{b}"] = ("f8", amp)
        out[f"chap95_nu_{b}"] = ("f8", nu)
    return out


def pack(tables: dict) -> tuple[bytes, list[dict]]:
    manifest: list[dict] = []
    blobs: list[bytes] = []
    offset = 0
    for name, (dtype, values) in tables.items():
        code = "d" if dtype == "f8" else "i"
        raw = struct.pack(f"<{len(values)}{code}", *values)
        manifest.append({"name": name, "dtype": dtype, "offset": offset, "count": len(values)})
        blobs.append(raw)
        offset += len(raw)
    return b"".join(blobs), manifest


def main() -> int:
    sdist = Path(sys.argv[1] if len(sys.argv) > 1 else "ephem-4.2.1")
    out_bin = Path(sys.argv[2] if len(sys.argv) > 2 else "python/mojoephem/data/tables.bin")
    out_json = Path(sys.argv[3] if len(sys.argv) > 3 else "python/mojoephem/data/tables.json")
    lib = sdist / "libastro"

    tables: dict[str, tuple[str, list]] = {}
    tables.update(vsop_tables((lib / "vsop87_data.c").read_text()))
    tables.update(nutation_tables((lib / "nutation.c").read_text()))
    tables.update(moon_tables((lib / "moon.c").read_text()))
    tables.update(chap95_tables((lib / "chap95_data.c").read_text()))
    tables.update(deltat_tables((lib / "deltat.c").read_text()))

    blob, manifest = pack(tables)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(blob)
    out_json.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest)} tables, {len(blob)} bytes -> {out_bin}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
