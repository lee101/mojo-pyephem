# mojo-pyephem

PyEphem's solar-system ephemeris, ported from C to Mojo.

`ephem` is Brandon Rhodes' Python binding for XEphem's `libastro`, a C
library of astronomical routines. This repo takes the compute-bound core of
that library - the VSOP87D planetary theory, Chapront-95 for the outer
planets, Moshier's truncated ELP lunar theory, IAU-1980 nutation, the
sidereal-time and coordinate machinery around them - and reimplements it in
Mojo, behind a Python API that mirrors upstream's names, argument order and
defaults.

It is a drop-in replacement for the covered subset: the same classes, the
same methods, the same numbers.

## Install

```bash
pixi install
pixi run build      # src/ported.mojo -> dist/libmojo-pyephem.so
pixi run test       # 776 tests against the real upstream ephem
pixi run bench      # benchmarks, printed as a markdown table
```

`python` is put on `PYTHONPATH` by the pixi activation, so `import
mojoephem` works inside any `pixi run` task. Outside pixi, add `python/` to
`PYTHONPATH` yourself.

## Usage

```python
import mojoephem

d = mojoephem.Date('2024/6/21 22:00')

sun = mojoephem.Sun()
sun.compute(d)
print(sun.a_ra, sun.a_dec)            # 6:02:53.77  23:26:03.32
print(sun.earth_distance)             # 1.0162611222117512

moon = mojoephem.Moon()
moon.compute(d)
print(moon.a_ra, moon.phase)          # 17:55:32.55  99.79852791765525

obs = mojoephem.Observer()
obs.lat, obs.lon, obs.elevation = '51.4778', '-0.0015', 70.0
obs.date = d
moon.compute(obs)
print(moon.alt, moon.az)              # 5:29:38.52  154:50:28.62
print(obs.next_rising(moon))          # 2024/6/22 21:38:28

# and the batch entry point, which is the shape the kernel is fastest in:
import numpy as np
out = mojoephem.compute_series(mojoephem.Moon.__planet__,
                               np.linspace(float(d), float(d) + 30, 4000))
ra = [mojoephem.hours(x) for x in out[:, 8]]
print(ra[0], ra[-1])                  # 17:57:05.17  20:38:58.79
```

Angles are `Angle`, a float of radians that remembers its unit and prints
sexagesimally, exactly as upstream's. `Date` is a float of days since
1900 January 0.5 UTC and accepts the same string forms.

## What is covered

Ported from `libastro`, function by function, keeping upstream's names,
argument order, branch structure and arithmetic order. `src/ported.mojo`
reads top to bottom in upstream's own file order.

| upstream file | functions |
| --- | --- |
| `mjd.c` | `cal_mjd`, `mjd_cal`, `mjd_day`, `mjd_hr`, `mjd_year`, `year_mjd`, `mjd_dow`, `isleapyear`, `range`, `radecrange` |
| `deltat.c` | `deltat` (Morrison & Stephenson and Astronomical Almanac tables, Besselian interpolation, long-term polynomial) |
| `auxil.c` | `mm_mjed` |
| `sphcart.c` | `sphcart`, `cartsph` |
| `obliq.c` | `obliquity` |
| `reduce.c` | `reduce_elements` |
| `anomaly.c` | `anomaly` |
| `precess.c` | `precess` (rigorous, Astronomical Ephemeris 1989 B18) |
| `eq_ecl.c` | `eq_ecl`, `ecl_eq` |
| `nutation.c` | `nutation` (IAU 1980, 106 terms), `nut_eq` |
| `aberration.c` | `ab_ecl`, `ab_eq` |
| `vsop87.c` | `vsop87` (VSOP87D, all eight bodies) |
| `chap95.c` | `chap95` (Jupiter through Pluto) |
| `sun.c` | `sunpos` |
| `plans.c` | `plans`, `planpos`, `pluto_ell`, `chap_trans`, `vis_elements` |
| `rings.c` | `satrings` |
| `moon.c` | `moon`, `moon_fast`, `gecmoon`, `g2plan`, `g1plan`, `mean_elements`, `mods3600`, `sscc` |
| `misc.c` | `solve_sphere`, `now_lst`, `radec2ha` |
| `aa_hadec.c` | `aa_hadec`, `hadec_aa`, `aaha_aux`, `geoc_lat` |
| `parallax.c` | `ta_par` |
| `refract.c` | `refract`, `unrefract`, `unrefractLT15`, `unrefractGE15` |
| `utc_gst.c` | `utc_gst`, `gst_utc`, `gmst0` |
| `circum.c` | `obj_planet`, `cir_pos`, `cir_sky`, `sun_cir`, `moon_cir`, `elongation`, `deflect` |

On the Python side, mirroring upstream's `ephem/__init__.py` and the
`_libastro` C extension: `Date` (with `triple`, `tuple` and `datetime`),
`Angle` (with `norm` and `znorm`), `Observer` (with `sidereal_time`,
`radec_of`, `compute_pressure`), `Body`/`Planet` and the ten built-in bodies
(with `compute` and `parallactic_angle`), the `Body_getset` and
`Planet_getset` attribute tables, `julian_date`, `delta_t`, `separation`,
`unrefract`, `now`, `localtime`, `to_timezone`, `UTC`, `precess`, `eq_ecl`,
`ecl_eq`, the eight rise, set and transit searches, `disallow_circumpolar`,
`newton`, the equinox and solstice helpers under all three of upstream's
spellings, the eight moon-phase helpers, the `acos`/`cos`/`isnan`/`sin`
re-exports upstream's `from math import ...` line makes public, and the
`Equatorial` / `Ecliptic` / `LonLatCoordinate` classes with their
conversions.

`mojoephem.compute_series(body, dates, observer=None)` is the one addition:
the same `obj_planet` call repeated over a buffer in a single crossing of
the ABI.

### Accuracy

Measured against upstream `ephem` 4.2.1 (`pixi run test`), over ten bodies
and eight epochs from 1600 to 2200:

- astrometric and apparent geocentric right ascension and declination agree
  to better than 1 milliarcsecond for every body from 1600 to 2200;
- heliocentric longitude and latitude agree to better than 0.1 arcsecond;
- earth and sun distances agree to within upstream's own `float` storage
  (a few times 1e-6 AU, which is where the two ports start to differ);
- altitude, azimuth, hour angle, the rise/set/transit searches, the
  equinox and solstice dates and the moon phases all agree to the tolerance
  upstream's field storage allows - 0.1 arcsecond for alt/az, 0.1 s of
  clock time for the event searches.

The worst deviations measured over that matrix are 1.0e-6 arcsecond on
astrometric and geocentric right ascension and declination, 0.05 arcsecond
on heliocentric longitude and latitude, 0.05 arcsecond on azimuth, and
4e-3 s of clock time on the event searches.

The port is a *literal* transcription, so the agreement is bit-for-bit
wherever upstream computes in `double` (`precess` differs in 5 of 24
probed cases by at most 2.2 units in the last place and otherwise matches
exactly; `eq_ecl` and `ecl_eq` to 1e-10 arcsecond). The remaining differences
are upstream's storage, not the port's arithmetic: `s_alt`, `s_az`, `s_elong`,
`s_phase`, `s_edist` and `s_size` are C `float`s upstream, and this port
keeps them as doubles, so it is the more precise of the two. The one place
that quantisation *is* reproduced is the visual magnitude, because
`set_smag` rounds it to hundredths and users can see that; see the
divergence note below.

## What is not covered

Named explicitly, with the reason:

- **Planetary moons** (`Phobos` .. `Miranda`, upstream's `PlanetMoon` and
  the `plmoon.c`, `jupmoon.c`, `marsmoon.c`, `satmoon.c` theories). These
  are separate, much larger theories; this port covers the ten
  `libastro` bodies that VSOP87, Chapront-95 and the lunar theory describe.
- **Comets, asteroids and artificial satellites** (`EllipticalBody`,
  `HyperbolicBody`, `ParabolicBody`, `EarthSatellite`, `comet.c`, `twobody.c`,
  `sgp4.c`, `sdp4.c`, `earthsat.c`, and `Observer.next_pass`). The generic
  two-body machinery they need is not ported, so those classes do not exist
  here and `next_pass` raises `NotImplementedError`.
- **Fixed objects**: `ephem.star`, `ephem.readdb`, `ephem.readtle`,
  `ephem.constellation`, `ephem.city`, and the `FixedBody` /
  `EquatorialBody` / `DeepSkyBody` types. `readdb` is a Python parser over
  a text catalogue in `ephem/`; it is not compute-bound and not part of
  `libastro`.
- **Atmospheric refraction applied to alt/az**: no, it *is* ported
  (`refract.c`); what is missing is the Miller/Russell refraction model
  selected by `Observer.refraction` upstream.
- **`Body.rise_time`, `set_time`, `transit_time`, `rise_az`, `set_az`,
  `transit_alt`, `circumpolar`, `neverup`** - the legacy `riset_cir()` path
  in `riset.c` / `riset_cir.c`. Upstream's own `Observer.next_rising` and
  friends do not use it (they are pure Python, in `ephem/__init__.py`, and
  are ported here unchanged), so the modern API is complete; the old
  per-body fields are not.
- **Lunar libration and selenography** (`libration_lat`,
  `libration_long`, `colong`, `subsolar_lat`; `libration.c`, 2211 lines) and
  **central-meridian longitudes** (`Jupiter.cmlI/cmlII`,
  `Saturn.earth_tilt/sun_tilt` beyond the ring tilt that `plans` already
  uses). `satrings` is ported because `plans` needs it for Saturn's
  magnitude, but the `libration.c` and `jupmoon.c`/`satmoon.c` machinery is
  not.
- **Database serialisation and copying**: `Body.writedb` and
  `Body.copy`/`__copy__`. Both are thin wrappers over XEphem's `db_`
  catalogue format and a `memcpy` of the `Obj` struct; neither needs a
  kernel. This port keeps bodies as plain Python objects, so a copy is
  `copy.copy(body)` and there is no catalogue writer.
- **`parallelLDA`/`parallelLHD`** as standalone C entry points. The
  geometry they implement - `solve_sphere` with the observer's latitude as
  the third side - is ported, and `Body.parallactic_angle` uses it; only the
  two free functions are not exported, because nothing in the Python API
  calls them.
- **The GPU.** The kernels are CPU-only, and this is a deliberate choice,
  not a missing API: `max.gpu.host`'s `DeviceContext`,
  `enqueue_create_buffer`, `enqueue_copy`, `enqueue_function` and
  `synchronize` all compile on this toolchain. The obstacle is the shape of the
  work, and it is measured rather than assumed. A `@gpu` kernel can only call
  GPU-callable code, and `obj_planet` is forty-odd host `def` functions in a
  chain - VSOP87, Chapront-95, the lunar theory, nutation, aberration - so a
  GPU path means re-expressing the whole ephemeris, not porting one loop. What
  is left per date is a serial chain of dependent `cos`/`sin` evaluations,
  well under the roughly 2-flops-per-byte where a GPU pays: those kernels are
  bound by transcendental throughput, not by cores or bandwidth, and a GPU's
  special-function units would be the limiter. The parallelism that does pay
  is over a batch of dates, and `compute_series` takes it on all 36 cores.

### Documented divergences

Three, all mechanical, all visible in `src/ported.mojo`'s header:

1. **Out-parameters become return values.** Upstream threads results
   through `double *`, including pointers to its own locals. Mojo locals
   have no address, so kernels return values or tuples and the `@export`
   wrappers unpack them into the caller-owned buffers the C ABI wants. The
   arithmetic between a return and its caller is unchanged.
2. **`static` memo slots are gone.** Upstream caches `lastmj`, `lasteps`
   and friends. That is caching, not semantics, so each call recomputes -
   which is also what makes the batch path safe to run across cores.
3. **The numeric tables cross the ABI as addresses.** `mojo build --emit
   shared-lib` cannot place read-only comptime data in a shared object: the
   tables land in `.text` with PC-32 relocations that `ld` refuses for a
   PIC object. So `tools/gen_tables.py` packs them into
   `python/mojoephem/data/tables.bin` and `mojoephem._tables` maps that at
   import, handing the kernel a registry of base addresses. The numbers are
   transcribed verbatim from the C source, not refitted.

One deliberate semantic choice: upstream's `s_elong`, `s_phase`, `s_edist`
and `s_size` are 24-bit floats, so the values it hands back are quantised.
This port keeps them as doubles. The magnitude is the exception - `set_smag`
is applied in the kernel exactly as upstream does, so `mag` is comparable
to the last digit.

Two places where this port answers the documented question rather than
copying upstream's output. Both are upstream defects, both are covered by
tests, and both are recorded here rather than silently "fixed":

- **`Observer.radec_of`** works, where upstream's does not. Upstream's
  `Observer_radec_of` calls `ap_as(&now, epoch, &ra, &dec)` with five
  arguments to a four-argument function, and `ap_as` itself reads an
  undeclared `mjd`. The result is a declination no azimuth and altitude pair
  can produce: for az=100 deg, alt=30 deg at Greenwich on 2024/6/21 it
  answers -43.7 degrees where the geometry gives +17.3. This port computes
  the alt/az to ra/dec conversion its own docstring describes, from
  `unrefract` and `mpe_aa_hadec`, and `tests/test_api_surface.py` pins it
  against those formulas and round-trips it back through `mpe_hadec_aa`.
- **`Body.parallactic_angle`** uses the hour angle the body's own compute
  stored. Upstream recomputes it with `radec2ha`, which - when the global
  `epoch` is not `EOD` - hands `as_ap` the addresses of `radec2ha`'s own
  *local* parameters, so the apparent place is converted to astrometric in
  place and the answer stops matching the `a_ra`/`a_dec` the same body
  publishes. For the Moon at Greenwich upstream answers -17.506 degrees
  where its own published fields give -17.690. This port is self-consistent
  with the fields it reports, and the test asserts that consistency rather
  than upstream's number.

## Benchmarks

Measured on this machine with `pixi run bench` (best of three; the pixi
task holds a machine-wide flock, so run benchmarks that way and never as
`python bench/bench.py`).

```
machine: Linux-6.8.0-142-generic-x86_64-with-glibc2.39
python:  3.13.14  numpy: 2.5.1
cpu:     Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (72 threads)
upstream ephem 4.2.1; mojo-pyephem 0.1.0
| case | mojo-pyephem | upstream ephem | speedup |
| --- | ---: | ---: | ---: |
| Sun.compute, 20000 dates (batched) | 30.3 ms | 584.9 ms | 19.31x |
| Moon.compute, 20000 dates (batched) | 56.1 ms | 1356.8 ms | 24.17x |
| Saturn.compute, 20000 dates (batched) | 43.5 ms | 1206.2 ms | 27.72x |
| Mars.compute, 20000 dates (batched) | 86.1 ms | 1450.4 ms | 16.85x |
| Sun.compute, 20000 dates, topocentric (batched) | 48.0 ms | 586.1 ms | 12.22x |
| Sun.compute, 20000 calls, one date at a time | 480.1 ms | 587.9 ms | 1.22x |
| Sun.compute, 2000 calls, one date at a time | 47.9 ms | 58.4 ms | 1.22x |
| Observer.next_rising(Sun), 2000 searches | 266.9 ms | 204.8 ms | 0.77x (mojo slower) |
| delta_t, 200000 dates | 195.7 ms | 96.6 ms | 0.49x (mojo slower) |
| precess, 200000 pairs | 612.1 ms | 126.0 ms | 0.21x (mojo slower) |
| separation, 200000 pairs | 240.6 ms | 118.4 ms | 0.49x (mojo slower) |
```

What the optimisation pass changed, and what it did not:

- **Redundant `sunpos` calls removed (kept).** `obj_planet` computed the
  solar longitude and distance and then, on the Sun (`p == 8`) and Moon
  (`p == 9`) paths, threw them away because `sun_cir`/`moon_cir` compute their
  own. It now dispatches those two bodies first. For a planet, `plans` and
  `cir_pos` each recomputed the same `sunpos(mjed)` that `obj_planet` already
  had; the values are now computed once and threaded in. One `sunpos` is
  8.6 us, so this removes a full trigonometry-heavy call from every compute:
  a raw `mpe_obj_planet` call is now 18.9 us for the Sun and 31.5 us for
  Saturn, against 8.6 us of sunpos inside each. One call at a time went from
  0.83x/0.80x slower to **1.16x/1.24x faster**, and `next_rising` from 0.54x
  to 0.75x.
- **Worker cap raised 16 -> 36 (kept).** The old cap was justified by a
  `pids.max` abort risk that does not apply here (measured
  `pids.max = 1030864`, usage 72). `num_physical_cores()` is 36 here, so the
  cap was not binding before and the batch now uses the whole machine: a
  20000-date Sun batch went from 54 ms to 30 ms.
- **SIMD on the trig term sums (not kept, measured).** VSOP87 and nutation
  are dominated by `cos`/`sin`. A probe on this toolchain
  (`SIMD[DType.float64, 4]`, `simd_width_of`, which reports 4 here) shows
  that a straight-line SIMD `cos` really is much faster than scalar - 1.3
  ns per element against 22.1 - so the obstacle is not the transcendental
  itself. It is the shape of the loop. The term sum is

      for each (a, b, c) in the block:
          if a >= p: term += a * cos(b + c * t)

  which gathers a strided triple, skips terms on a data-dependent test, and
  runs to a data-dependent bound. Driving that exact shape with SIMD on this
  toolchain gives 28.8 ns per term against 30.9 scalar - a 1.07x speedup, so
  the gather and the mask eat the whole vector width. Removing the
  `cos`/`sin` calls entirely, which is what the `sunpos` change below does,
  is worth far more than widening the ones that remain.
- **GPU (not built, deliberate).** `max.gpu.host`'s `DeviceContext`,
  `enqueue_create_buffer`, `enqueue_copy`, `enqueue_function` and
  `synchronize` all compile on this toolchain. The kernels here are far below
  the roughly 2-flops-per-byte threshold that would justify a GPU: an
  `obj_planet` call is a long serial chain of dependent `cos`/`sin`, limited
  by transcendental throughput rather than cores or bandwidth, and a GPU's
  special-function units would be the limiter. The parallelism that does pay
  is over a batch of dates, and `compute_series` takes it.

Read this honestly:

- **Batched, the port is 12x to 28x faster** than upstream, from two places:
  the epoch loop in `mpe_obj_planet_series` running on all 36 physical cores,
  and the redundant-`sunpos` removal above, which cut the scalar work per
  date by 16-25%.
- **One call at a time, the port is now faster** (1.22x on both sizes), where it
  was 0.83x before this pass. The gain is the same redundancy removal; the
  arithmetic per date is unchanged, there are just fewer duplicate
  trigonometry calls.
- **The small scalar routines remain slower and cannot be fixed.** `delta_t`,
  `precess` and `separation` are called one element at a time, and on this
  machine a single empty `ctypes` crossing (measured with `mpe_isleapyear` on
  a `ctypes.c_int64`) costs about 620 ns - already more than upstream's
  *entire* `delta_t` (310 ns), `separation` (480 ns) or `precess` (620 ns),
  because upstream computes them in a C extension with no FFI crossing and no
  `Angle` object to build. Our own wrappers add only a little on top of the
  raw kernel: `delta_t` is 840 ns raw and 960 ns through `mojoephem`, and
  `precess` is 1.2 us raw and 3.2 us wrapped, the difference being the
  `Angle` objects upstream builds in C. The marshalling floor, not the
  kernel, is what these numbers measure, and it cannot be lowered without
  changing the API. `compute_series` is the shape in which these bodies are
  fast; that is the point of the batch entry point.

This is a shared machine, and the pixi flock only excludes other benchmark
runs, not other tenants, so the milliseconds here are a snapshot of one
moment's load rather than a property of this code. Across four runs of
this same table the batched speedups ranged from 12x to 28x and the
one-at-a-time figures sat between 1.16x and 1.24x, while the *upstream*
column - which is not what is being optimised - moved by 5%. Read the
speedup column as an order of magnitude, not as four significant figures.

The benchmark harness also has a trap worth repeating: upstream's
`Body.compute()` is a statement that returns `None`. It records the date
and marks the body dirty; the ephemeris is evaluated on the first attribute
read. A benchmark that calls `compute()` in a loop without reading a field
is timing Python's `__init__`, and reports a large "speedup" for upstream
that is entirely an artefact. `bench/bench.py` reads the fields.

## How it works

**FFI.** `mojo build --emit shared-lib src/ported.mojo` produces
`dist/libmojo-pyephem.so`, exporting 48 `mpe_*` symbols. Every buffer
crosses as an `Int` address, because `@export` rejects parametric functions
and a pointer with an inferred origin is parametric. Each wrapper is a
thin unpack: it rebuilds a `Pointer` from the address, calls the kernel
that carries the real work, and scatters the result into the caller's
buffer. Nothing in the library allocates, so nothing can leak, and the
caller owns all memory including scratch. `python/mojoephem/_lib.py` is
the ctypes side: it declares each signature once and, on first use,
compiles the library if `dist/` is missing or stale.

`tests/test_exports.py` calls all 48 by name through ctypes and asserts on
what each one returns, so a kernel cannot be wired to the wrong argument or
the wrong out-slot and still pass. Where upstream exposes the same routine
to Python the assertion is against upstream; where it does not, it is
against the formula transcribed from the C or a physical invariant - a
Kepler equation, a refraction table, a round trip through `solve_sphere`.
`test_every_exported_symbol_is_declared_and_has_a_test` fails if a symbol
is added to the C ABI without both a declared signature and a test, so the
coverage cannot quietly rot.

**Memory layout.** Two regions, both owned by Python and both stable for
the life of the process:

- the *table registry*, an `int64` vector of 51 entries built by
  `mojoephem._tables` from `data/tables.bin`; slot *i* holds the base
  address of table *i*, and the kernel's `tf(reg, slot)` / `ti(reg, slot)`
  turn a slot into a typed pointer. The slot order is the manifest order of
  `tools/gen_tables.py` and is asserted against `data/tables.json` at
  import, so a stale data file fails loudly instead of silently computing
  garbage. The tables themselves are contiguous: 8 VSOP87D bodies as
  (amplitude, phase, frequency) triples, 8 `vn_` block-boundary tables, the
  106-term nutation tables, Moshier's two `plantbl` argument and
  coefficient tables, the five Chapront-95 record tables, and the two
  deltaT tables.
- the *output vector*, a 17-element `float64` per body, in the order
  `obj_planet` fills it: `hlon, hlat, sun_distance, earth_distance,
  elong (degrees), phase, mag, size, g_ra, g_dec, a_ra, a_dec, ha, alt, az,
  rho, mj`. `compute_series` writes `n` of them contiguously, so a batch is
  one `n x 17` array.

**The batch path.** `mpe_obj_planet_series` is the same `obj_planet` call
repeated over a buffer of dates, and it is the only place this port uses
more than one core. Two properties make that safe: upstream's `static`
memo slots (`lastmj`, `lasteps`) are gone, so no call depends on the one
before it, and every table is read-only. Each worker takes a contiguous
run of dates - so it also writes a contiguous run of the output - via
`max.algorithm.parallelize` with an explicit `num_physical_cores()`
worker count. Below 256 dates the worker launch costs more than the work
it hands out, and the same loop runs serially; `test_series_sizes_cover_
every_date` walks the sizes either side of that line.

**One solar position per compute.** `obj_planet` computes the solar
ecliptic longitude, distance and latitude with a single `sunpos(mjed)`
and threads them into `plans` (which needs them for the cartesian sun and
Saturn's ring tilt) and `cir_sky`/`cir_pos` (which need them for aberration,
deflection and phase). Upstream's structure computes the same `sunpos`
several times per date and, on the Sun and Moon paths, computes it and
discards it; those duplicates are removed here. Nutation is still computed
twice on some paths, because it is evaluated at `mjed` for the apparent
place and at `mjd` for local sidereal time, and those are genuinely
different arguments.

**What lives where.** `src/ported.mojo` is one compilation unit: every
ported routine followed by the C ABI, in upstream's file order.
`python/mojoephem/__init__.py` is the API - the same shape as upstream's
own `ephem/__init__.py`, which is pure Python over a C extension.
`tools/gen_tables.py` transcribes the C tables; `build/data.sh`
regenerates them from an unpacked ephem sdist when moving to a new upstream
release.

## Licence

MIT, per `LICENSE`. The coefficient tables in
`python/mojoephem/data/tables.bin` are transcribed verbatim from
libastro's VSOP87D, Chapront-95 and Moshier tables; libastro is LGPL, so
those tables are a derivative of LGPL sources and the authors of this port
should settle that before publishing. Everything else here - the Mojo
kernels, the Python layer, the benchmarks and the tests - is original work
under the MIT terms.
