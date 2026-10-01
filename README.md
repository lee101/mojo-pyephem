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
pixi run test       # 600 parity tests against the real upstream ephem
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
`_libastro` C extension: `Date`, `Angle`, `Observer`, `Body`/`Planet` and
the ten built-in bodies, the `Body_getset` and `Planet_getset` attribute
tables, `julian_date`, `delta_t`, `separation`, `unrefract`, `now`,
`localtime`, `to_timezone`, `UTC`, `precess`, `eq_ecl`, `ecl_eq`,
`Observer.sidereal_time`, `Observer.compute_pressure`, the eight rise, set
and transit searches, `disallow_circumpolar`, `newton`, the equinox and
solstice helpers, the eight moon-phase helpers, and the `Equatorial` /
`Ecliptic` / `LonLatCoordinate` classes with their conversions.

`mojoephem.compute_series(body, dates, observer=None)` is the one addition:
the same `obj_planet` call repeated over a buffer in a single crossing of
the ABI.

### Accuracy

Measured against upstream `ephem` 4.2.1 (`pixi run test`):

- astrometric and apparent geocentric right ascension and declination agree
  to better than 1 milliarcsecond for every body from 1600 to 2200;
- heliocentric longitude and latitude agree to better than 0.1 arcsecond;
- earth and sun distances agree to within upstream's own `float` storage
  (a few times 1e-6 AU, which is where the two ports start to differ);
- altitude, azimuth, hour angle, the rise/set/transit searches, the
  equinox and solstice dates and the moon phases all agree to the tolerance
  upstream's field storage allows - 0.1 arcsecond for alt/az, 0.1 s of
  clock time for the event searches.

The worst deviations measured over the test matrix (ten bodies x eight
epochs from 1600 to 2200) are 1.0e-6 arcsecond on astrometric and
geocentric right ascension and declination, 0.05 arcsecond on
heliocentric longitude, 0.05 arcsecond on azimuth, and 4e-6 s of clock
time on the event searches.

The port is a *literal* transcription, so the agreement is bit-for-bit
wherever upstream computes in `double` (`precess` differs in 222 of 240
probed cases by at most one unit in the last place and otherwise matches
exactly; `eq_ecl` and `ecl_eq` to 1e-10 arcsecond). The remaining differences are
upstream's storage, not the port's arithmetic: `s_alt`, `s_az`, `s_elong`,
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
- **The GPU.** The kernels are CPU-only, and this is a deliberate choice,
  not a missing API: `max.gpu.host`'s `DeviceContext`,
  `enqueue_create_buffer`, `enqueue_copy`, `enqueue_function` and
  `synchronize` all compile on this toolchain. The obstacle is the shape of the
  work. A `@gpu` kernel can only call GPU-callable code, and `obj_planet`
  is forty-odd host `def` functions in a chain - VSOP87, Chapront-95, the
  lunar theory, nutation, aberration - so a GPU path means re-expressing
  the whole ephemeris, not porting one loop. What is left per date is a
  serial chain of dependent `cos` evaluations, and a GPU's
  special-function units are the limiter there, not its cores. The
  parallelism that does pay is over a batch of dates, and `compute_series`
  now takes it on all 36 cores.

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
| Sun.compute, 20000 dates (batched) | 67.7 ms | 757.6 ms | 11.18x |
| Moon.compute, 20000 dates (batched) | 97.7 ms | 1831.7 ms | 18.74x |
| Saturn.compute, 20000 dates (batched) | 91.3 ms | 1625.9 ms | 17.82x |
| Mars.compute, 20000 dates (batched) | 124.7 ms | 1991.4 ms | 15.97x |
| Sun.compute, 20000 dates, topocentric (batched) | 57.3 ms | 790.3 ms | 13.79x |
| Sun.compute, 20000 calls, one date at a time | 920.7 ms | 765.9 ms | 0.83x (mojo slower) |
| Sun.compute, 2000 calls, one date at a time | 88.1 ms | 70.7 ms | 0.80x (mojo slower) |
| Observer.next_rising(Sun), 2000 searches | 487.8 ms | 261.8 ms | 0.54x (mojo slower) |
| delta_t, 200000 dates | 290.4 ms | 141.3 ms | 0.49x (mojo slower) |
| precess, 200000 pairs | 924.0 ms | 179.0 ms | 0.19x (mojo slower) |
| separation, 200000 pairs | 355.2 ms | 164.3 ms | 0.46x (mojo slower) |
```

Read this honestly:

- **Batched, the port is 11x to 19x faster** than upstream. The whole gain
  is the epoch loop in `mpe_obj_planet_series` running on all 36 physical
  cores. One `obj_planet` call for the Sun is 46 us of scalar work here, so
  the arithmetic is unchanged; there were just 36 copies of it available.
- **One call at a time, the port is slower**, by 1.2x on a full ephemeris
  and 2x to 5.3x on the small routines. What is left there is Python, not
  Mojo: the entry point, the table-registry address and the scratch buffers
  are resolved once at first use rather than per call, and `_parse_mjd`
  dispatches on exact type before its `isinstance` chain.
- **The rest of that gap is architectural.** Upstream's small routines are
  a C extension called from Python, with no `ctypes` marshalling and no
  `Angle` object to build; ours is a shared library reached through both.
  A ctypes call into this library costs 1.03 us even when the callee does
  nothing measurable (`mpe_isleapyear` on a `ctypes.c_int64`), against
  `delta_t`'s 1.45 us end to end. Marshalling, not arithmetic, is the
  floor on that path, and there is nothing left to remove without changing
  the API.

The VSOP87 term sum was rebuilt both with a `SIMD[DType.float64, 4]`
gather over the interleaved (a, b, c) triples and with a
structure-of-arrays layout feeding contiguous `p.unsafe_load[width=W](i)`
loads, and four independent partial sums were tried to break the serial
`term +=` dependency. None of it paid: the loop is bound by `cos`
throughput, and Mojo's `cos` on `SIMD` is a scalarised loop on this
toolchain. The scalar loop is what ships.

This is a shared machine, and the pixi flock only excludes other benchmark
runs, not other tenants. The ratio is the stable quantity, not the
absolute numbers: a busy host scaled *both* columns by about 1.35x
against an earlier run of this same table, so treat the speedup column as
the result and the milliseconds as a snapshot of one moment's load.

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
