"""Pyephem's libastro core, ported function by function.

Read this top to bottom against the upstream C it came from. The sections
follow upstream's own file order (mjd.c, deltat.c, auxil.c, sphcart.c,
obliq.c, reduce.c, anomaly.c, precess.c, eq_ecl.c, nutation.c, aberration.c,
vsop87.c, sun.c, chap95.c, plans.c, rings.c, moon.c, misc.c, aa_hadec.c,
parallax.c, refract.c, utc_gst.c, circum.c) and each function keeps
upstream's name, argument order, branch structure and arithmetic order.

Three mechanical divergences, all forced by the ABI rather than chosen here:

  * Upstream threads results through `double *` out-parameters, including
    pointers to its own locals. Mojo locals have no address, so kernels
    return values or tuples and the `@export` wrappers below unpack them
    into the caller-owned buffers the C ABI wants. The arithmetic and the
    branches between a return and its caller are unchanged.
  * Upstream memoises through `static` slots (`lastmj`, `lasteps`, ...).
    That is caching, never semantics, so it is dropped and each call
    recomputes.
  * Upstream's numeric tables are C file-scope arrays. `mojo build --emit
    shared-lib` cannot place read-only comptime data in a shared object (the
    tables land in `.text` with PC-32 relocations `ld` refuses), so every
    table arrives as an `Int` address in a registry that
    `mojoephem._tables` builds from `data/tables.bin`. The slot indices are
    the manifest order of `tools/gen_tables.py`; `tf` fetches a float64
    table and `ti` an int32 one.
"""

from std.math import (
    abs,
    acos,
    asin,
    atan2,
    cos,
    cosh,
    floor,
    log10,
    pow,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)
from max.algorithm import parallelize
from std.sys import num_physical_cores
from std.utils import StaticTuple

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime Reg = Pointer[Int, AnyOrigin[mut=True]]

# Registry slots, in tools/gen_tables.py manifest order.
#    0 vsop_vx_mercury    1 vsop_vn_mercury
#    2 vsop_vx_venus      3 vsop_vn_venus
#    4 vsop_vx_earth      5 vsop_vn_earth
#    6 vsop_vx_mars       7 vsop_vn_mars
#    8 vsop_vx_jupiter    9 vsop_vn_jupiter
#   10 vsop_vx_saturn     11 vsop_vn_saturn
#   12 vsop_vx_uranus     13 vsop_vn_uranus
#   14 vsop_vx_neptune   15 vsop_vn_neptune
#   16 nut_delaunay      17 nut_multarg    18 nut_ampsecul    19 nut_ampconst
#   20 moon_moonlr_harmonic  21 moon_moonlr_power
#   22 moon_moonlr_args  23 moon_moonlr_tabl  24 moon_moonlr_tabb
#   25 moon_moonlr_tabr  26 moon_moonlr_scalars
#   27 moon_moonlat_harmonic 28 moon_moonlat_power
#   29 moon_moonlat_args 30 moon_moonlat_tabl 31 moon_moonlat_tabb
#   32 moon_moonlat_tabr 33 moon_moonlat_scalars
#   34 chap95_n_jupiter   35 chap95_amp_jupiter   36 chap95_nu_jupiter
#   37 chap95_n_saturn    38 chap95_amp_saturn    39 chap95_nu_saturn
#   40 chap95_n_uranus    41 chap95_amp_uranus    42 chap95_nu_uranus
#   43 chap95_n_neptune  44 chap95_amp_neptune  45 chap95_nu_neptune
#   46 chap95_n_pluto    47 chap95_amp_pluto    48 chap95_nu_pluto
#   49 deltat_ms         50 deltat_dt

# astro.h: MJD0 and J2000, the two epochs every epoch-relative formula here
# measures from.
comptime MJD0: Float64 = 2415020.0
comptime J2000: Float64 = 2451545.0 - 2415020.0


def tf(reg: Reg, slot: Int) -> Ptr:
    return Ptr(unsafe_from_address=reg.unsafe_offset(slot)[])


def ti(reg: Reg, slot: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=reg.unsafe_offset(slot)[])


# ============================================================= astro.h
# libastro keeps its unit conversions as C macros. They stay literals here
# for the same reason the tables came over the ABI: a shared object cannot
# own comptime storage in this toolchain.
def degrad(x: Float64) -> Float64:
    return x * 3.141592653589793 / 180.0


def raddeg(x: Float64) -> Float64:
    return x * 180.0 / 3.141592653589793


def hrdeg(x: Float64) -> Float64:
    return x * 15.0


def deghr(x: Float64) -> Float64:
    return x / 15.0


def hrrad(x: Float64) -> Float64:
    return degrad(hrdeg(x))


def radhr(x: Float64) -> Float64:
    return deghr(raddeg(x))


# ============================================================== mjd.c
# Insure 0 <= v < r.
def range(v: Float64, r: Float64) -> Float64:
    return v - r * floor(v / r)


# Insure 0 <= ra < 2PI and -PI/2 <= dec <= PI/2. If dec needs
# complementing, reflect ra too.
def radecrange(ra: Float64, dec: Float64) -> Tuple[Float64, Float64]:
    var r = ra
    var d = dec
    if d < -3.141592653589793 / 2.0:
        d = -3.141592653589793 - d
        r += 3.141592653589793
    elif d > 3.141592653589793 / 2.0:
        d = 3.141592653589793 - d
        r += 3.141592653589793
    return range(r, 2.0 * 3.141592653589793), d


# Given an mjd, truncate it to the beginning of the whole day.
def mjd_day(mj: Float64) -> Float64:
    return floor(mj - 0.5) + 0.5


# Given an mjd, return the number of hours past midnight of the whole day.
def mjd_hr(mj: Float64) -> Float64:
    return (mj - mjd_day(mj)) * 24.0


# Given a date in months mn, days dy, years yr, return the modified Julian
# date (number of days elapsed since 1900 jan 0.5).
def cal_mjd(mn: Int, dy: Float64, yr: Int) -> Float64:
    var m = mn
    var y = yr
    if yr < 0:
        y = yr + 1
    if mn < 3:
        m += 12
        y -= 1

    # cal_mjd() uses Gregorian dates on or after Oct 15, 1582.
    var b: Int = 0
    if not (yr < 1582 or (yr == 1582 and (mn < 10 or (mn == 10 and dy < 15)))):
        var a = Int(y / 100)
        b = 2 - a + a / 4

    var c: Int
    if y < 0:
        c = Int(floor(365.25 * Float64(y)) - 0.75) - 694025
    else:
        c = Int(365.25 * Float64(y)) - 694025

    var d = Int(30.6001 * Float64(m + 1))
    return Float64(b + c + d) + dy - 0.5


# Given the modified Julian date mj, return the calendar date as
# (months, days, years).
def mjd_cal(mj: Float64) -> Tuple[Int, Float64, Int]:
    # We get called with 0 quite a bit from unused epoch fields.
    # 0 is noon the last day of 1899.
    if mj == 0.0:
        return 12, 31.5, 1899

    var d = mj + 0.5
    var i = floor(d)
    var f = d - i
    if f == 1:
        f = 0
        i += 1

    if i > -115860.0:
        var a = floor(i / 36524.25 + 0.99835726) + 14
        i += 1 + a - floor(a / 4.0)

    var b = floor(i / 365.25 + 0.802601)
    var ce = i - floor(365.25 * b + 0.750001) + 416
    var g = floor(ce / 30.6001)
    var mn = Int(g - 1)
    var dy = ce - floor(30.6001 * g) + f
    var yr = Int(b + 1899)

    if g > 13.5:
        mn = Int(g - 13)
    if Float64(mn) < 2.5:
        yr = Int(b + 1900)
    if yr < 1:
        yr -= 1
    return mn, dy, yr


# Given a decimal year, return mjd.
def year_mjd(y: Float64) -> Float64:
    var yf = Int(floor(y))
    if yf == -1:
        yf = -2
    var e0 = cal_mjd(1, 1.0, yf)
    var e1 = cal_mjd(1, 1.0, yf + 1)
    return e0 + (y - Float64(yf)) * (e1 - e0)


# Given an mjd, return the decimal year containing it.
def mjd_year(mj: Float64) -> Float64:
    var _, _, y = mjd_cal(mj)
    if y == -1:
        y = -2
    var e0 = cal_mjd(1, 1.0, y)
    var e1 = cal_mjd(1, 1.0, y + 1)
    return Float64(y) + (mj - e0) / (e1 - e0)


# Given a year, return whether it is a leap year.
def isleapyear(y: Int) -> Int:
    if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0:
        return 1
    return 0


# Given an mjd, return 0..6 for the day of the week (0 = sunday), or -1
# when the British 1752 correction makes it unworkable.
def mjd_dow(mj: Float64) -> Int:
    if mj < -53798.5:
        return -1
    var dow = (Int(floor(mj - 0.5)) + 1) % 7  # 1/1/1900 is a Monday
    if dow < 0:
        dow += 7
    return dow


# ============================================================ deltat.c
# DeltaT = Ephemeris Time - Universal Time, in seconds. Input is XEphem's
# MJD. The tables run 1620 (TABSTART) to 2018 (TABEND).
def deltat(mj: Float64, reg: Reg) -> Float64:
    var Y = mjd_year(mj)
    var dt = ti(reg, 50)
    var ms = ti(reg, 49)

    if Y > 2018.0:
        # Extrapolate future values beyond the lookup table.
        if Y > 2118.0:
            # Morrison & Stephenson (2004) long-term curve fit.
            var B = 0.01 * (Y - 1820.0)
            return 32.0 * B * B - 20.0
        else:
            # Cubic interpolation between last tabulated value and the
            # long-term curve evaluated at 100 years later.
            var a = 0.01 * Float64(dt.unsafe_offset(398)[])
            # Approximate slope in past 10 years.
            var b = 0.001 * Float64(dt.unsafe_offset(398)[] - dt.unsafe_offset(388)[])
            var B = 0.01 * (2018.0 + 100.0 - 1820.0)
            var m0 = 32.0 * B * B - 20.0
            var m1 = 0.64 * B
            var d = 2.0e-6 * (50.0 * (m1 + b) - m0 + a)
            var c = 1.0e-4 * (m0 - a - 100.0 * b - 1.0e6 * d)
            var p = Y - 2018.0
            return a + p * (b + p * (c + p * d))

    # Use Morrison and Stephenson (2004) prior to the year 1700.
    if Y < 1700.0:
        if Y <= -1000.0:
            # Morrison & Stephenson long-term fit.
            var B = 0.01 * (Y - 1820.0)
            return 32.0 * B * B - 20.0
        else:
            # Morrison & Stephenson recommend linear interpolation
            # between tabulations.
            var iy = (Int(Y) + 1000) / 100
            var B = Float64(-1000 + 100 * iy)
            var p = Float64(ms.unsafe_offset(iy)[])
            return p + 0.01 * (Y - B) * (Float64(ms.unsafe_offset(iy + 1)[]) - p)

    # Avoid a segmentation fault if year is NaN.
    if Y != Y:
        return 0.0

    # Besselian interpolation between tabulated values in the telescopic
    # era. See AA page K11.
    var p = floor(Y)
    var iy = Int(p - 1620.0)
    # Zeroth order estimate is the value at the start of the year.
    var ans = Float64(dt.unsafe_offset(iy)[])
    var k = iy + 1
    if k >= 399:
        return ans * 0.01  # no data, can't go on

    # The fraction of the tabulation interval.
    p = Y - p

    # First order interpolated value.
    ans += p * (Float64(dt.unsafe_offset(k)[]) - Float64(dt.unsafe_offset(iy)[]))
    if (iy - 1 < 0) or (iy + 2 >= 399):
        return ans * 0.01  # can't do second differences

    # Make a table of first differences.
    var d = StaticTuple[Float64, 5](0.0, 0.0, 0.0, 0.0, 0.0)
    k = iy - 2
    var i = 0
    while i < 5:
        if (k < 0) or (k + 1 >= 399):
            d[i] = 0
        else:
            d[i] = Float64(dt.unsafe_offset(k + 1)[] - dt.unsafe_offset(k)[])
        k += 1
        i += 1

    # Compute second differences.
    i = 0
    while i < 4:
        d[i] = d[i + 1] - d[i]
        i += 1
    var B = 0.25 * p * (p - 1.0)
    ans += B * (d[1] + d[2])
    if iy + 2 >= 399:
        return ans * 0.01

    # Compute third differences.
    i = 0
    while i < 3:
        d[i] = d[i + 1] - d[i]
        i += 1
    B = 2.0 * B / 3.0
    ans += (p - 0.5) * B * d[1]
    if (iy - 2 < 0) or (iy + 3 > 399):
        return ans * 0.01

    # Compute fourth differences.
    i = 0
    while i < 2:
        d[i] = d[i + 1] - d[i]
        i += 1
    B = 0.125 * B * (p + 1.0) * (p - 2.0)
    ans += B * (d[0] + d[1])
    return ans * 0.01


# ============================================================= auxil.c
# Return mjd modified for terrestrial dynamical time.
def mm_mjed(mjd: Float64, reg: Reg) -> Float64:
    return mjd + deltat(mjd, reg) / 86400.0


# =========================================================== sphcart.c
# Transformation from spherical to cartesian coordinates.
def sphcart(l: Float64, b: Float64, r: Float64) -> Tuple[Float64, Float64, Float64]:
    var rcb = r * cos(b)
    return rcb * cos(l), rcb * sin(l), r * sin(b)


# Transformation from cartesian to spherical coordinates.
def cartsph(x: Float64, y: Float64, z: Float64) -> Tuple[Float64, Float64, Float64]:
    var rho = x * x + y * y
    if rho > 0.0:  # standard case: off axis
        return (
            range(atan2(y, x), 2.0 * 3.141592653589793),
            atan2(z, sqrt(rho)),
            sqrt(rho + z * z),
        )
    else:  # degenerate case; avoid math error
        var half_pi = 3.141592653589793 / 2.0
        if z == 0.0:
            return 0.0, 0.0, abs(z)
        return 0.0, (half_pi if z > 0.0 else -half_pi), abs(z)


# ============================================================ obliq.c
# Given the modified Julian date mj, find the mean obliquity of the
# ecliptic in radians. IAU expression (see e.g. Astron. Almanac 1984).
def obliquity(mj: Float64) -> Float64:
    var t = (mj - 36525.0) / 36525.0  # centuries from J2000
    return degrad(
        23.4392911
        + t
        * (
            -46.8150
            + t * (
                -0.00059 + t * (0.001813)
            )
        )
        / 3600.0
    )


# ============================================================ reduce.c
# Convert those orbital elements that change from epoch mj0 to epoch mj.
# Returns (inc, ap, om).
def reduce_elements(
    mj0: Float64, mj: Float64, inc0: Float64, ap0: Float64, om0: Float64
) -> Tuple[Float64, Float64, Float64]:
    if abs(mj - mj0) < 1e-5:
        # sin(eta) blows for inc < 10 degrees -- anyway, no need
        return inc0, ap0, om0

    var t0 = mj0 / 365250.0
    var t1 = mj / 365250.0

    var tt = t1 - t0
    var tt2 = tt * tt
    var t02 = t0 * t0
    var tt3 = tt * tt2
    var eta = (471.07 - 6.75 * t0 + 0.57 * t02) * tt + (0.57 * t0 - 3.37) * tt2 + 0.05 * tt3
    var th0 = 32869.0 * t0 + 56 * t02 - (8694 + 55 * t0) * tt + 3 * tt2
    eta = degrad(eta / 3600.0)
    th0 = degrad(th0 / 3600.0 + 173.950833)
    var th = (50256.41 + 222.29 * t0 + 0.26 * t02) * tt + (111.15 + 0.26 * t0) * tt2 + 0.1 * tt3
    th = th0 + degrad(th / 3600.0)
    var cinc = cos(inc0)
    var sinc = sin(inc0)
    var ot = om0 - th0
    var sot = sin(ot)
    var cot = cos(ot)
    var seta = sin(eta)
    var ceta = cos(eta)
    var a = sinc * sot
    var b = ceta * sinc * cot - seta * cinc
    var ot1 = atan2(a, b)
    if b < 0:
        ot1 += 3.141592653589793
    b = sinc * ceta - cinc * seta * cot
    a = -1 * seta * sot
    var dap = atan2(a, b)
    if b < 0:
        dap += 3.141592653589793

    var ap = range(ap0 + dap, 2.0 * 3.141592653589793)
    var om = range(ot1 + th, 2.0 * 3.141592653589793)

    var inc: Float64
    if inc0 < 0.175:
        inc = asin(a / sin(dap))
    else:
        inc = 1.570796327 - asin(cinc * ceta + sinc * seta * cot)
    return inc, ap, om


# =========================================================== anomaly.c
# Given the mean anomaly ma and the eccentricity s of elliptical motion,
# find the true anomaly nu and the eccentric anomaly ea.
def anomaly(ma: Float64, s: Float64) -> Tuple[Float64, Float64]:
    var m: Float64
    var fea: Float64
    var corr: Float64
    var two_pi = 2.0 * 3.141592653589793
    var pi = 3.141592653589793
    var stoperr = 1e-8
    var nu: Float64

    if s < 1.0:
        # elliptical
        m = ma - two_pi * Float64(Int(ma / two_pi))
        if m > pi:
            m -= two_pi
        if m < -pi:
            m += two_pi
        fea = m

        while True:
            var dla = fea - s * sin(fea) - m
            if abs(dla) < stoperr:
                break
            # avoid runaway corrections for e>.97 and M near 0
            corr = 1 - s * cos(fea)
            if corr < 0.1:
                corr = 0.1
            dla /= corr
            fea -= dla
        nu = 2.0 * atan2(sqrt((1 + s) / (1 - s)) * tan(fea / 2.0), 1.0)
    else:
        # hyperbolic
        m = abs(ma)
        fea = m / (s - 1.0)
        var fea1 = pow(6.0 * m / (s * s), 1.0 / 3.0)
        # whichever is smaller is the better initial guess
        if fea1 < fea:
            fea = fea1
        corr = 1
        while abs(corr) > stoperr:
            corr = (m - s * sinh(fea) + fea) / (s * cosh(fea) - 1)
            fea += corr
        if ma < 0.0:
            fea = -fea
        nu = 2.0 * atan2(sqrt((s + 1) / (s - 1)) * tanh(fea / 2.0), 1.0)
    return nu, fea


# =========================================================== precess.c
# Correct ra and dec, both in radians, for precession from epoch 1 to
# epoch 2. Rigorous precession, from Astronomical Ephemeris 1989, p. B18.
def precess(
    mjd1: Float64, mjd2: Float64, ra: Float64, dec: Float64
) -> Tuple[Float64, Float64]:
    var from_equinox = mjd_year(mjd1)
    var to_equinox = mjd_year(mjd2)

    # convert coords in rads to degs
    var alpha_in = raddeg(ra)
    var delta_in = raddeg(dec)

    var alpha2000: Float64
    var delta2000: Float64

    # From from_equinox to 2000.0. Precession progresses about 1 arc second
    # in .047 years, so inside .02 years the two branches agree to the last
    # digit and the cheap one is taken.
    if abs(from_equinox - 2000.0) > 0.02:
        var T = (from_equinox - 2000.0) / 100.0
        var zeta_A = 0.6406161 * T + 0.0000839 * T * T + 0.0000050 * T * T * T
        var z_A = 0.6406161 * T + 0.0003041 * T * T + 0.0000051 * T * T * T
        var theta_A = 0.5567530 * T - 0.0001185 * T * T - 0.0000116 * T * T * T

        var A = sin(degrad(alpha_in - z_A)) * cos(degrad(delta_in))
        var B = (
            cos(degrad(alpha_in - z_A)) * cos(degrad(theta_A)) * cos(degrad(delta_in))
            + sin(degrad(theta_A)) * sin(degrad(delta_in))
        )
        var C = (
            -cos(degrad(alpha_in - z_A)) * sin(degrad(theta_A)) * cos(degrad(delta_in))
            + cos(degrad(theta_A)) * sin(degrad(delta_in))
        )

        alpha2000 = range(raddeg(atan2(A, B)) - zeta_A, 360.0)
        delta2000 = raddeg(asin(C))
    else:
        alpha2000 = alpha_in
        delta2000 = delta_in

    # From 2000.0 to to_equinox
    var alpha: Float64
    var delta: Float64
    if abs(to_equinox - 2000.0) > 0.02:
        var T = (to_equinox - 2000.0) / 100.0
        var zeta_A = 0.6406161 * T + 0.0000839 * T * T + 0.0000050 * T * T * T
        var z_A = 0.6406161 * T + 0.0003041 * T * T + 0.0000051 * T * T * T
        var theta_A = 0.5567530 * T - 0.0001185 * T * T - 0.0000116 * T * T * T

        var A = sin(degrad(alpha2000 + zeta_A)) * cos(degrad(delta2000))
        var B = (
            cos(degrad(alpha2000 + zeta_A)) * cos(degrad(theta_A)) * cos(degrad(delta2000))
            - sin(degrad(theta_A)) * sin(degrad(delta2000))
        )
        var C = (
            cos(degrad(alpha2000 + zeta_A)) * sin(degrad(theta_A)) * cos(degrad(delta2000))
            + cos(degrad(theta_A)) * sin(degrad(delta2000))
        )

        alpha = range(raddeg(atan2(A, B)) + z_A, 360.0)
        delta = raddeg(asin(C))
    else:
        alpha = alpha2000
        delta = delta2000

    return degrad(alpha), degrad(delta)


# ============================================================ eq_ecl.c
# sw == 1: x is ra, y is dec, output (lg, lt).
# sw == -1: x is lg, y is lt, output (ra, dec).
def ecleq_aux(sw: Int, mj: Float64, x: Float64, y: Float64) -> Tuple[Float64, Float64]:
    var eps = obliquity(mj)  # mean obliquity for date
    var seps = sin(eps)
    var ceps = cos(eps)

    var sy = sin(y)
    var cy = cos(y)  # always non-negative
    if abs(cy) < 1e-20:
        cy = 1e-20  # insure > 0
    var ty = sy / cy
    var cx = cos(x)
    var sx = sin(x)
    var sq = sy * ceps - cy * seps * sx * Float64(sw)
    if sq < -1.0:
        sq = -1.0
    if sq > 1.0:
        sq = 1.0
    var q = asin(sq)
    # Upstream computes atan(y/cx) and then adds PI when cx < 0; atan2 already
    # picks the right quadrant, so that fixup would double-count it.
    var p = atan2(sx * ceps + ty * seps * Float64(sw), cx)
    return range(p, 2.0 * 3.141592653589793), q


# Given mj and equatorial ra and dec, find geocentric ecliptic longitude
# and latitude.
def eq_ecl(mj: Float64, ra: Float64, dec: Float64) -> Tuple[Float64, Float64]:
    return ecleq_aux(1, mj, ra, dec)


# Given mj and geocentric ecliptic latitude and longitude, find equatorial
# ra and dec. The nutation effect on the obliquity is not included.
def ecl_eq(mj: Float64, lt: Float64, lg: Float64) -> Tuple[Float64, Float64]:
    return ecleq_aux(-1, mj, lg, lt)


# =========================================================== nutation.c
# NUT_SERIES is 106, NUT_MAXMUL is 4, NUT_SCALE is 1e4. Returns
# (nutation in obliquity, nutation in longitude) in radians.
def nutation(mj: Float64, reg: Reg) -> Tuple[Float64, Float64]:
    var delcache = StaticTuple[Float64, 45]()
    var T = (mj - 36525.0) / 36525.0
    var T2 = T * T
    var T3 = T2 * T
    var T10 = T / 10.0
    var lastdpsi: Float64 = 0.0
    var lastdeps: Float64 = 0.0

    # calculate delaunay args and place in cache
    var dl = tf(reg, 16)
    var i = 0
    var j = 0
    while i < 5:
        var x = dl.unsafe_offset(i * 4)[] + dl.unsafe_offset(i * 4 + 1)[] * T + dl.unsafe_offset(i * 4 + 2)[] * T2 + dl.unsafe_offset(i * 4 + 3)[] * T3

        # convert to radians
        x /= 3600.0 * 360.0
        x -= floor(x)
        x *= 2.0 * 3.141592653589793

        # fill cache table: [M',M,F,D,Om][-min*x, .., 0, .., max*x]
        j = 0
        while j < 9:
            delcache[i * 9 + j] = (Float64(j) - 4.0) * x

            j += 1
        i += 1
    # find dpsi and deps
    var mult = ti(reg, 17)
    var secul = ti(reg, 18)
    var con = ti(reg, 19)
    var isecul = 0
    i = 0
    while i < 106:
        var arg = 0.0
        var ampsin: Float64
        var ampcos: Float64
        if con.unsafe_offset(i * 2)[] != 0 or con.unsafe_offset(i * 2 + 1)[] != 0:
            # take non-secular terms from the simple array
            ampsin = Float64(con.unsafe_offset(i * 2)[])
            ampcos = Float64(con.unsafe_offset(i * 2 + 1)[])
        else:
            # secular terms from a different array
            ampsin = Float64(secul.unsafe_offset(isecul * 5 + 1)[]) + Float64(secul.unsafe_offset(isecul * 5 + 2)[]) * T10
            ampcos = Float64(secul.unsafe_offset(isecul * 5 + 3)[]) + Float64(secul.unsafe_offset(isecul * 5 + 4)[]) * T10
            isecul += 1

        j = 0
        while j < 5:
            arg += delcache[j * 9 + 4 + Int(mult.unsafe_offset(i * 5 + j)[])]
            j += 1
        lastdpsi += ampsin * sin(arg)
        lastdeps += ampcos * cos(arg)

        i += 1
    # convert to radians
    return (
        degrad(lastdeps / 3600.0 / 1e4),
        degrad(lastdpsi / 3600.0 / 1e4),
    )


# Given mj, correct the right ascension ra and declination dec (both in
# radians) for nutation.
def nut_eq(mj: Float64, ra: Float64, dec: Float64, reg: Reg) -> Tuple[Float64, Float64]:
    var epsilon = obliquity(mj)
    var deps, dpsi = nutation(mj, reg)

    # The rotation matrix a applies the nutation correction to a vector of
    # equatoreal coordinates Xeq to Xeq' by 3 subsequent rotations: R1 from
    # equatoreal to ecliptic by epsilon about x, R2 rotating the ecliptic by
    # -dpsi about its z, R3 from ecliptic to equatoreal by
    # -(epsilon + deps). Xeq' = A * Xeq = R3 * R2 * R1 * Xeq.
    var se = sin(epsilon)
    var ce = cos(epsilon)
    var sp = sin(dpsi)
    var cp = cos(dpsi)
    var sede = sin(epsilon + deps)
    var cede = cos(epsilon + deps)

    var a00 = cp
    var a01 = -sp * ce
    var a02 = -sp * se
    var a10 = cede * sp
    var a11 = cede * cp * ce + sede * se
    var a12 = cede * cp * se - sede * ce
    var a20 = sede * sp
    var a21 = sede * cp * ce - cede * se
    var a22 = sede * cp * se + cede * ce

    var xold, yold, zold = sphcart(ra, dec, 1.0)
    var x = a00 * xold + a01 * yold + a02 * zold
    var y = a10 * xold + a11 * yold + a12 * zold
    var z = a20 * xold + a21 * yold + a22 * zold
    var ra2, dec2, _ = cartsph(x, y, z)  # radius should be 1.0
    if ra2 < 0.0:
        ra2 += 2.0 * 3.141592653589793  # make positive for display
    return ra2, dec2


# ======================================================= aberration.c
# Apply the aberration correction to equatoreal coordinates ra and dec (in
# radians) for a given time m and a handily supplied longitude of the sun.
# Montenbruck's form: it agrees with Meeus to within 0.2 arcsec until dec
# exceeds 89.9 deg, and behaves smoothly with ta_par there.
def ab_eq(mj: Float64, lsn: Float64, ra: Float64, dec: Float64) -> Tuple[Float64, Float64]:
    var x, y, z = sphcart(ra, dec, 1.0)

    # helio long of earth
    var L = 2.0 * 3.141592653589793 * (0.27908 + 100.00214 * (mj - 36525.0) / 36525.0)
    var cL = cos(L)
    var vx = -0.994e-4 * sin(L)
    var vy = 0.912e-4 * cL
    var vz = 0.395e-4 * cL
    x += vx
    y += vy
    z += vz

    var ra2, dec2, _r = cartsph(x, y, z)
    return ra2, dec2


# Apply the aberration correction to ecliptical coordinates lam and bet (in
# radians) for a given time m and a handily supplied longitude of the sun.
def ab_ecl(
    mj: Float64, lsn: Float64, lam: Float64, bet: Float64
) -> Tuple[Float64, Float64]:
    var aberr_const = degrad(20.49552 / 3600.0)
    var T = (mj - 36525.0) / 36525.0  # centuries since J2000
    var eexc = 0.016708617 - (42.037e-6 + 0.1236e-6 * T) * T
    var leperi = degrad(102.93735 + (0.71953 + 0.00046 * T) * T)

    var dlsun = lsn - lam
    var dlperi = leperi - lam

    # valid only for bet != +-PI/2
    return (
        lam - aberr_const / cos(bet) * (cos(dlsun) - eexc * cos(dlperi)),
        bet - aberr_const * sin(bet) * (sin(dlsun) - eexc * sin(dlperi)),
    )


# ============================================================ vsop87.c
# Adapted from the BdL FORTRAN code: VSOP87D, heliocentric spherical,
# mean ecliptic of date. Returns (longitude, latitude, radius) as radians,
# radians and au.
def vsop87(
    mj: Float64, obj: Int, prec: Float64, reg: Reg
) -> Tuple[Float64, Float64, Float64]:
    # semimajor axes; for precision control only
    var a0 = StaticTuple[Float64, 9](
        0.39, 0.72, 1.5, 5.2, 9.6, 19.2, 30.1, 39.5, 1.0
    )

    # vx_map/vn_map indexed by PLCode; pluto has no VSOP87 table and the
    # sun reuses earth's.
    var slot: Int = 0
    if obj == 0:
        slot = 0  # mercury
    elif obj == 1:
        slot = 2  # venus
    elif obj == 2:
        slot = 6  # mars
    elif obj == 3:
        slot = 8  # jupiter
    elif obj == 4:
        slot = 10  # saturn
    elif obj == 5:
        slot = 12  # uranus
    elif obj == 6:
        slot = 14  # neptune
    elif obj == 8:
        slot = 4  # sun -> earth

    # The `a0[obj]` and `vn_obj` reads below are unchecked, so a negative or
    # out-of-range PLCode has to leave here rather than index off the front
    # of the tuple.
    if obj < 0 or obj == 7 or obj > 8:
        return 0.0, 0.0, 0.0  # object out of range
    if prec < 0.0 or prec > 1e-3:
        return 0.0, 0.0, 0.0  # precision out of range

    var vx_obj = tf(reg, slot)
    var vn_obj = ti(reg, slot + 1)

    # time and its powers
    var t = StaticTuple[Float64, 6](0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var t_abs = StaticTuple[Float64, 6](0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    t[0] = 1.0
    t[1] = (mj - 36525.0) / 365250.0
    var i = 2
    while i < 6:
        t[i] = t[i - 1] * t[1]
        i += 1
    t_abs[0] = 1.0
    i = 1
    while i < 6:
        t_abs[i] = abs(t[i])
        i += 1
    # precision control
    var q = -log10(prec + 1e-35) - 2.0  # decades below 1e-2
    q = 1e8 * prec / 10.0 / q  # reduce threshold for higher precision

    # do the term summation; first the spatial dimensions
    var ret = StaticTuple[Float64, 3](0.0, 0.0, 0.0)
    var cooidx = 0
    while cooidx < 3:
        # then the powers of time
        var alpha = 0
        while vn_obj.unsafe_offset((alpha + 1) * 3 + cooidx)[] != 0:
            var p: Float64
            # precision threshold
            if alpha != 0:
                p = q / (t_abs[alpha] + Float64(alpha) * t_abs[alpha - 1] * 1e-4 + 1e-35)
            else:
                p = q
            # scale by semimajor axis for radius
            p *= a0[obj]

            var term = 0.0
            i = Int(vn_obj.unsafe_offset(alpha * 3 + cooidx)[])
            while i < Int(vn_obj.unsafe_offset((alpha + 1) * 3 + cooidx)[]):
                var a = vx_obj.unsafe_offset(i * 3)[]
                if a >= p:
                    var b = vx_obj.unsafe_offset(i * 3 + 1)[]
                    var c = vx_obj.unsafe_offset(i * 3 + 2)[]
                    term += a * cos(b + c * t[1])
                i += 1

            ret[cooidx] += t[alpha] * term
            alpha += 1

        cooidx += 1
    i = 0
    while i < 3:
        ret[i] /= 1e8
        i += 1
    # reduce longitude to 0..2pi
    ret[0] -= floor(ret[0] / (2.0 * 3.141592653589793)) * (2.0 * 3.141592653589793)

    # reduction from the dynamical equinox of VSOP87 to FK5
    if prec < 5e-7:  # 5e-7 rad = 0.1 arc second
        var L1 = ret[0] - degrad(13.97 * t[1] - 0.031 * t[2])
        var c1 = cos(L1)
        var s1 = sin(L1)
        ret[0] += degrad(-0.09033 + 0.03916 * (c1 + s1) * tan(ret[1])) / 3600.0
        ret[1] += degrad(0.03916 * (c1 - s1)) / 3600.0

    return ret[0], ret[1], ret[2]


# ============================================================ chap95.c
# Heliocentric rectangular equatorial coordinates of Jupiter to Pluto,
# from Chapront's expansion of DE200; mean equator J2000.0. Returns
# (x, y, z) in au.
def chap95(m: Float64, obj: Int, prec: Float64, nrec: Int, reg: Reg) -> Tuple[Float64, Float64, Float64]:
    # semimajor axes for precision control
    var a0 = StaticTuple[Float64, 9](
        0.39, 0.72, 1.5, 5.2, 9.6, 19.2, 30.1, 39.5, 1.0
    )

    # JDs of validity period: 2338032.5 .. 2542032.5
    if m < 2338032.5 - 2415020.0 or m > 2542032.5 - 2415020.0:
        return 0.0, 0.0, 0.0
    if obj < 3 or obj > 7:
        return 0.0, 0.0, 0.0
    if prec < 0.0 or prec > 1e-3:
        return 0.0, 0.0, 0.0

    # The registry holds n/amp/nu triples for jupiter, saturn, uranus,
    # neptune and pluto in that order.
    var base = 34 + 3 * (obj - 3)
    var ns = ti(reg, base)
    var amps = tf(reg, base + 1)
    var nus = tf(reg, base + 2)

    var sum = StaticTuple[Float64, 9](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    var T = (m - 36525.0) / 36525.0  # centuries since J2000.0

    # modify the precision threshold for
    # a) term storing scale, b) converting radians to au, c) accounting for
    # skipped terms, d) augmenting for secular terms
    var precT = StaticTuple[Float64, 3](0.0, 0.0, 0.0)
    precT[0] = prec * 1e10 * a0[obj] / (10.0 * (-log10(prec + 1e-35) - 2.0))
    var t = 1.0 / (abs(T) + 1e-35)
    precT[1] = precT[0] * t
    precT[2] = precT[1] * t

    t = T * 100.0  # YEARS since J2000.0

    var ca: Float64 = 0.0
    var sa: Float64 = 0.0
    var Nu: Float64 = 0.0

    # do the term summation into sum[T^n] slots
    var rec = 0
    while rec < nrec:
        var n = ns.unsafe_offset(rec)[]
        var cooidx = 0
        while cooidx < 3:
            var amp = rec * 6 + cooidx * 2
            var C = amps.unsafe_offset(amp)[]
            var S = amps.unsafe_offset(amp + 1)[]

            # drop term if too small
            if abs(C) + abs(S) < precT[Int(n)]:
                continue

            if n == 0 and cooidx == 0:  # new Nu only here
                Nu = nus.unsafe_offset(rec)[]
                var arg = Nu * t
                arg -= floor(arg / (2.0 * 3.141592653589793)) * (2.0 * 3.141592653589793)
                ca = cos(arg)  # blast it - even for Nu = 0.0
                sa = sin(arg)

            sum[Int(n) * 3 + cooidx] += C * ca + S * sa

            cooidx += 1
        rec += 1
    # apply powers of time and sum up
    return (
        (sum[0] + T * (sum[3] + T * sum[6])) / 1e10,
        (sum[1] + T * (sum[4] + T * sum[7])) / 1e10,
        (sum[2] + T * (sum[5] + T * sum[8])) / 1e10,
    )


# ============================================================== sun.c
# Given the modified JD mj, return the true geocentric ecliptic longitude
# of the sun for the mean equinox of the date, the sun-earth distance in
# AU, and the latitude in radians.
def sunpos(mj: Float64, reg: Reg) -> Tuple[Float64, Float64, Float64]:
    var l, b, r = vsop87(mj, 8, 0.0, reg)  # full precision earth pos
    return range(l - 3.141592653589793, 2.0 * 3.141592653589793), r, -b


# ============================================================ rings.c
# Rings of Saturn by Olson et al, BASIC code from Sky & Telescope, May
# 1995. Finds the tilt from earth and from sun, radians south.
def satrings(
    sb: Float64, sl: Float64, sr: Float64, el: Float64, er: Float64, JD: Float64
) -> Tuple[Float64, Float64]:
    var t = (JD - 2451545.0) / 365250.0
    var i = degrad(28.04922 - 0.13 * t + 0.0004 * t * t)
    var om = degrad(169.53 + 13.826 * t + 0.04 * t * t)

    var x = sr * cos(sb) * cos(sl) - er * cos(el)
    var y = sr * cos(sb) * sin(sl) - er * sin(el)
    var z = sr * sin(sb)

    # Upstream computes atan(y/x) and then adds PI when x < 0; atan2 already
    # picks the right quadrant, so that fixup would double-count it.
    var la = atan2(y, x)
    var be = atan2(z, sqrt(x * x + y * y))

    var s = sin(i) * cos(be) * sin(la - om) - cos(i) * sin(be)
    var b = atan2(s, sqrt(1.0 - s * s))
    var sp = sin(i) * cos(sb) * sin(sl - om) - cos(i) * sin(sb)
    var bp = atan2(sp, sqrt(1.0 - sp * sp))

    return b, bp


# ============================================================ plans.c
# Visual elements of planets:
#   [planet][0] angular size at 1 AU
#   [planet][1] magnitude at 1 AU from sun and earth and 0 deg phase angle
#   [planet][2] A   [planet][3] B   [planet][4] C
#   where mag correction = A*(i/100) + B*(i/100)^2 + C*(i/100)^3 and
#   i = angle between sun and earth from planet, degrees.
# From Explanatory Supplement, 1992.
def vis_elements(p: Int) -> StaticTuple[Float64, 5]:
    if p == 0:  # Mercury
        return StaticTuple[Float64, 5](6.74, -0.36, 3.8, -2.73, 2.00)
    elif p == 1:  # Venus
        return StaticTuple[Float64, 5](16.92, -4.29, 0.09, 2.39, -0.65)
    elif p == 2:  # Mars
        return StaticTuple[Float64, 5](9.36, -1.52, 1.60, 0.0, 0.0)
    elif p == 3:  # Jupiter
        return StaticTuple[Float64, 5](196.74, -9.25, 0.50, 0.0, 0.0)
    elif p == 4:  # Saturn
        return StaticTuple[Float64, 5](165.6, -8.88, 4.40, 0.0, 0.0)
    elif p == 5:  # Uranus
        return StaticTuple[Float64, 5](70.481, -7.19, 0.28, 0.0, 0.0)
    elif p == 6:  # Neptune
        return StaticTuple[Float64, 5](68.294, -6.87, 0.0, 0.0, 0.0)
    else:  # Pluto
        return StaticTuple[Float64, 5](8.2, -1.01, 4.1, 0.0, 0.0)


# Coordinate transformation from J2000.0 rectangular equatoreal {x,y,z}
# to mean equinox of date spherical ecliptical {l,b,r}.
def chap_trans(mj: Float64, x: Float64, y: Float64, z: Float64) -> Tuple[Float64, Float64, Float64]:
    var ra, dec, r = cartsph(x, y, z)
    ra, dec = precess(36525.0, mj, ra, dec)
    var eps = obliquity(mj)
    var sr = sin(ra)
    var cr = cos(ra)
    var sd = sin(dec)
    var cd = cos(dec)
    var se = sin(eps)
    var ce = cos(eps)
    return atan2(sr * ce + sd / cd * se, cr), asin(sd * ce - cd * se * sr), r


# Low precision ecliptic coordinates of Pluto from mean orbit. Only for
# completeness outside the available perturbation theories.
def pluto_ell(mj: Float64) -> Tuple[Float64, Float64, Float64]:
    # Mean orbital elements of Pluto, Astronomical Almanac 2020 page G3.
    var a = 39.789  # semimajor axis, au
    var e = 0.252  # excentricity
    var inc0 = 17.097  # inclination, deg
    var Om0 = 110.297  # long asc node, deg
    var omeg0 = 115.058  # arg of perihel, deg
    var mjp = 2459000.5 - 2415020.0  # epoch of perihel
    var mjeq = 36525.0  # equinox of elements
    var n = 0.0039  # daily motion, deg

    var inc, omeg, Om = reduce_elements(
        mjeq, mj, degrad(inc0), degrad(omeg0), degrad(Om0)
    )
    var ma = degrad((mj - mjp) * n)
    var nu, ea = anomaly(ma, e)
    var r = a * (1.0 - e * cos(ea))
    var lo = omeg + nu
    var slo = sin(lo)
    var clo = cos(lo)
    var b = asin(slo * sin(inc))
    return atan2(slo * cos(inc), clo) + Om, b, r


# Geometric heliocentric position of a planet, mean ecliptic of date
# (not corrected for light-time).
def planpos(
    mj: Float64, obj: Int, prec: Float64, nrec: Int, reg: Reg
) -> Tuple[Float64, Float64, Float64]:
    if mj >= 2338032.5 - 2415020.0 and mj <= 2542032.5 - 2415020.0:
        if obj >= 3 and obj <= 7:  # prefer Chapront for the outer planets
            var x, y, z = chap95(mj, obj, prec, nrec, reg)
            return chap_trans(mj, x, y, z)
        else:  # VSOP for the inner planets
            return vsop87(mj, obj, prec, reg)
    else:  # outside the Chapront time span
        if obj != 7:  # VSOP for all but Pluto
            return vsop87(mj, obj, prec, reg)
        else:  # Pluto mean elliptic orbit
            return pluto_ell(mj)


# Given a modified Julian date mj and a planet p, find lpd0 (heliocentric
# longitude), psi0 (heliocentric latitude), rp0 (sun distance), rho0
# (earth distance, none corrected for light time), lam and bet
# (geocentric ecliptic longitude and latitude, each corrected for light
# time), dia (angular diameter in arcsec at 1 AU) and mag (visual
# magnitude). All angles are in radians, all distances in AU.
def plans(
    mj: Float64,
    p: Int,
    nrec: Int,
    lsn: Float64,
    rsn: Float64,
    bsn: Float64,
    reg: Reg,
) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64]:
    # The solar ecliptical longitude (lsn), sun distance (rsn) and latitude
    # (bsn) at `mj` are supplied by the caller. obj_planet already computed
    # them for the same mjed and passes them straight through, so the sunpos
    # call is not repeated here.
    var xsn, ysn, zsn = sphcart(lsn, bsn, rsn)

    # first find the true position of the planet at mj, then repeat a second
    # time for a slightly different time based on the position found in the
    # first pass to account for light-travel time.
    var dt = 0.0
    var lp: Float64 = 0.0
    var bp: Float64 = 0.0
    var rp: Float64 = 0.0
    var lpd0: Float64 = 0.0
    var psi0: Float64 = 0.0
    var rp0: Float64 = 0.0
    var rho0: Float64 = 0.0
    var rho: Float64 = 0.0
    var lam: Float64 = 0.0
    var bet: Float64 = 0.0
    # upstream's C variable is `pass`; renamed because this dialect reserves it
    var passno = 0
    while passno < 2:
        # get spherical coordinates of the planet from the precision
        # routines, retarded for light time in the second pass
        lp, bp, rp = planpos(mj - dt, p, 0.0, nrec, reg)

        var xp, yp, zp = sphcart(lp, bp, rp)
        lam, bet, rho = cartsph(xp + xsn, ysn + yp, zp + zsn)

        if passno == 0:
            # save heliocentric coordinates at first pass since, being true,
            # they are NOT to be corrected for light-travel time
            lpd0 = range(lp, 2.0 * 3.141592653589793)
            psi0 = bp
            rp0 = rp
            # rho0 is the geometric earth-planet distance; the light-time
            # corrected direction below is what lam and bet carry.
            rho0 = rho

        # when we view a planet we see it in the position it occupied dt days
        # ago, where rho is the distance between it and earth, in AU
        dt = rho * 5.7755183e-3

        passno += 1
    var vp = vis_elements(p)
    var dia = vp[0]

    # solve plane triangle, assume sun/earth dist == 1
    var ci = (rp0 * rp0 + rho0 * rho0 - 1.0) / (2.0 * rp0 * rho0)

    # expl supp equation for mag
    if ci < -1.0:
        ci = -1.0
    if ci > 1.0:
        ci = 1.0
    var i = raddeg(acos(ci)) / 100.0
    var mag = vp[1] + 5.0 * log10(rho0 * rp0) + i * (vp[2] + i * (vp[3] + i * vp[4]))

    # rings contribution if SATURN
    if p == 4:
        var et, _st = satrings(bp, lp, rp, lsn + 3.141592653589793, rsn, mj + 2415020.0)
        var set = sin(abs(et))
        mag += (-2.60 + 1.25 * set) * set

    return lpd0, psi0, rp0, rho0, lam, bet, dia, mag


# ============================================================= moon.c
# Stephen Moshier's truncated ELP2000-82. NARGS is 18 and T is time in
# units of 10,000 Julian years from JED 2451545.0.
comptime STR: Float64 = 4.8481368110953599359e-6  # radians per arc second
comptime AUKM: Float64 = 1.4959787e8
comptime EARTHRADIUS: Float64 = 6378.16  # kilometers


# Reduce arc seconds modulo 360 degrees; the answer is in arc seconds.
def mods3600(x: Float64) -> Float64:
    return x - 1296000.0 * floor(x / 1296000.0)


# Calculate sin(i*MM), cos(i*MM) for argument k (0..17) up to harmonic n,
# into the 30-wide row k of the two caches.
def sscc(
    mut ss: StaticTuple[Float64, 540],
    mut cc: StaticTuple[Float64, 540],
    k: Int,
    arg: Float64,
    n: Int,
):
    var s = STR * arg
    var su = sin(s)
    var cu = cos(s)
    ss[k * 30] = su
    cc[k * 30] = cu
    var sv = 2.0 * su * cu
    var cv = cu * cu - su * su
    ss[k * 30 + 1] = sv
    cc[k * 30 + 1] = cv
    var i = 2
    while i < n:
        s = su * cv + cu * sv
        cv = cu * cv - su * sv
        sv = s
        ss[k * 30 + i] = sv
        cc[k * 30 + i] = cv


        i += 1
# Time argument is a Julian ephemeris date. Fills the mean-element cache
# `args` and the three scalars upstream keeps beside it.
def mean_elements(
    mut args: StaticTuple[Float64, 18],
    jed: Float64,
) -> Tuple[Float64, Float64, Float64]:
    var T = (jed - 2451545.0) / 36525.0  # Julian centuries
    var T2 = T * T
    var x: Float64

    # Mean longitudes of planets (Simon et al, 1994); .047" subtracted from
    # the constant term for the offset to the DE403 origin.

    # Mercury
    x = mods3600(538101628.6889819 * T + 908103.213)
    x += (6.39e-6 * T - 0.0192789) * T2
    args[0] = x

    # Venus
    x = mods3600(210664136.4335482 * T + 655127.236)
    x += (-6.27e-6 * T + 0.0059381) * T2
    args[1] = x

    # Earth
    # upstream also caches this as Ea_arcsec, which only its own libration
    # path reads; this port does not.
    x = mods3600(129597742.283429 * T + 361679.198)
    x += (-5.23e-6 * T - 2.04411e-2) * T2
    args[2] = x

    # Mars
    x = mods3600(68905077.493988 * T + 1279558.751)
    x += (-1.043e-5 * T + 0.0094264) * T2
    args[3] = x

    # Jupiter
    x = mods3600(10925660.377991 * T + 123665.420)
    x += (((((-3.4e-10 * T + 5.91e-8) * T + 4.667e-6) * T + 5.706e-5) * T - 3.060378e-1) * T2)
    args[4] = x

    # Saturn
    x = mods3600(4399609.855372 * T + 180278.752)
    x += ((((8.3e-10 * T - 1.452e-7) * T - 1.1484e-5) * T - 1.6618e-4) * T + 7.561614E-1) * T2
    args[5] = x

    # Uranus
    x = mods3600(1542481.193933 * T + 1130597.971) + (0.00002156 * T - 0.0175083) * T2
    args[6] = x

    # Neptune
    x = mods3600(786550.320744 * T + 1095655.149) + (-0.00000895 * T + 0.0021103) * T2
    args[7] = x

    # Mean elongation of moon = D, copied from cmoon.c, DE404 version
    x = mods3600(1.6029616009939659e+09 * T + 1.0722612202445078e+06)
    x += (((((-3.207663637426e-013 * T + 2.555243317839e-011) * T + 2.560078201452e-009) * T
        - 3.702060118571e-005) * T + 6.9492746836058421e-03) * T
        - 6.7352202374457519e+00) * T2
    args[9] = x

    # Mean distance of moon from its ascending node = F
    x = mods3600(1.7395272628437717e+09 * T + 3.3577951412884740e+05)
    x += (((((4.474984866301e-013 * T + 4.189032191814e-011) * T - 2.790392351314e-009) * T
        - 2.165750777942e-006) * T - 7.5311878482337989e-04) * T
        - 1.3117809789650071e+01) * T2
    var NF_arcsec = x
    args[10] = x

    # Mean anomaly of sun = l' (J. Laskar)
    x = mods3600(1.2959658102304320e+08 * T + 1.2871027407441526e+06)
    x += ((((((((1.62e-20 * T - 1.0390e-17) * T - 3.83508e-15) * T + 4.237343e-13) * T
        + 8.8555011e-11) * T - 4.77258489e-8) * T - 1.1297037031e-5) * T
        + 8.7473717367324703e-05) * T - 5.5281306421783094e-01) * T2
    args[11] = x

    # Mean anomaly of moon = l
    x = mods3600(1.7179159228846793e+09 * T + 4.8586817465825332e+05)
    x += (((((-1.755312760154e-012) * T + 3.452144225877e-011 * T - 2.506365935364e-008) * T
        - 2.536291235258e-004) * T + 5.2099641302735818e-02) * T
        + 3.1501359071894147e+01) * T2
    args[12] = x

    # Mean longitude of moon, re mean ecliptic and equinox of date = L
    x = mods3600(1.7325643720442266e+09 * T + 7.8593980921052420e+05)
    x += (((((7.200592540556e-014 * T + 2.235210987108e-010) * T - 1.024222633731e-008) * T
        - 6.073960534117e-005) * T + 6.9017248528380490e-03) * T
        - 5.6550460027471399e+00) * T2
    var LP_equinox = x
    args[13] = x

    # Precession of the equinox
    var pA_precession = (((((((((-8.66e-20 * T - 4.759e-17) * T + 2.424e-15) * T
        + 1.3095e-12) * T + 1.7451e-10) * T - 1.8055e-8) * T - 0.0000235316) * T
        + 0.000076) * T + 1.105414) * T + 5028.791959) * T

    # Free librations.  LB 2.891725 years, psi amplitude 1.8"
    args[14] = mods3600(4.48175409e7 * T + 8.060457e5)

    # 24.2 years
    args[15] = mods3600(5.36486787e6 * T - 391702.8)

    # LA 74.7 years.
    args[17] = mods3600(1.73573e6 * T)

    return LP_equinox, NF_arcsec, pA_precession


# Generic program to accumulate the sum of trigonometric series in one
# variable. `base` selects the moonlr (20) or moonlat (27) registry block.
# Returns (sl, sr): the accumulated longitude and radius.
def g2plan(
    jed: Float64, reg: Reg, base: Int, flag: Int
) -> Tuple[Float64, Float64]:
    var args = StaticTuple[Float64, 18]()
    var LP_equinox, NF_arcsec, pA_precession = mean_elements(args, jed)

    # For librations, moon's longitude is sidereal.
    if flag != 0:
        args[13] -= pA_precession

    # struct plantbl holds distance, timescale, trunclvl in that order.
    var scalars = tf(reg, 26 if base == 20 else 33)
    var T = (jed - 2451545.0) / scalars.unsafe_offset(1)[]

    var ss = StaticTuple[Float64, 540]()
    var cc = StaticTuple[Float64, 540]()
    var harmonic = ti(reg, base)
    var i = 0
    while i < 18:
        var j = Int(harmonic.unsafe_offset(i)[])
        if j > 0:
            sscc(ss, cc, i, args[i], j)
        i += 1

    # Upstream keeps a separate cursor into the argument list and into each
    # of the two coefficient lists; they advance in lockstep but not in step.
    var argtbl = ti(reg, base + 2)
    var lontbl = ti(reg, base + 3)
    var radtbl = ti(reg, base + 5)
    var pa = 0
    var pl = 0
    var pr = 0

    var sl = 0.0
    var sr = 0.0
    var ip = 0
    while True:
        # number of periodic arguments
        var np = Int(argtbl.unsafe_offset(pa)[])
        pa += 1
        if np < 0:
            break
        if np == 0:
            # it is a polynomial term
            var nt = Int(argtbl.unsafe_offset(pa)[])
            pa += 1
            # longitude polynomial
            var cu = Float64(lontbl.unsafe_offset(pl)[])
            pl += 1
            ip = 0
            while ip < nt:
                cu = cu * T + Float64(lontbl.unsafe_offset(pl)[])
                pl += 1
                ip += 1
            sl += cu
            # radius polynomial
            cu = Float64(radtbl.unsafe_offset(pr)[])
            pr += 1
            ip = 0
            while ip < nt:
                cu = cu * T + Float64(radtbl.unsafe_offset(pr)[])
                pr += 1
                ip += 1
            sr += cu
            continue

        var k1 = 0
        var cv = 0.0
        var sv = 0.0
        ip = 0
        while ip < np:
            # which harmonic
            var j = Int(argtbl.unsafe_offset(pa)[])
            pa += 1
            # which planet
            var m = Int(argtbl.unsafe_offset(pa)[]) - 1
            pa += 1
            if j != 0:
                var k = abs(j) - 1
                var su = ss[m * 30 + k]  # sin(k*angle)
                if j < 0:
                    su = -su
                var cu = cc[m * 30 + k]
                if k1 == 0:
                    sv = su  # set first angle
                    cv = cu
                    k1 = 1
                else:
                    var t = su * cv + cu * sv  # combine angles
                    cv = cu * cv - su * sv
                    sv = t
            ip += 1
        # highest power of T
        var nt = Int(argtbl.unsafe_offset(pa)[])
        pa += 1
        # longitude
        var cu = Float64(lontbl.unsafe_offset(pl)[])
        var su = Float64(lontbl.unsafe_offset(pl + 1)[])
        pl += 2
        ip = 0
        while ip < nt:
            cu = cu * T + Float64(lontbl.unsafe_offset(pl)[])
            su = su * T + Float64(lontbl.unsafe_offset(pl + 1)[])
            pl += 2
            ip += 1
        sl += cu * cv + su * sv
        # radius
        cu = Float64(radtbl.unsafe_offset(pr)[])
        su = Float64(radtbl.unsafe_offset(pr + 1)[])
        pr += 2
        ip = 0
        while ip < nt:
            cu = cu * T + Float64(radtbl.unsafe_offset(pr)[])
            su = su * T + Float64(radtbl.unsafe_offset(pr + 1)[])
            pr += 2
            ip += 1
        sr += cu * cv + su * sv

    var t = scalars.unsafe_offset(2)[]  # plan->trunclvl
    return t * sl, t * sr


# The single-variable sibling of g2plan, used for latitude; it accumulates
# only the longitude-like series.
def g1plan(jed: Float64, reg: Reg, base: Int) -> Float64:
    var args = StaticTuple[Float64, 18]()
    var LP_equinox, NF_arcsec, pA_precession = mean_elements(args, jed)

    # struct plantbl holds distance, timescale, trunclvl in that order.
    var scalars = tf(reg, 26 if base == 20 else 33)
    var T = (jed - 2451545.0) / scalars.unsafe_offset(1)[]

    var ss = StaticTuple[Float64, 540]()
    var cc = StaticTuple[Float64, 540]()
    var harmonic = ti(reg, base)
    var i = 0
    while i < 18:
        var j = Int(harmonic.unsafe_offset(i)[])
        if j > 0:
            sscc(ss, cc, i, args[i], j)
        i += 1

    var argtbl = ti(reg, base + 2)
    var lontbl = ti(reg, base + 3)
    var pa = 0
    var pl = 0

    var sl = 0.0
    var ip = 0
    while True:
        var np = Int(argtbl.unsafe_offset(pa)[])
        pa += 1
        if np < 0:
            break
        if np == 0:
            var nt = Int(argtbl.unsafe_offset(pa)[])
            pa += 1
            var cu = Float64(lontbl.unsafe_offset(pl)[])
            pl += 1
            ip = 0
            while ip < nt:
                cu = cu * T + Float64(lontbl.unsafe_offset(pl)[])
                pl += 1
                ip += 1
            sl += cu
            continue

        var k1 = 0
        var cv = 0.0
        var sv = 0.0
        ip = 0
        while ip < np:
            var j = Int(argtbl.unsafe_offset(pa)[])
            pa += 1
            var m = Int(argtbl.unsafe_offset(pa)[]) - 1
            pa += 1
            if j != 0:
                var k = abs(j) - 1
                var su = ss[m * 30 + k]
                if j < 0:
                    su = -su
                var cu = cc[m * 30 + k]
                if k1 == 0:
                    sv = su
                    cv = cu
                    k1 = 1
                else:
                    var t = su * cv + cu * sv
                    cv = cu * cv - su * sv
                    sv = t
            ip += 1
        var nt = Int(argtbl.unsafe_offset(pa)[])
        pa += 1
        var cu = Float64(lontbl.unsafe_offset(pl)[])
        var su = Float64(lontbl.unsafe_offset(pl + 1)[])
        pl += 2
        ip = 0
        while ip < nt:
            cu = cu * T + Float64(lontbl.unsafe_offset(pl)[])
            su = su * T + Float64(lontbl.unsafe_offset(pl + 1)[])
            pl += 2
            ip += 1
        sl += cu * cv + su * sv

    return scalars.unsafe_offset(2)[] * sl  # plan->trunclvl * sl

def gecmoon(jed: Float64, reg: Reg) -> Tuple[Float64, Float64, Float64]:
    var args = StaticTuple[Float64, 18]()
    var LP_equinox, _, _pA = mean_elements(args, jed)
    var scalars = tf(reg, 26)

    var sl, sr = g2plan(jed, reg, 20, 0)
    var x = sl
    x += LP_equinox
    if x < -6.45e5:
        x += 1.296e6
    if x > 6.45e5:
        x -= 1.296e6
    var lam = STR * x
    var bet = STR * g1plan(jed, reg, 27)
    return lam, bet, (STR * sr + 1.0) * scalars.unsafe_offset(0)[]


# The older, faster moon series; good to about 10 arcsec in longitude and
# 3 in latitude, and the only one outside Moshier's validity window.
def moon_fast(mj: Float64) -> Tuple[Float64, Float64, Float64, Float64, Float64]:
    var t = mj / 36525.0
    var t2 = t * t

    var m1 = mj / 27.32158213
    m1 = 360.0 * (m1 - floor(m1))
    var m2 = mj / 365.2596407
    m2 = 360.0 * (m2 - floor(m2))
    var m3 = mj / 27.55455094
    m3 = 360.0 * (m3 - floor(m3))
    var m4 = mj / 29.53058868
    m4 = 360.0 * (m4 - floor(m4))
    var m5 = mj / 27.21222039
    m5 = 360.0 * (m5 - floor(m5))
    var m6 = mj / 6798.363307
    m6 = 360.0 * (m6 - floor(m6))

    var ld = 270.434164 + m1 - (0.001133 - 0.0000019 * t) * t2
    var ms = 358.475833 + m2 - (0.00015 + 0.0000033 * t) * t2
    var md = 296.104608 + m3 + (0.009192 + 0.0000144 * t) * t2
    var de = 350.737486 + m4 - (0.001436 - 0.0000019 * t) * t2
    var f = 11.250889 + m5 - (0.003211 + 0.0000003 * t) * t2
    var n = 259.183275 - m6 + (0.002078 + 0.000022 * t) * t2

    var a = degrad(51.2 + 20.2 * t)
    var sa = sin(a)
    var sn = sin(degrad(n))
    var b = 346.56 + (132.87 - 0.0091731 * t) * t
    var sb = 0.003964 * sin(degrad(b))
    var c = degrad(n + 275.05 - 2.3 * t)
    var sc = sin(c)
    ld = ld + 0.000233 * sa + sb + 0.001964 * sn
    ms = ms - 0.001778 * sa
    md = md + 0.000817 * sa + sb + 0.002541 * sn
    f = f + sb - 0.024691 * sn - 0.004328 * sc
    de = de + 0.002011 * sa + sb + 0.001964 * sn
    var e = 1 - (0.002495 + 7.52e-06 * t) * t
    var e2 = e * e

    ld = degrad(ld)
    ms = degrad(ms)
    n = degrad(n)
    de = degrad(de)
    f = degrad(f)
    md = degrad(md)

    var l = (
        6.28875 * sin(md) + 1.27402 * sin(2 * de - md) + 0.658309 * sin(2 * de)
        + 0.213616 * sin(2 * md) - e * 0.185596 * sin(ms) - 0.114336 * sin(2 * f)
        + 0.058793 * sin(2 * (de - md)) + 0.057212 * e * sin(2 * de - ms - md)
        + 0.05332 * sin(2 * de + md) + 0.045874 * e * sin(2 * de - ms)
        + 0.041024 * e * sin(md - ms)
    )
    l = (
        l - 0.034718 * sin(de) - e * 0.030465 * sin(ms + md) + 0.015326 * sin(2 * (de - f))
        - 0.012528 * sin(2 * f + md) - 0.01098 * sin(2 * f - md) + 0.010674 * sin(4 * de - md)
        + 0.010034 * sin(3 * md) + 0.008548 * sin(4 * de - 2 * md) - e * 0.00791 * sin(ms - md + 2 * de)
        - e * 0.006783 * sin(2 * de + ms)
    )
    l = (
        l + 0.005162 * sin(md - de) + e * 0.005 * sin(ms + de) + 0.003862 * sin(4 * de)
        + e * 0.004049 * sin(md - ms + 2 * de) + 0.003996 * sin(2 * (md + de))
        + 0.003665 * sin(2 * de - 3 * md) + e * 0.002695 * sin(2 * md - ms)
        + 0.002602 * sin(md - 2 * (f + de)) + e * 0.002396 * sin(2 * (de - md) - ms)
        - 0.002349 * sin(md + de)
    )
    l = (
        l + e2 * 0.002249 * sin(2 * (de - ms)) - e * 0.002125 * sin(2 * md + ms)
        - e2 * 0.002079 * sin(2 * ms) + e2 * 0.002059 * sin(2 * (de - ms) - md)
        - 0.001773 * sin(md + 2 * (de - f)) - 0.001595 * sin(2 * (f + de))
        + e * 0.00122 * sin(4 * de - ms - md) - 0.00111 * sin(2 * (md + f))
        + 0.000892 * sin(md - 3 * de)
    )
    l = (
        l - e * 0.000811 * sin(ms + md + 2 * de) + e * 0.000761 * sin(4 * de - ms - 2 * md)
        + e2 * 0.000704 * sin(md - 2 * (ms + de)) + e * 0.000693 * sin(ms - 2 * (md - de))
        + e * 0.000598 * sin(2 * (de - f) - ms) + 0.00055 * sin(md + 4 * de)
        + 0.000538 * sin(4 * md) + e * 0.000521 * sin(4 * de - ms)
        + 0.000486 * sin(2 * md - de)
    )
    l = l + e2 * 0.000717 * sin(md - 2 * ms)
    var lam = range(ld + degrad(l), 2.0 * 3.141592653589793)

    var g = (
        5.12819 * sin(f) + 0.280606 * sin(md + f) + 0.277693 * sin(md - f)
        + 0.173238 * sin(2 * de - f) + 0.055413 * sin(2 * de + f - md) + 0.046272 * sin(2 * de - f - md)
        + 0.032573 * sin(2 * de + f) + 0.017198 * sin(2 * md + f) + 0.009267 * sin(2 * de + md - f)
        + 0.008823 * sin(2 * md - f) + e * 0.008247 * sin(2 * de - ms - f)
    )
    g = (
        g + 0.004323 * sin(2 * (de - md) - f) + 0.0042 * sin(2 * de + f + md)
        + e * 0.003372 * sin(f - ms - 2 * de) + e * 0.002472 * sin(2 * de + f - ms - md)
        + e * 0.002222 * sin(2 * de + f - ms) + e * 0.002072 * sin(2 * de - f - ms - md)
        + e * 0.001877 * sin(f - ms + md) + 0.001828 * sin(4 * de - f - md) - e * 0.001803 * sin(f + ms)
        - 0.00175 * sin(3 * f)
    )
    g = (
        g + e * 0.00157 * sin(md - ms - f) - 0.001487 * sin(f + de) - e * 0.001481 * sin(f + ms + md)
        + e * 0.001417 * sin(f - ms - md) + e * 0.00135 * sin(f - ms) + 0.00133 * sin(f - de)
        + 0.001106 * sin(f + 3 * md) + 0.00102 * sin(4 * de - f) + 0.000833 * sin(f + 4 * de - md)
        + 0.000781 * sin(md - 3 * f) + 0.00067 * sin(f + 4 * de - 2 * md)
    )
    g = (
        g + 0.000606 * sin(2 * de - 3 * f) + 0.000597 * sin(2 * (de + md) - f)
        + e * 0.000492 * sin(2 * de + md - ms - f) + 0.00045 * sin(2 * (md - de) - f)
        + 0.000439 * sin(3 * md - f) + 0.000423 * sin(f + 2 * (de + md))
        + 0.000422 * sin(2 * de - f - 3 * md) - e * 0.000367 * sin(ms + f + 2 * de - md)
        - e * 0.000353 * sin(ms + f + 2 * de) + 0.000331 * sin(f + 4 * de)
    )
    g = (
        g + e * 0.000317 * sin(2 * de + f - ms + md) + e2 * 0.000306 * sin(2 * (de - ms) - f)
        - 0.000283 * sin(md + 3 * f)
    )
    var w1 = 0.0004664 * cos(n)
    var w2 = 0.0000754 * cos(c)
    var bet = degrad(g) * (1 - w1 - w2)

    var hp = (
        0.950724 + 0.051818 * cos(md) + 0.009531 * cos(2 * de - md) + 0.007843 * cos(2 * de)
        + 0.002824 * cos(2 * md) + 0.000857 * cos(2 * de + md) + e * 0.000533 * cos(2 * de - ms)
        + e * 0.000401 * cos(2 * de - md - ms) + e * 0.00032 * cos(md - ms) - 0.000271 * cos(de)
        - e * 0.000264 * cos(ms + md) - 0.000198 * cos(2 * f - md)
    )
    hp = (
        hp + 0.000173 * cos(3 * md) + 0.000167 * cos(4 * de - md) - e * 0.000111 * cos(ms)
        + 0.000103 * cos(4 * de - 2 * md) - 0.000084 * cos(2 * md - 2 * de)
        - e * 0.000083 * cos(2 * de + ms) + 0.000079 * cos(2 * de + 2 * md) + 0.000072 * cos(4 * de)
        + e * 0.000064 * cos(2 * de - ms + md) - e * 0.000063 * cos(2 * de + ms - md)
        + e * 0.000041 * cos(ms + de)
    )
    hp = (
        hp + e * 0.000035 * cos(2 * md - ms) - 0.000033 * cos(3 * md - 2 * de)
        - 0.00003 * cos(md + de) - 0.000029 * cos(2 * (f - de)) - e * 0.000029 * cos(2 * md + ms)
        + e2 * 0.000026 * cos(2 * (de - ms)) - 0.000023 * cos(2 * (f - de) + md)
        + e * 0.000019 * cos(4 * de - ms - md)
    )
    hp = degrad(hp)

    return lam, bet, hp, ms, md


# Given the mjd, find the geocentric ecliptic longitude lam, the latitude
# bet, the earth-moon distance rho in au, and the sun's and the moon's mean
# anomalies. Mosher's range is 1221000.5 .. 2798525.5.
def moon(mj: Float64, reg: Reg) -> Tuple[Float64, Float64, Float64, Float64, Float64]:
    var lam: Float64
    var bet: Float64
    var rho: Float64
    var msp: Float64
    var mdp: Float64

    if mj >= 1221000.5 - 2415020.0 and mj <= 2798525.5 - 2415020.0:
        var fast_lam, fast_bet, hp, fast_ms, fast_md = moon_fast(mj)
        rho = EARTHRADIUS / AUKM / sin(hp)
        # retard for light time
        var dt = rho * 5.7755183e-3  # speed of light in a.u/day
        var elam, ebet, erho = gecmoon(mj + 2415020.0 - dt, reg)
        lam = range(elam, 2.0 * 3.141592653589793)
        bet = ebet
        rho = erho
        msp = fast_ms
        mdp = fast_md
    else:
        var fast_lam, fast_bet, hp, fast_ms, fast_md = moon_fast(mj)
        lam = fast_lam
        bet = fast_bet
        rho = EARTHRADIUS / AUKM / sin(hp)
        msp = fast_ms
        mdp = fast_md
    return lam, bet, rho, msp, mdp


# ============================================================= misc.c
# Solve the spherical triangle; used by the alt/az transforms. Returns
# (cap, B) where cap is cos of the angle opposite and B the azimuthal angle.
def solve_sphere(
    A: Float64, b: Float64, cc: Float64, sc: Float64
) -> Tuple[Float64, Float64]:
    var cb = cos(b)
    var sb = sin(b)
    var cA = cos(A)

    var ca = cb * cc + sb * sc * cA
    if ca > 1.0:
        ca = 1.0
    if ca < -1.0:
        ca = -1.0

    var B: Float64
    if sc < 1e-7:
        B = (A if cc < 0 else 3.141592653589793 - A)
    else:
        var sA = sin(A)
        var y = sA * sb * sc
        var x = cb - ca * cc
        if y != 0.0:
            if x != 0.0:
                B = atan2(y, x)
            else:
                B = (3.141592653589793 / 2.0 if y > 0 else -3.141592653589793 / 2.0)
        else:
            B = (0.0 if x >= 0 else 3.141592653589793)
    return ca, range(B, 2.0 * 3.141592653589793)


# Given a Now, find the local apparent sidereal time, in hours.
def now_lst(mjd: Float64, lng: Float64, reg: Reg) -> Float64:
    var lst = utc_gst(mjd_day(mjd), mjd_hr(mjd))
    lst += radhr(lng)

    var eps = obliquity(mjd)
    var deps, dpsi = nutation(mjd, reg)
    lst += radhr(dpsi * cos(eps + deps))

    return range(lst, 24.0)


# Given ra and dec (radians), find the hour angle in range 0..2*PI.
def radec2ha(lst: Float64, ra: Float64) -> Float64:
    var ha = hrrad(lst) - ra
    if ha < 0:
        ha += 2.0 * 3.141592653589793
    return ha


# ========================================================== utc_gst.c
# Given a modified julian date mj and a universally coordinated time utc,
# return greenwich mean sidereal time in hours. mj must be at the beginning
# of the day.
def utc_gst(mj: Float64, utc: Float64) -> Float64:
    return range(utc / 0.9972695677 + gmst0(mj), 24.0)


# Given a modified julian date mj and a greenwich mean sidereal time gst,
# return universally coordinated time in hours.
def gst_utc(mj: Float64, gst: Float64) -> Float64:
    return range(gst - gmst0(mj), 24.0) * 0.9972695677


# Return Greenwich Mean Sidereal Time at 0h UT.
def gmst0(mj: Float64) -> Float64:
    var T = (Float64(Int(mj - 0.5)) + 0.5 - 36525.0) / 36525.0
    var x = 24110.54841 + (8640184.812866 + (0.093104 - 6.2e-6 * T) * T) * T
    x /= 3600.0
    return range(x, 24.0)


# ========================================================== aa_hadec.c
# The formula is the same for both transformation directions, so it is
# written once for each way. Returns (p, q).
def aaha_aux(lt: Float64, x: Float64, y: Float64) -> Tuple[Float64, Float64]:
    var slt = sin(lt)
    var clt = cos(lt)
    var cap, B = solve_sphere(-x, 3.141592653589793 / 2.0 - y, slt, clt)
    return B, 3.141592653589793 / 2.0 - acos(cap)


# Given geographical latitude lt (north positive, radians), altitude alt
# (up positive) and azimuth az (east from north), return hour angle and
# declination.
def aa_hadec(
    lt: Float64, alt: Float64, az: Float64
) -> Tuple[Float64, Float64]:
    var ha, dec = aaha_aux(lt, az, alt)
    if ha > 3.141592653589793:
        ha -= 2.0 * 3.141592653589793
    return ha, dec


# Given geographical latitude lt, hour angle ha and declination dec, return
# altitude and azimuth. aaha_aux yields (azimuth, altitude); the C signature
# asks for them the other way round.
def hadec_aa(
    lt: Float64, ha: Float64, dec: Float64
) -> Tuple[Float64, Float64]:
    var az, alt = aaha_aux(lt, ha, dec)
    return alt, az


# Given a geographic (surface-normal) latitude phi, return the geocentric
# latitude psi.
def geoc_lat(phi: Float64) -> Float64:
    var maxlat = degrad(89.9999)  # avoid tan() greater than this
    if abs(phi) > maxlat:
        return phi
    return atan2(tan(phi) / 1.00674, 1.0)


# =========================================================== parallax.c
# Given true hour angle tha and declination tdec, the geographical latitude
# phi, the height above sea level as a fraction of the earth's radius ht,
# and the geocentric distance rho in earth radii, find the apparent hour
# angle and declination allowing for parallax.
def ta_par(
    tha: Float64, tdec: Float64, phi: Float64, ht: Float64, rho: Float64
) -> Tuple[Float64, Float64, Float64]:
    var e2 = (2 - 1.0 / 298.257) / 298.257
    var cphi = cos(phi)
    var sphi = sin(phi)
    var robs = 1 / sqrt(1 - e2 * sphi * sphi)

    # observer coordinates: x to meridian, y east, z north
    var xobs = (robs + ht) * cphi
    var zobs = (robs * (1 - e2) + ht) * sphi

    var x, y, z = sphcart(-tha, tdec, rho)
    var aha, adec, rho2 = cartsph(x - xobs, y, z - zobs)
    return range(-aha, 2.0 * 3.141592653589793), adec, rho2


# =========================================================== refract.c
def unrefractGE15(pr: Float64, tr: Float64, aa: Float64) -> Float64:
    var r = 7.888888e-5 * pr / ((273 + tr) * tan(aa))
    return aa - r


def unrefractLT15(pr: Float64, tr: Float64, aa: Float64) -> Float64:
    var aadeg = raddeg(aa)
    var a = ((2e-5 * aadeg + 1.96e-2) * aadeg + 1.594e-1) * pr
    var b = (273 + tr) * ((8.45e-2 * aadeg + 5.05e-1) * aadeg + 1)
    var r = degrad(a / b)
    # 0 below ~5 degs
    if aa < 0.0 and r < 0.0:
        return aa
    return aa - r


# Invert the refraction model: given an apparent altitude, find the true one.
def unrefract(pr: Float64, tr: Float64, aa: Float64) -> Float64:
    var ltim = 14.5
    var gelim = 15.5
    var aadeg = raddeg(aa)

    if aadeg < ltim:
        return unrefractLT15(pr, tr, aa)
    elif aadeg >= gelim:
        return unrefractGE15(pr, tr, aa)
    else:
        # smooth blend -- important for inverse
        var taLT = unrefractLT15(pr, tr, aa)
        var taGE = unrefractGE15(pr, tr, aa)
        var p = (aadeg - ltim) / (gelim - ltim)
        return taLT + (taGE - taLT) * p


# Correct the true altitude ta for refraction to the apparent altitude aa,
# each in radians, given the local pressure in mbar and temperature in C.
def refract(pr: Float64, tr: Float64, ta: Float64) -> Float64:
    var maxrerr = degrad(0.1 / 3600.0)  # desired accuracy, rads

    if ta != ta:  # isnan
        return ta

    # first guess of the error is to go backwards. delta-apparent is always
    # less than delta-true.
    var t = unrefract(pr, tr, ta)
    var d = 0.8 * (ta - t)
    var t0 = t
    var a = ta

    # use the secant method to discover a value that unrefracts to ta.
    # max=7 ave=2.4 loops in hundreds of test cases.
    while True:
        a += d
        t = unrefract(pr, tr, a)
        if abs(ta - t) <= maxrerr:
            break
        d *= -(ta - t) / (t0 - t)
        t0 = t

    return a


# ========================================================== circum.c
# Given geocentric ecliptic longitude lam and latitude bet of some object
# and the longitude of the sun lsn, find the elongation el. This is the
# actual angular separation of the object from the sun, not just the
# difference in longitude. The sign is set by a test on longitude, so el is
# > 0 for an evening object and < 0 for a morning object.
def elongation(lam: Float64, bet: Float64, lsn: Float64) -> Float64:
    var el = acos(cos(bet) * cos(lam - lsn))
    if lam > lsn + 3.141592653589793 or (lam > lsn - 3.141592653589793 and lam < lsn):
        el = -el
    return el


# Apply the relativistic light bending correction to ra/dec. From
# "Mean and apparent place computations in the new IAU system. III",
# Kaplan, Hughes, Seidelmann and Yallop, AJ 97, 1989, p. 1197-1210.
def deflect(
    mjd1: Float64,
    lpd: Float64,
    psi: Float64,
    rsn: Float64,
    lsn: Float64,
    rho: Float64,
    ra: Float64,
    dec: Float64,
    reg: Reg,
) -> Tuple[Float64, Float64]:
    # heliocentric elongation object--earth
    var el = abs(elongation(lpd, psi, lsn - 3.141592653589793))
    # only continue if the object is within about 10 deg around the sun, not
    # obscured by the sun's disc (radius 0.25 deg) and farther away than
    # the sun. Precise geocentric deflection is g1*tan(el/2) radially
    # outwards from the sun; the vector munging below applies this
    # component-wise. g1 is always about 0.004 arc seconds, g2 varies from
    # 0 (highest contribution) to 2.
    if el < degrad(170) or el > degrad(179.75) or rho < rsn:
        return ra, dec

    # get cartesian vectors
    var u = StaticTuple[Float64, 3]()
    var i = 0
    var x, y, z = sphcart(ra, dec, rho)
    u[0] = x
    u[1] = y
    u[2] = z

    var q = StaticTuple[Float64, 3]()
    var hra, hdec = ecl_eq(mjd1, psi, lpd)
    var qx, qy, qz = sphcart(hra, hdec, 1.0)
    q[0] = qx
    q[1] = qy
    q[2] = qz

    var e = StaticTuple[Float64, 3]()
    var era, edec = ecl_eq(mjd1, 0.0, lsn - 3.141592653589793)
    var ex, ey, ez = sphcart(era, edec, 1.0)
    e[0] = ex
    e[1] = ey
    e[2] = ez

    # evaluate scalar products
    var qe = 0.0
    var uq = 0.0
    var eu = 0.0
    while i < 3:
        qe += q[i] * e[i]
        uq += u[i] * q[i]
        eu += e[i] * u[i]

        i += 1
    var g = 1.32712438e20  # heliocentric grav const; m^3*s^-2
    var c = 299792458.0  # speed of light in m/s
    var g1 = 2 * g / (c * c * 1.4959787e11) / rsn
    var g2 = 1 + qe

    # now deflect the geocentric vector
    g1 /= g2
    i = 0
    while i < 3:
        u[i] += g1 * (uq * e[i] - eu * q[i])

        i += 1
    # back to spherical
    var ra2, dec2, _rho = cartsph(u[0], u[1], u[2])
    return ra2, dec2


# Fill equatoreal and horizontal fields. lam/bet/rho arrive as geocentric
# mean ecliptic and equinox of day.
#   ecl_eq   --> ra/dec  geocentric mean equatoreal EOD (via mean obliq)
#   deflect  --> ra/dec    relativistic deflection
#   nut_eq   --> ra/dec  geocentric true equatoreal EOD
#   ab_eq    --> ra/dec  geocentric apparent equatoreal EOD
#   ta_par   --> ra/dec  topocentric apparent equatoreal EOD
#   hadec_aa --> alt/az   topocentric horizontal
#   refract  --> alt/az   observed
# Returns (ra, dec, gaera, gaedec, astrora, astrodec, ha, alt, az, rho).
def cir_pos(
    mjed: Float64,
    bet: Float64,
    lam: Float64,
    rho: Float64,
    hlong: Float64,
    hlat: Float64,
    p: Int,
    mjd: Float64,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    lsn: Float64,
    rsn: Float64,
    reg: Reg,
) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64]:
    # The solar ecliptical longitude (lsn) and sun distance (rsn) at `mjed`
    # are supplied by the caller. Every caller already computed them at this
    # mjed to feed cir_pos's own aberration and deflection steps, so the
    # sunpos call is not repeated here.
    # convert to equatoreal [mean equator, with mean obliquity]
    var ra, dec = ecl_eq(mjed, bet, lam)
    # keep mean coordinates
    var tra = ra
    var tdec = dec

    # precess and save astrometric coordinates
    if mjed != epoch:
        tra, tdec = precess(mjed, epoch, tra, tdec)
    var astrora = tra
    var astrodec = tdec

    # allow for relativistic light bending near the sun (avoid calling
    # deflect() for the sun itself)
    if p != 8 and p != 9:
        ra, dec = deflect(mjed, hlong, hlat, rsn, lsn, rho, ra, dec, reg)

    # correct ra/dec to form geocentric apparent
    ra, dec = nut_eq(mjed, ra, dec, reg)
    if p != 9:
        ra, dec = ab_eq(mjed, lsn, ra, dec)
    var gaera = ra
    var gaedec = dec

    # find the parallax correction for equatoreal coords
    var lst = now_lst(mjd, lng, reg)
    var ha_in = hrrad(lst) - ra
    var rho_topo = rho * 1.4959787e11 / 6.37816e6  # convert to earth radii
    var ha_out, dec_out, rho_topo2 = ta_par(ha_in, dec, lat, elev, rho_topo)

    # transform into alt/az and apply refraction
    var alt, az = hadec_aa(lat, ha_out, dec_out)
    alt = refract(pressure, temp, alt)
    var ha = ha_out

    # Get parallax differences and apply to apparent or astrometric place
    # as needed. For the astrometric place, rotating the CORRECTIONS back
    # from the nutated equator to the mean equator will be neglected. This
    # is an effect of about 0.1" at moon distance. We currently don't have
    # an inverse nutation rotation.
    var rho_out = rho
    if topocentric != 0:
        # ra sign is opposite of ha
        ra += ha_in - ha_out
        dec += dec_out - dec
        rho_out = rho_topo2 * 6.37816e6 / 1.4959787e11  # topocentric distance in AU
    return (
        range(ra, 2.0 * 3.141592653589793),
        dec,
        gaera,
        gaedec,
        astrora,
        astrodec,
        ha,
        alt,
        az,
        rho_out,
    )


# Find the sun's circumstances now. Writes into fields 0..16 of `dest`.
def sun_cir(
    mj: Float64,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    reg: Reg,
    dest: Ptr,
):
    var mjed = mm_mjed(mj, reg)
    var lsn, rsn, bsn = sunpos(mjed, reg)  # sun's true coords, mean ecliptic

    dest.unsafe_offset(2)[] = 0.0  # s_sdist
    dest.unsafe_offset(4)[] = 0.0  # s_elong
    dest.unsafe_offset(5)[] = 100.0  # s_phase
    dest.unsafe_offset(6)[] = -26.8  # set_smag; TODO upstream too
    # geo- to helio- centric
    var dhlong = range(lsn - 3.141592653589793, 2.0 * 3.141592653589793)
    dest.unsafe_offset(0)[] = dhlong  # s_hlong
    dest.unsafe_offset(1)[] = -bsn  # s_hlat

    dest.unsafe_offset(6)[] = _set_smag(-26.8)  # set_smag(op, -26.8); TODO upstream too

    # fill the sun's ra/dec and alt/az. cir_pos takes the distance by
    # reference upstream and leaves the topocentric one in it.
    var res = cir_pos(
        mjed, bsn, lsn, rsn, dhlong, -bsn, 8, mj, lat, lng, elev, epoch,
        topocentric, pressure, temp, lsn, rsn, reg,
    )
    dest.unsafe_offset(8)[] = res[0]
    dest.unsafe_offset(9)[] = res[1]
    dest.unsafe_offset(10)[] = res[4]
    dest.unsafe_offset(11)[] = res[5]
    dest.unsafe_offset(12)[] = res[6]
    dest.unsafe_offset(13)[] = res[7]
    dest.unsafe_offset(14)[] = res[8]
    dest.unsafe_offset(3)[] = res[9]  # s_edist, topocentric when asked for
    dest.unsafe_offset(7)[] = raddeg(4.65242e-3 / res[9]) * 3600.0 * 2.0  # s_size


# Find the moon's circumstances now. Writes into fields 0..16 of `out`.
def moon_cir(
    mj: Float64,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    reg: Reg,
    dest: Ptr,
):
    var mjed = mm_mjed(mj, reg)
    # mean ecliptic & EOD
    var lam, bet, edistau, ms, md = moon(mjed, reg)
    var lsn, rsn, _bsn = sunpos(mjed, reg)  # mean ecliptic & EOD

    dest.unsafe_offset(0)[] = lam  # save geo in helio fields
    dest.unsafe_offset(1)[] = bet

    # find angular separation from the sun
    var el = elongation(lam, bet, lsn)
    dest.unsafe_offset(4)[] = raddeg(el)  # want degrees

    # solve triangle of earth, sun, and elongation for moon-sun dist
    var sdist = sqrt(edistau * edistau + rsn * rsn - 2.0 * edistau * rsn * cos(el))
    dest.unsafe_offset(2)[] = sdist

    # find phase -- allow for projection effects
    var i = 0.1468 * sin(el) * (1 - 0.0549 * sin(md)) / (1 - 0.0167 * sin(ms))
    var phase = (1 + cos(3.141592653589793 - el - degrad(i))) / 2.0 * 100.0
    dest.unsafe_offset(5)[] = phase

    # Moon's geocentric magnitude from Lane & Irvine 1973, V band.
    var pang = raddeg(acos(phase / 50.0 - 1.0))  # need 0-180 degrees
    var p: Float64
    var mag: Float64
    if pang <= 40.0:
        p = pang - 20.0
        mag = -12.72 + 0.0267 * p + 0.534
    else:
        p = pang - 80.0
        mag = -12.72 + p * (0.03188 + p * (1.9621e-4 + p * 1.7256e-6)) + 2.14

    # correct for earth-moon distance
    var mean_em_dist = 60.2665 * 6.37816e6 / 1.4959787e11
    mag += 5.0 * log10(edistau / mean_em_dist)
    # correct for moon-sun distance
    mag += 5.0 * log10(sdist)
    dest.unsafe_offset(6)[] = _set_smag(mag)

    # fill the moon's ra/dec, alt/az and update for topo dist
    var res = cir_pos(
        mjed, bet, lam, edistau, lam, bet, 9, mj, lat, lng, elev, epoch,
        topocentric, pressure, temp, lsn, rsn, reg,
    )
    dest.unsafe_offset(8)[] = res[0]
    dest.unsafe_offset(9)[] = res[1]
    dest.unsafe_offset(10)[] = res[4]
    dest.unsafe_offset(11)[] = res[5]
    dest.unsafe_offset(12)[] = res[6]
    dest.unsafe_offset(13)[] = res[7]
    dest.unsafe_offset(14)[] = res[8]
    dest.unsafe_offset(3)[] = res[9]  # s_edist
    # moon angular dia, seconds
    dest.unsafe_offset(7)[] = 3600.0 * 2.0 * raddeg(asin(1.740e6 / 1.4959787e11 / res[9]))


# Fill in all of the s_* fields except s_size and s_mag. Used for solar
# system objects (except the sun and moon); never FIXED.
def cir_sky(
    mj: Float64,
    lpd: Float64,
    psi: Float64,
    rp: Float64,
    rho: Float64,
    lam: Float64,
    bet: Float64,
    lsn: Float64,
    rsn: Float64,
    p: Int,
    mjd: Float64,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    reg: Reg,
    dest: Ptr,
):
    # compute elongation and phase
    var el = raddeg(elongation(lam, bet, lsn))
    dest.unsafe_offset(4)[] = el
    var f = 0.25 * ((rp + rho) * (rp + rho) - rsn * rsn) / (rp * rho)
    dest.unsafe_offset(5)[] = f * 100.0  # percent

    # set heliocentric long/lat; mean ecliptic and EOD
    dest.unsafe_offset(0)[] = lpd
    dest.unsafe_offset(1)[] = psi

    # fill the solar system body's ra/dec and alt/az (updates rho)
    var res = cir_pos(
        mj, bet, lam, rho, lpd, psi, p, mjd, lat, lng, elev, epoch,
        topocentric, pressure, temp, lsn, rsn, reg,
    )
    dest.unsafe_offset(8)[] = res[0]
    dest.unsafe_offset(9)[] = res[1]
    dest.unsafe_offset(10)[] = res[4]
    dest.unsafe_offset(11)[] = res[5]
    dest.unsafe_offset(12)[] = res[6]
    dest.unsafe_offset(13)[] = res[7]
    dest.unsafe_offset(14)[] = res[8]

    # set earth/planet and sun/planet distance
    dest.unsafe_offset(3)[] = res[9]
    dest.unsafe_offset(2)[] = rp


# Fill in the sky position (s_*) fields of a solar system object. `p` is a
# PLCode. `mjd` is the UTC-based date, `mjed` its dynamical-time twin.
def obj_planet(
    mj: Float64,
    p: Int,
    mjd: Float64,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    nrec: Int,
    reg: Reg,
    dest: Ptr,
):
    # The sun and the moon each run their own sunpos inside sun_cir/moon_cir,
    # so they are dispatched before mm_mjed/sunpos are computed here. Upstream
    # computes both and throws the results away; the values are provably dead
    # on these two paths, and sunpos is ~20% of a solar compute.
    if p == 8:
        sun_cir(
            mj, lat, lng, elev, epoch, topocentric, pressure, temp, reg, dest
        )
        return
    if p == 9:
        moon_cir(mj, lat, lng, elev, epoch, topocentric, pressure, temp, reg, dest)
        return

    var mjed = mm_mjed(mj, reg)

    # find solar ecliptical longitude and distance to sun from earth; `bsn` is
    # needed by the cartesian sun in plans, `lsn`/`rsn` by cir_sky and satrings
    var lsn, rsn, bsn = sunpos(mjed, reg)

    # find helio long/lat; sun/planet and earth/planet dist; ecliptic
    # long/lat; diameter and mag.
    var lpd, psi, rp, rho, lam, bet, dia, mag = plans(
        mjed, p, nrec, lsn, rsn, bsn, reg
    )

    cir_sky(
        mjed, lpd, psi, rp, rho, lam, bet, lsn, rsn, p, mjd, lat, lng, elev,
        epoch, topocentric, pressure, temp, reg, dest,
    )

    # set magnitude and angular size
    dest.unsafe_offset(6)[] = _set_smag(mag)
    dest.unsafe_offset(7)[] = dia / dest.unsafe_offset(3)[]  # cir_sky left the working distance in dest[3]


def _set_smag(m: Float64) -> Float64:
    """Upstream's set_smag: the visual magnitude lives in a short of
    hundredths, so the reported value is quantised to 0.01."""
    return floor(m * 100.0 + 0.5) / 100.0


def chap_nrec(p: Int) -> Int:
    # Record counts of the checked-in chap95 tables. They are a property of
    # data/tables.bin, not a second copy of the coefficients, so they live
    # here beside the C ABI rather than in the registry.
    if p == 3:
        return 216
    elif p == 4:
        return 192
    elif p == 5:
        return 129
    elif p == 6:
        return 117
    else:
        return 98


# ================================================================ C ABI
# Every buffer crosses as an Int address, because @export rejects
# parametric functions and a pointer with an inferred origin is
# parametric. Callers own all memory, including scratch, so nothing here
# allocates and nothing here can leak. `reg` is the table registry from
# mojoephem._tables; `nrec` is the chap95 record count for `p`, which the
# registry does not carry.
#
# chap95 record counts: jupiter 216, saturn 192, uranus 129, neptune 117,
# pluto 98. They are constants of the checked-in data blob, not a second
# copy of the tables.


def regp(reg: Int) -> Reg:
    return Reg(unsafe_from_address=reg)


def fp(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


@export("mpe_range")
def mpe_range(v: Int, r: Float64) abi("C"):
    var p = fp(v)
    p.unsafe_offset(0)[] = range(p.unsafe_offset(0)[], r)


@export("mpe_radecrange")
def mpe_radecrange(ra: Int, dec: Int) abi("C"):
    var p = fp(ra)
    var d = fp(dec)
    var r, dd = radecrange(p.unsafe_offset(0)[], d.unsafe_offset(0)[])
    p.unsafe_offset(0)[] = r
    d.unsafe_offset(0)[] = dd


@export("mpe_mjd_day")
def mpe_mjd_day(mj: Float64) abi("C") -> Float64:
    return mjd_day(mj)


@export("mpe_mjd_hr")
def mpe_mjd_hr(mj: Float64) abi("C") -> Float64:
    return mjd_hr(mj)


@export("mpe_cal_mjd")
def mpe_cal_mjd(mn: Int, dy: Float64, yr: Int) abi("C") -> Float64:
    return cal_mjd(mn, dy, yr)


@export("mpe_mjd_cal")
def mpe_mjd_cal(mj: Float64, mn: Int, dy: Int, yr: Int) abi("C"):
    var a, b, c = mjd_cal(mj)
    Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=mn).unsafe_offset(0)[] = a
    fp(dy).unsafe_offset(0)[] = b
    Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=yr).unsafe_offset(0)[] = c


@export("mpe_year_mjd")
def mpe_year_mjd(y: Float64) abi("C") -> Float64:
    return year_mjd(y)


@export("mpe_mjd_year")
def mpe_mjd_year(mj: Float64) abi("C") -> Float64:
    return mjd_year(mj)


@export("mpe_isleapyear")
def mpe_isleapyear(y: Int) abi("C") -> Int:
    return isleapyear(y)


@export("mpe_mjd_dow")
def mpe_mjd_dow(mj: Float64) abi("C") -> Int:
    return mjd_dow(mj)


@export("mpe_deltat")
def mpe_deltat(mj: Float64, reg: Int) abi("C") -> Float64:
    return deltat(mj, regp(reg))


@export("mpe_mm_mjed")
def mpe_mm_mjed(mjd: Float64, reg: Int) abi("C") -> Float64:
    return mm_mjed(mjd, regp(reg))


@export("mpe_sphcart")
def mpe_sphcart(l: Float64, b: Float64, r: Float64, dst: Int) abi("C"):
    var x, y, z = sphcart(l, b, r)
    var p = fp(dst)
    p.unsafe_offset(0)[] = x
    p.unsafe_offset(1)[] = y
    p.unsafe_offset(2)[] = z


@export("mpe_cartsph")
def mpe_cartsph(x: Float64, y: Float64, z: Float64, dst: Int) abi("C"):
    var l, b, r = cartsph(x, y, z)
    var p = fp(dst)
    p.unsafe_offset(0)[] = l
    p.unsafe_offset(1)[] = b
    p.unsafe_offset(2)[] = r


@export("mpe_obliquity")
def mpe_obliquity(mj: Float64, dst: Int) abi("C"):
    fp(dst).unsafe_offset(0)[] = obliquity(mj)


@export("mpe_reduce_elements")
def mpe_reduce_elements(
    mj0: Float64, mj: Float64, inc0: Float64, ap0: Float64, om0: Float64, dst: Int
) abi("C"):
    var inc, ap, om = reduce_elements(mj0, mj, inc0, ap0, om0)
    var p = fp(dst)
    p.unsafe_offset(0)[] = inc
    p.unsafe_offset(1)[] = ap
    p.unsafe_offset(2)[] = om


@export("mpe_anomaly")
def mpe_anomaly(ma: Float64, s: Float64, dst: Int) abi("C"):
    var nu, ea = anomaly(ma, s)
    var p = fp(dst)
    p.unsafe_offset(0)[] = nu
    p.unsafe_offset(1)[] = ea


@export("mpe_precess")
def mpe_precess(mjd1: Float64, mjd2: Float64, radec: Int) abi("C"):
    var p = fp(radec)
    var r, dd = precess(mjd1, mjd2, p.unsafe_offset(0)[], p.unsafe_offset(1)[])
    p.unsafe_offset(0)[] = r
    p.unsafe_offset(1)[] = dd


@export("mpe_eq_ecl")
def mpe_eq_ecl(mj: Float64, ra: Float64, dec: Float64, dst: Int) abi("C"):
    var lt, lg = eq_ecl(mj, ra, dec)
    var p = fp(dst)
    p.unsafe_offset(0)[] = lt
    p.unsafe_offset(1)[] = lg


@export("mpe_ecl_eq")
def mpe_ecl_eq(mj: Float64, lt: Float64, lg: Float64, dst: Int) abi("C"):
    var ra, dec = ecl_eq(mj, lt, lg)
    var p = fp(dst)
    p.unsafe_offset(0)[] = ra
    p.unsafe_offset(1)[] = dec


@export("mpe_nutation")
def mpe_nutation(mj: Float64, reg: Int, dst: Int) abi("C"):
    var deps, dpsi = nutation(mj, regp(reg))
    var p = fp(dst)
    p.unsafe_offset(0)[] = deps
    p.unsafe_offset(1)[] = dpsi


@export("mpe_nut_eq")
def mpe_nut_eq(mj: Float64, reg: Int, radec: Int) abi("C"):
    var p = fp(radec)
    var r, dd = nut_eq(mj, p.unsafe_offset(0)[], p.unsafe_offset(1)[], regp(reg))
    p.unsafe_offset(0)[] = r
    p.unsafe_offset(1)[] = dd


@export("mpe_ab_ecl")
def mpe_ab_ecl(mj: Float64, lsn: Float64, lambet: Int) abi("C"):
    var p = fp(lambet)
    var a, b = ab_ecl(mj, lsn, p.unsafe_offset(0)[], p.unsafe_offset(1)[])
    p.unsafe_offset(0)[] = a
    p.unsafe_offset(1)[] = b


@export("mpe_ab_eq")
def mpe_ab_eq(mj: Float64, lsn: Float64, radec: Int) abi("C"):
    var p = fp(radec)
    var r, dd = ab_eq(mj, lsn, p.unsafe_offset(0)[], p.unsafe_offset(1)[])
    p.unsafe_offset(0)[] = r
    p.unsafe_offset(1)[] = dd


@export("mpe_vsop87")
def mpe_vsop87(mj: Float64, obj: Int, prec: Float64, reg: Int, dst: Int) abi("C"):
    var l, b, r = vsop87(mj, obj, prec, regp(reg))
    var p = fp(dst)
    p.unsafe_offset(0)[] = l
    p.unsafe_offset(1)[] = b
    p.unsafe_offset(2)[] = r


@export("mpe_chap95")
def mpe_chap95(mj: Float64, obj: Int, prec: Float64, reg: Int, dst: Int) abi("C"):
    var x, y, z = chap95(mj, obj, prec, chap_nrec(obj), regp(reg))
    var p = fp(dst)
    p.unsafe_offset(0)[] = x
    p.unsafe_offset(1)[] = y
    p.unsafe_offset(2)[] = z


@export("mpe_sunpos")
def mpe_sunpos(mj: Float64, reg: Int, dst: Int) abi("C"):
    var lsn, rsn, bsn = sunpos(mj, regp(reg))
    var p = fp(dst)
    p.unsafe_offset(0)[] = lsn
    p.unsafe_offset(1)[] = rsn
    p.unsafe_offset(2)[] = bsn


@export("mpe_pluto_ell")
def mpe_pluto_ell(mj: Float64, dst: Int) abi("C"):
    var l, b, r = pluto_ell(mj)
    var p = fp(dst)
    p.unsafe_offset(0)[] = l
    p.unsafe_offset(1)[] = b
    p.unsafe_offset(2)[] = r


@export("mpe_planpos")
def mpe_planpos(mj: Float64, obj: Int, prec: Float64, reg: Int, dst: Int) abi("C"):
    var l, b, r = planpos(mj, obj, prec, chap_nrec(obj), regp(reg))
    var p = fp(dst)
    p.unsafe_offset(0)[] = l
    p.unsafe_offset(1)[] = b
    p.unsafe_offset(2)[] = r


@export("mpe_satrings")
def mpe_satrings(
    sb: Float64, sl: Float64, sr: Float64, el: Float64, er: Float64, jd: Float64, dst: Int
) abi("C"):
    var et, st = satrings(sb, sl, sr, el, er, jd)
    var p = fp(dst)
    p.unsafe_offset(0)[] = et
    p.unsafe_offset(1)[] = st


@export("mpe_plans")
def mpe_plans(mj: Float64, p: Int, reg: Int, dst: Int) abi("C"):
    # plans() takes the solar geometry from its caller, so the standalone
    # entry point computes it here. obj_planet passes its own sunpos in.
    var lsn, rsn, bsn = sunpos(mj, regp(reg))
    var a, b, c, d, e, f, g, h = plans(
        mj, p, chap_nrec(p), lsn, rsn, bsn, regp(reg)
    )
    var o = fp(dst)
    o.unsafe_offset(0)[] = a
    o.unsafe_offset(1)[] = b
    o.unsafe_offset(2)[] = c
    o.unsafe_offset(3)[] = d
    o.unsafe_offset(4)[] = e
    o.unsafe_offset(5)[] = f
    o.unsafe_offset(6)[] = g
    o.unsafe_offset(7)[] = h


@export("mpe_moon")
def mpe_moon(mj: Float64, reg: Int, dst: Int) abi("C"):
    var lam, bet, rho, msp, mdp = moon(mj, regp(reg))
    var p = fp(dst)
    p.unsafe_offset(0)[] = lam
    p.unsafe_offset(1)[] = bet
    p.unsafe_offset(2)[] = rho
    p.unsafe_offset(3)[] = msp
    p.unsafe_offset(4)[] = mdp


@export("mpe_solve_sphere")
def mpe_solve_sphere(A: Float64, b: Float64, cc: Float64, sc: Float64, dst: Int) abi("C"):
    var cap, B = solve_sphere(A, b, cc, sc)
    var p = fp(dst)
    p.unsafe_offset(0)[] = cap
    p.unsafe_offset(1)[] = B


@export("mpe_aa_hadec")
def mpe_aa_hadec(lt: Float64, alt: Float64, az: Float64, dst: Int) abi("C"):
    var ha, dec = aa_hadec(lt, alt, az)
    var p = fp(dst)
    p.unsafe_offset(0)[] = ha
    p.unsafe_offset(1)[] = dec


@export("mpe_hadec_aa")
def mpe_hadec_aa(lt: Float64, ha: Float64, dec: Float64, dst: Int) abi("C"):
    var alt, az = hadec_aa(lt, ha, dec)
    var p = fp(dst)
    p.unsafe_offset(0)[] = alt
    p.unsafe_offset(1)[] = az


@export("mpe_geoc_lat")
def mpe_geoc_lat(phi: Float64) abi("C") -> Float64:
    return geoc_lat(phi)


@export("mpe_ta_par")
def mpe_ta_par(
    tha: Float64, tdec: Float64, phi: Float64, ht: Float64, rho: Int, dst: Int
) abi("C"):
    var p = fp(rho)
    var aha, adec, r2 = ta_par(tha, tdec, phi, ht, p.unsafe_offset(0)[])
    p.unsafe_offset(0)[] = r2
    var o = fp(dst)
    o.unsafe_offset(0)[] = aha
    o.unsafe_offset(1)[] = adec


@export("mpe_refract")
def mpe_refract(pr: Float64, tr: Float64, ta: Float64, dst: Int) abi("C"):
    fp(dst).unsafe_offset(0)[] = refract(pr, tr, ta)


@export("mpe_unrefract")
def mpe_unrefract(pr: Float64, tr: Float64, aa: Float64, dst: Int) abi("C"):
    fp(dst).unsafe_offset(0)[] = unrefract(pr, tr, aa)


@export("mpe_utc_gst")
def mpe_utc_gst(mj: Float64, utc: Float64, dst: Int) abi("C"):
    fp(dst).unsafe_offset(0)[] = utc_gst(mj, utc)


@export("mpe_gst_utc")
def mpe_gst_utc(mj: Float64, gst: Float64, dst: Int) abi("C"):
    fp(dst).unsafe_offset(0)[] = gst_utc(mj, gst)


@export("mpe_gmst0")
def mpe_gmst0(mj: Float64) abi("C") -> Float64:
    return gmst0(mj)


@export("mpe_now_lst")
def mpe_now_lst(mjd: Float64, lng: Float64, reg: Int, dst: Int) abi("C"):
    fp(dst).unsafe_offset(0)[] = now_lst(mjd, lng, regp(reg))


@export("mpe_radec2ha")
def mpe_radec2ha(lst: Float64, ra: Float64) abi("C") -> Float64:
    return radec2ha(lst, ra)


@export("mpe_elongation")
def mpe_elongation(lam: Float64, bet: Float64, lsn: Float64, dst: Int) abi("C"):
    fp(dst).unsafe_offset(0)[] = elongation(lam, bet, lsn)


@export("mpe_deflect")
def mpe_deflect(
    mjd1: Float64,
    lpd: Float64,
    psi: Float64,
    rsn: Float64,
    lsn: Float64,
    rho: Float64,
    radec: Int,
    reg: Int,
) abi("C"):
    var p = fp(radec)
    var r, dd = deflect(mjd1, lpd, psi, rsn, lsn, rho, p.unsafe_offset(0)[], p.unsafe_offset(1)[], regp(reg))
    p.unsafe_offset(0)[] = r
    p.unsafe_offset(1)[] = dd


# obj_planet for one date. `dst` is a 17-element float64 scratch the caller
# owns - the Obj field vector plus the mj that produced it - and this wrapper
# clears all 17 before filling them. The field order is documented in
# python/mojoephem/__init__.py next to the _F_* indices.
@export("mpe_obj_planet")
def mpe_obj_planet(
    mj: Float64,
    p: Int,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    reg: Int,
    dst: Int,
) abi("C"):
    var o = fp(dst)
    var i = 0
    while i < 17:  # the Obj field vector, including the mj that produced it
        o.unsafe_offset(i)[] = 0.0
        i += 1
    o.unsafe_offset(16)[] = mj
    obj_planet(
        mj, p, mj, lat, lng, elev, epoch, topocentric, pressure, temp,
        chap_nrec(p), regp(reg), o,
    )


# The Obj field vector width, including the mj that produced it.
comptime SERIES_WIDTH: Int = 17

# One obj_planet call per date, and nothing is shared between calls - the
# `static` memo slots upstream relied on are gone - so the epoch loop is
# embarrassingly parallel. Below this many dates the worker launch costs
# more than the work it hands out.
comptime SERIES_PARALLEL_MIN: Int = 256

# The pool is sized to the work, not just to the core count: below this many
# dates per worker the launch costs more than the chunk it hands out, which is
# the common case just over SERIES_PARALLEL_MIN.
comptime SERIES_MIN_DATES_PER_WORKER: Int = 64

# Ceiling on workers. This is a throughput/latency knob, not a correctness
# one - the serial path and the worker-split path compute the same numbers.
comptime SERIES_MAX_WORKERS: Int = 36


# MAX sizes its worker pool from the logical CPU count and spawns the whole
# pool eagerly on the first parallelize() call. A host that caps threads per
# process (containers, CI runners) then aborts the caller's process with
# `LLVM ERROR: pthread_create failed`, which no `try:` can intercept - the port
# would take the host down rather than lose the speedup. `budget` is the
# caller's measured thread headroom (see _worker_budget in the Python layer);
# a value below 2 means "do not launch", so an over-subscribed host silently
# takes the serial loop below instead of dying.


def _series_one(
    src: Ptr,
    dest: Ptr,
    i: Int,
    p: Int,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    nrec: Int,
    r: Reg,
):
    var mj = src.unsafe_load(i)
    var o = dest.unsafe_offset(i * SERIES_WIDTH)
    var k = 0
    while k < SERIES_WIDTH:
        o.unsafe_offset(k)[] = 0.0
        k += 1
    o.unsafe_offset(SERIES_WIDTH - 1)[] = mj
    obj_planet(
        mj, p, mj, lat, lng, elev, epoch, topocentric, pressure, temp,
        nrec, r, o,
    )


# Worker `w` takes a contiguous run of dates, so each core writes a
# contiguous run of the output and the two buffers stay streaming-friendly.
def _series_chunk(
    src: Ptr,
    dest: Ptr,
    w: Int,
    n: Int,
    nw: Int,
    p: Int,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    nrec: Int,
    r: Reg,
):
    var chunk = (n + nw - 1) // nw
    var lo = w * chunk
    var hi = min(lo + chunk, n)
    var i = lo
    while i < hi:
        _series_one(
            src, dest, i, p, lat, lng, elev, epoch, topocentric, pressure,
            temp, nrec, r,
        )
        i += 1


# Straight-line elementwise map over dates, the same call repeated. This is
# the batch entry point the Python layer's `compute_series` drives.
@export("mpe_obj_planet_series")
def mpe_obj_planet_series(
    mjs: Int,
    dst: Int,
    n: Int,
    p: Int,
    lat: Float64,
    lng: Float64,
    elev: Float64,
    epoch: Float64,
    topocentric: Int,
    pressure: Float64,
    temp: Float64,
    reg: Int,
    budget: Int,
) abi("C"):
    var src = fp(mjs)
    var nrec = chap_nrec(p)
    var r = regp(reg)
    var dest = fp(dst)
    var nw = num_physical_cores()
    if nw > budget:
        nw = budget
    if nw > SERIES_MAX_WORKERS:
        nw = SERIES_MAX_WORKERS
    var by_work = n // SERIES_MIN_DATES_PER_WORKER
    if nw > by_work:
        nw = by_work
    if nw < 1:
        nw = 1
    if n >= SERIES_PARALLEL_MIN and nw > 1:
        var par = lambda (w: Int) -> None: _series_chunk(
            src, dest, w, n, nw, p, lat, lng, elev, epoch, topocentric,
            pressure, temp, nrec, r,
        )
        parallelize(par, nw, nw)
        return
    var i = 0
    while i < n:
        _series_one(
            src, dest, i, p, lat, lng, elev, epoch, topocentric, pressure,
            temp, nrec, r,
        )
        i += 1
