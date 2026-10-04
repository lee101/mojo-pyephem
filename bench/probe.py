"""Per-call cost decomposition for the scalar paths. Not the benchmark.

Run it the same way as the benchmark, through pixi:

    pixi run python bench/probe.py

Every scratch buffer here is sized to what its kernel actually writes - 17
slots for `mpe_obj_planet`, 2 for `mpe_precess`, and so on. Under-sizing one
corrupts the heap, and the resulting Bus error surfaces inside the dynamic
loader on the *next* dlopen rather than at the call, which is very hard to
attribute back to the buffer that was too small.
"""

import ctypes
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import ephem  # noqa: E402  upstream
import numpy as np  # noqa: E402
import mojoephem as msk  # noqa: E402
from mojoephem import _lib, _tables  # noqa: E402

lib = _lib.lib()
F = ctypes.c_double
I = ctypes.c_int64
P = ctypes.c_void_p
reg = _tables.reg()


def t(fn, n=200000, repeat=3):
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(n)
        best = min(best, time.perf_counter() - t0)
    return best / n * 1e6


# mpe_obj_planet writes the whole 17-slot Obj vector; the smaller kernels
# write 2 or 3, so each gets its own correctly sized buffer.
obj_buf = (F * 17)()
obj_addr = ctypes.addressof(obj_buf)
out = (F * 8)()
out_addr = ctypes.addressof(out)
pair = (F * 2)()
pair_addr = ctypes.addressof(pair)

deltat = getattr(lib, "mpe_deltat")
precess = getattr(lib, "mpe_precess")
isleap = getattr(lib, "mpe_isleapyear")
vsop = getattr(lib, "mpe_vsop87")
nut = getattr(lib, "mpe_nutation")
sunpos_f = getattr(lib, "mpe_sunpos")
obj = getattr(lib, "mpe_obj_planet")

d = np.linspace(float(msk.Date("2023/1/1")), float(msk.Date("2023/1/1")) + 200000, 200000)
dl = d.tolist()
mj = float(msk.Date("2023/6/1"))
mj2 = 45290.5

print(f"{'ctypes floor isleapyear':38s} {t(lambda n: [isleap(2000) for _ in range(n)]):7.3f} us")
print(f"{'raw mpe_deltat':38s} {t(lambda n: [deltat(mj, reg) for _ in range(n)]):7.3f} us")
print(f"{'msk.delta_t (wrapper)':38s} {t(lambda n: [msk.delta_t(mj) for _ in range(n)]):7.3f} us")
# mpe_precess takes (mjd1, mjd2, radec) and works in place on the buffer.
print(f"{'raw mpe_precess':38s} {t(lambda n: [precess(mj2, mj2, pair_addr) for _ in range(n)]):7.3f} us")
print(f"{'msk.precess (wrapper)':38s} {t(lambda n: [msk.precess(msk.J2000, mj2, 0.5, 0.2) for _ in range(n)]):7.3f} us")
print(f"{'raw mpe_nutation':38s} {t(lambda n: [nut(mj, reg, out_addr) for _ in range(n)]):7.3f} us")
print(f"{'raw mpe_vsop87 earth':38s} {t(lambda n: [vsop(mj, 8, 0.0, reg, out_addr) for _ in range(n)]):7.3f} us")
print(f"{'raw mpe_sunpos':38s} {t(lambda n: [sunpos_f(mj, reg, out_addr) for _ in range(n)]):7.3f} us")
print(f"{'raw mpe_obj_planet Sun':38s} {t(lambda n: [obj(mj, 8, 0., 0., 0., msk.J2000, 0, 0., 15., reg, obj_addr) for _ in range(n)], 20000):7.3f} us")
print(f"{'raw mpe_obj_planet Saturn':38s} {t(lambda n: [obj(mj, 4, 0., 0., 0., msk.J2000, 0, 0., 15., reg, obj_addr) for _ in range(n)], 20000):7.3f} us")
print(f"{'msk.separation':38s} {t(lambda n: [msk.separation((0.5, 0.2), (0.6, 0.2)) for _ in range(n)]):7.3f} us")
print(f"{'ephem.separation':38s} {t(lambda n: [ephem.separation((0.5, 0.2), (0.6, 0.2)) for _ in range(n)]):7.3f} us")
print(f"{'ephem.delta_t':38s} {t(lambda n: [ephem.delta_t(ephem.Date(mj)) for _ in range(n)]):7.3f} us")
libp = ephem._libastro
print(f"{'ephem._libastro.precess':38s} {t(lambda n: [libp.precess(ephem.J2000, 45290.5, 0.5, 0.2) for _ in range(n)]):7.3f} us")

pairs = list(zip(np.linspace(0, 6.28, 200000), np.linspace(-1.5, 1.5, 200000)))
print(f"{'msk.precess np pairs':38s} {t(lambda n: [msk.precess(msk.J2000, mj2, a, b) for a, b in pairs[:n]]):7.3f} us")
print(f"{'msk.separation np pairs':38s} {t(lambda n: [msk.separation((a, b), (a + 0.1, b)) for a, b in pairs[:n]]):7.3f} us")

sun = msk.Sun()
print(f"{'Sun.compute+scalar read':38s} {t(lambda n: [sun.compute(msk.Date(x)) or (float(sun.a_ra) + float(sun.earth_distance)) for x in dl[:n]], 20000):7.3f} us")
sunt = ephem.Sun()
dtl = [ephem.Date(x) for x in dl[:20000]]
print(f"{'ephem Sun.compute+read':38s} {t(lambda n: [sunt.compute(x) or (float(sunt.a_ra) + float(sunt.earth_distance)) for x in dtl[:n]], 20000):7.3f} us")
