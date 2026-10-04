"""mojoutm against upstream `utm` on the same data.

Run it with `pixi run bench`, which takes a machine-wide lock first; running
this file directly will give you numbers distorted by whatever else is building.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojoutm  # noqa: E402
import utm  # noqa: E402

REPEAT = 5


def timeit(fn, repeat: int = REPEAT) -> float:
    best = math.inf
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def utm_points(n, rng):
    """Easting and northing spread over the middle of zone 31, which is what
    `to_latlon` consumes."""
    rng = np.random.default_rng(1)
    easting = rng.uniform(150_000.0, 850_000.0, n)
    northing = rng.uniform(100_000.0, 9_300_000.0, n)
    return easting, northing


def points(n, rng):
    """One zone, all north, so every case exercises the same kernel path."""
    lat = np.round(rng.uniform(0.001, 84.0, n), 6)
    lon = np.round(rng.uniform(-0.5, 0.5, n), 6)
    return lat, lon


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# A single scalar call takes well under a microsecond, so timing one call per
# repeat measures the clock, not the code. Repeat it enough times that the
# timer resolution is a rounding error against the total.
SCALAR_CALLS = 2_000


@case("from_latlon, scalar (per call)")
def _():
    rng = np.random.default_rng(0)
    lat, lon = points(SCALAR_CALLS, rng)
    la = [float(v) for v in lat]
    lo = [float(v) for v in lon]

    def mine():
        for a, b in zip(la, lo):
            mojoutm.from_latlon(a, b)

    def theirs():
        for a, b in zip(la, lo):
            utm.from_latlon(a, b)

    return (mine, theirs)


@case("from_latlon, 1k points")
def _():
    lat, lon = points(1_000, np.random.default_rng(0))
    return (lambda: mojoutm.from_latlon(lat, lon),
            lambda: utm.from_latlon(lat, lon))


@case("from_latlon, 100k points")
def _():
    lat, lon = points(100_000, np.random.default_rng(0))
    return (lambda: mojoutm.from_latlon(lat, lon),
            lambda: utm.from_latlon(lat, lon))


@case("from_latlon, 1M points")
def _():
    lat, lon = points(1_000_000, np.random.default_rng(0))
    return (lambda: mojoutm.from_latlon(lat, lon),
            lambda: utm.from_latlon(lat, lon))


@case("to_latlon, 1k points")
def _():
    e, n = utm_points(1_000, np.random.default_rng(0))
    return (lambda: mojoutm.to_latlon(e, n, 31, zone_letter="V"),
            lambda: utm.to_latlon(e, n, 31, zone_letter="V"))


@case("to_latlon, 100k points")
def _():
    e, n = utm_points(100_000, np.random.default_rng(0))
    return (lambda: mojoutm.to_latlon(e, n, 31, zone_letter="V"),
            lambda: utm.to_latlon(e, n, 31, zone_letter="V"))


@case("to_latlon, 1M points")
def _():
    e, n = utm_points(1_000_000, np.random.default_rng(0))
    return (lambda: mojoutm.to_latlon(e, n, 31, zone_letter="V"),
            lambda: utm.to_latlon(e, n, 31, zone_letter="V"))


@case("round trip (from + to), 100k points")
def _():
    lat, lon = points(100_000, np.random.default_rng(0))
    e, n, zn, zl = mojoutm.from_latlon(lat, lon)

    def mine():
        a, b, z, c = mojoutm.from_latlon(lat, lon)
        return mojoutm.to_latlon(a, b, z, zone_letter=c)

    def theirs():
        a, b, z, c = utm.from_latlon(lat, lon)
        return utm.to_latlon(a, b, z, zone_letter=c)

    assert abs(mojoutm.to_latlon(e, n, zn, zone_letter=zl)[0][0] - lat[0]) < 1e-5
    return (mine, theirs)


@case("latlon_to_zone_number, 1M points (per-point API)")
def _():
    lat, lon = points(1_000_000, np.random.default_rng(0))
    return (lambda: [mojoutm.latlon_to_zone_number(float(a), float(b)) for a, b in zip(lat, lon)],
            lambda: [utm.latlon_to_zone_number(float(a), float(b)) for a, b in zip(lat, lon)])


def main() -> None:
    print("# mojo-utm benchmark")
    print()
    print(f"machine: {platform.processor() or platform.machine()}, "
          f"{os.cpu_count()} logical cpus, "
          f"{platform.system()} {platform.release()}")
    print(f"python {platform.python_version()}, numpy {np.__version__}, "
          f"upstream utm {getattr(utm, '__version__', '0.9.0')}")
    print(f"timing: best of {REPEAT} runs per case")
    print()
    print("| case | mojo-utm | upstream utm | speedup |")
    print("| --- | ---: | ---: | ---: |")
    rows = []
    for name, fn in CASES:
        mine, theirs = fn()
        # `from_latlon, scalar` runs SCALAR_CALLS conversions per repeat, so
        # its row is reported per call like its name says.
        per_call = SCALAR_CALLS if "scalar" in name else 1
        a = timeit(mine) / per_call
        b = timeit(theirs) / per_call
        ratio = b / a if a > 0 else float("inf")
        rows.append((name, a, b, ratio))
        unit = "us" if per_call > 1 else "ms"
        scale = 1e6 if per_call > 1 else 1e3
        print(f"| {name} | {a * scale:.2f} {unit} | {b * scale:.2f} {unit} | {ratio:.2f}x |")
    print()
    worst = min(rows, key=lambda r: r[3])
    best = max(rows, key=lambda r: r[3])
    print(f"fastest: {best[0]} at {best[3]:.2f}x")
    print(f"slowest: {worst[0]} at {worst[3]:.2f}x")


if __name__ == "__main__":
    main()
