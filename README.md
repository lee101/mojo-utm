# mojo-utm

The [utm](https://github.com/Turbo87/utm) package's bidirectional WGS84 to UTM
conversion, implemented in [Mojo](https://www.modular.com/mojo) and callable from
Python with the same API.

```python
import mojoutm

mojoutm.from_latlon(51.2, 7.5)
# (395201.31038112973, 5673135.241182375, 32, 'U')
mojoutm.to_latlon(395201.31038112973, 5673135.241182375, 32, 'U')
# (51.2000000029224, 7.499999999862897)
```

`to_latlon`, `from_latlon`, `latlon_to_zone_number`, `latitude_to_zone_letter`,
`check_valid_zone`, `zone_number_to_central_longitude`,
`zone_letter_to_central_latitude` and `OutOfRangeError` all keep upstream's
names, argument order, defaults and return types, so changing the import is
usually the whole migration. The numerics live in `src/ported.mojo` and read
top to bottom next to `utm/conversion.py`, in upstream's own source order.

Every test in `tests/` compares against the real `utm` package on the same data.
Both entry points are bit-identical to upstream - `from_latlon` over 1,014,690
random points in both hemispheres, `to_latlon` over 744,032 points across all 60
zones and 20 bands, plus 360,000 more past the pole. See
[Divergences](#divergences) for the two places the shapes had to differ and what
they cost.

## Install

```bash
pixi install     # brings its own Mojo toolchain and builds nothing yet
pixi run build   # src/capi.mojo -> dist/libmojo-utm.so
pixi run test
pixi run bench
```

The shared library builds itself on first import and rebuilds whenever a
`.mojo` file is newer than it. To force it:

```bash
python -m mojoutm._lib --force
```

To install it as a package instead, build the library in a checkout first and
point the installed copy at it, because an installed copy has no `src/` to
compile from:

```bash
pip install .
MOJO_UTM_LIB=$PWD/build/capi.so python -c "import mojoutm; print(mojoutm.from_latlon(51.2, 7.5))"
```

Outside pixi, point the build at a compiler with `MOJO_UTM_MOJO=/path/to/mojo`.
To skip the build entirely, set `MOJO_UTM_LIB` to an already-built `capi.so`.
Note that a bare `mojo` outside pixi also needs `MODULAR_HOME` pointing at the
env's `share/max`, or every import fails with `unable to locate module 'std'`.

## Usage

Scalars give scalars, NumPy arrays give NumPy arrays, exactly as upstream:

```python
import numpy as np
import mojoutm

# three points, deliberately all inside UTM zone 32
lat = np.array([51.2, 49.0, 47.4])
lon = np.array([7.5, 8.4, 6.1])

easting, northing, zone_number, zone_letter = mojoutm.from_latlon(lat, lon)
# easting, northing: shape (3,), zone_number and zone_letter: scalars taken
# from the first point, as upstream documents. Points outside that zone would
# need force_zone_number or their own zone.

back = mojoutm.to_latlon(easting, northing, zone_number, zone_letter=zone_letter)
# back[0] ~= lat, back[1] ~= lon

mojoutm.from_latlon(lat, lon, force_northern=True)   # skip the hemisphere guess
mojoutm.from_latlon(lat, lon, force_zone_number=32)  # skip the zone guess
mojoutm.to_latlon(easting, northing, 32, northern=True, strict=False)

mojoutm.latlon_to_zone_number(60.0, 5.0)          # 32, the Norway exception
mojoutm.latitude_to_zone_letter(51.2)            # 'U'
mojoutm.zone_number_to_central_longitude(32)     # 9
mojoutm.zone_letter_to_central_latitude('X')     # 78

mojoutm.to_latlon(0.0, 0.0, 33, northern=True)
# mojoutm.error.OutOfRangeError: easting out of range ...
```

## Performance

Against upstream `utm` 0.9.0 on the same machine and the same arrays. Upstream is
numpy ufuncs, which are single-threaded; this is a scalar Mojo loop per element
with no temporary arrays, split across cores when the array is big enough.

| case | mojo-utm | upstream utm | speedup |
| --- | ---: | ---: | ---: |
| `from_latlon`, scalar (per call) | 24.21 us | 57.91 us | 2.39x |
| `from_latlon`, 1k points | 0.19 ms | 0.49 ms | 2.64x |
| `from_latlon`, 100k points | 4.00 ms | 45.42 ms | 11.36x |
| `from_latlon`, 1M points | 17.34 ms | 504.22 ms | 29.08x |
| `to_latlon`, 1k points | 0.25 ms | 0.60 ms | 2.37x |
| `to_latlon`, 100k points | 3.04 ms | 31.18 ms | 10.26x |
| `to_latlon`, 1M points | 22.66 ms | 555.27 ms | 24.51x |
| round trip (`from` + `to`), 100k points | 7.14 ms | 51.14 ms | 7.16x |
| `latlon_to_zone_number`, 1M points (per-point API) | 2357.58 ms | 1783.35 ms | **0.76x slower** |

One `pixi run bench` run, best of five per case, on one machine: an Intel Xeon
E5-2697 v4, 72 logical CPUs (18 cores per socket, 2 threads per core, 2
sockets), Linux 6.8.0-142-generic, Python 3.13.14, numpy 2.5.1, upstream `utm`
0.9.0, Mojo 1.2.0.dev2026092905.

The box is shared with other jobs, and the load was 78-128 across the runs this
table was taken from, so treat it as an order of magnitude rather than the last
digit. Two things to read honestly out of that. The *upstream* column is the one
that moves most: `from_latlon` at 1M landed between 503 and 1412 ms depending
on what else was running, because numpy is memory-bound and sensitive to
contention. The speedup column therefore swings far more than the mojo-utm
column does. Across runs `from_latlon` at 1M landed between 14.3 and 30.2 ms and
`to_latlon` at 1M between 22.7 and 40.2 ms.

To measure a change on this box, do not compare two `pixi run bench` runs
minutes apart: at this load the same code measured 0.73x and then 1.84x on the
same array. The numbers above were taken with the old and new wrappers loaded
into one process and alternated, which is what the "What made the difference"
section reports.

`to_latlon` pays a little for `mod_angle` going through libc's `fmod` rather
than a single instruction. See [Divergences](#divergences) for why that call is
there at all; buying bit-identity with upstream was worth a few percent.

### What made the difference

The three changes below were kept because an alternating A/B in a single process
measured them faster; everything else that was tried is listed under
[What did not pay](#what-did-not-pay).

- **One allocation, one address lookup.** `a.ctypes.data` builds a fresh
  `_ctypes` proxy object on every attribute access, about 1.3 us, which is more
  than the conversion of a single point. `from_latlon` needed six addresses per
  call and the chunked path needed two per chunk. Every buffer a call touches
  now lives in one allocation laid out `[out0, out1, zone_number, zone_letter]`,
  so a call looks the address up once and slices it by a fixed stride. That
  takes the address lookups in a scalar call from 6 to 3, and at 1M points from
  196 to 99 - 97 lookups removed, about 126 us. Measured, alternating: a scalar
  `from_latlon` went from 24.09 us to 20.96 us, and the large-array paths are
  unchanged (ratio 0.996 over 15 alternating measurements at 1M).
- **`np.ndim` is only called when the inputs are not two floats.** Deciding
  "scalars or arrays" with `np.ndim` costs 2.0 us per pair; `type(a) is float`
  costs 0.18 us. Anything that is not exactly `float` on both sides still falls
  through to `np.ndim`, so an integer, a numpy scalar, a 0-d array and a list
  are classified exactly as before. Combined with the change above, a scalar
  `from_latlon` measured 25.42 us -> 18.57 us in one A/B, and 24.09 us -> 20.96
  us in another; call it 1.15x-1.37x depending on the load.
- **Contiguous `float64` inputs skip the broadcast.** Two C-contiguous
  `float64` arrays of the same shape are the case a caller with numpy data
  actually passes, and they now go through a reshape instead of
  `broadcast_arrays` + `ascontiguousarray`. Everything else - a scalar mixed
  with an array, a non-contiguous input, an integer array, a nested list - still
  takes the old path and produces the same numbers.

The changes already in place when this pass started are the fused, vectorised
`reduce_minmax` (one SIMD pass answers all four bounds checks upstream spells),
the southern-hemisphere offset folded into the kernel instead of a second pass
over the array, and the cross-core split described below.

The last row is slower and will stay slower. `latlon_to_zone_number` takes two
scalars, so a million of them is a million FFI crossings. Measured in a tight
loop on this machine, one call costs 1.04 us here against 0.88 us for upstream
pure Python, and the kernel itself is three floating-point operations. The gap
is the crossing plus the `float()` conversions ctypes requires. The API shape is
upstream's, so the fix is not ours to take; use the array form of `from_latlon`
when you have a vector of points.

### What did not pay

Each of these was implemented, measured, and dropped.

- **SIMD in the elementwise kernel.** The obvious win, and it is not available
  here. Mojo's `sin`/`cos` on a `SIMD[DType.float64, W]` value do not agree with
  its scalar `sin`/`cos` bit for bit: measured over 1M and 2M arguments, **every
  single argument differed**. Scalar `sin`/`cos` matches numpy's (and so glibc's)
  on all 3M of them. Vectorising the kernel would trade the bit-exact parity
  this library is built on for speed, which is not a trade this port can make.
  The reductions stay vectorised because `min` and `max` on `float64` are exact
  and have no such caveat. (`sin` does work on a `SIMD[DType.float32, ...]`, but
  that is the same answer as above with less precision, not a different one.)
- **`max.algorithm.parallelize` instead of the thread pool.** The README used to
  call this unusable; that is out of date. It imports and it compiles on this
  toolchain, with the `{imm}` nested-worker form from MOJO_NOTES section 5b and
  the explicit worker count. It is also slower than the thread fan-out it would
  replace: at 1M points it ran `from_latlon` in 15.5 ms against 8.6 ms for the
  same kernel on the thread pool (0.56x), and at 100k in 7.3 ms against 1.5 ms
  (0.20x). Per-launch thread setup is most of it, and the pool is paid for once
  and reused. Best worker count was 64, still behind. Not shipped.
- **A GPU path.** See below - not a missing API, a hard compiler constraint.

### The elementwise kernel is split across cores

`to_latlon` and `from_latlon` are each two functions in `src/ported.mojo`: a
`prepare` that answers every question needing the whole array, and a `kernel`
that converts one contiguous range and touches nothing else. The ranges are
independent - point `i` of the output depends only on point `i` of the input -
so `python/mojoutm/conversion.py` slices the flat buffer and runs the kernel on
all the slices at once. ctypes drops the GIL around a foreign call, so the
chunks really do run concurrently. Measured on the machine above at 1M points,
the fan-out runs `from_latlon` in 8.6 ms against 115.7 ms for the same kernel
forced onto one core, which is 13.4x; on 4 lanes' worth of hardware the loops
themselves do not change.

The split costs one extra FFI crossing, which is why an array too small to pay
for it takes the single-crossing `mu_to_latlon` / `mu_from_latlon` path instead.
`MIN_ELEMENTS_PER_CHUNK = 8192` is where the curve turns: 1M points fan out to
48 chunks (the cap, `min(48, os.cpu_count())`), 100k to 12, and anything smaller
stays on one core. Both thresholds are asserted in `tests/test_parity.py`,
together with the SIMD tail and a chunked-versus-serial bit-for-bit comparison.

`max.algorithm.parallelize` does work on this toolchain, which this section
used to deny - the `{imm}` nested-worker form from MOJO_NOTES section 5b
compiles where the plain `def work(i: Int):` form does not, and
`parallelize(work, workers, workers)` type-checks with the pointers and scalars
captured. A plain nested `def` still fails with `Could not infer capture
convention of the captured value` for a captured `Int`, and
`parallelize[func](n)` still does not type-check. It was measured rather than
assumed, and it lost: see [What did not pay](#what-did-not-pay). The thread pool
is what ships, because a `ThreadPoolExecutor` is built once and reused for every
later call while `parallelize` pays its per-launch setup every time.

### No GPU path

This is a compiler constraint, not a missing API, and the distinction is worth
making precisely because the API is present and works.

On this pin `from max.gpu.host import DeviceContext` and `from max.gpu import
block_idx, thread_idx` resolve, and the host path is complete:
`enqueue_create_buffer`, `enqueue_copy`, `enqueue_function` and `synchronize`
all compile and run. A GPU `sin` was written and executed on the RTX 5090 in
this box, with one launch per call and a single copy-in/copy-out, and it returned
the right answer (status 0, so it really ran on the GPU).

It does not ship, for two reasons.

1. **`DType.float64` `sin` is not implemented for the GPU target.** Writing the
   kernel in `float64` fails at compile time with `constraint failed:
   DType.float64 is not supported for sin on NVIDIA GPU`. This is a hard
   constraint in `std.math`, not something a different spelling works around.
2. **The only width that does work is `float32`, and that breaks the contract.**
   The `float32` kernel builds and runs, but over 1M arguments every single one
   differed from `float64` numpy's `sin`, with a worst absolute error of 3.4e-07
   - nine orders of magnitude above `float64` epsilon. This library's entire
   claim is bit-identity with upstream, so a GPU path would have to mean giving
   that up.

The same wall is why the elementwise kernels stay scalar while the reductions are
vectorised, and it is worth separating the two reasons, because they are
different and the earlier version of this file got the CPU one wrong. SIMD
`min`/`max` on `float64` are exact and vectorising them changes no bit, so the
reductions are vectorised. SIMD `sin`/`cos` on `float64` are *not* the same
function as scalar `sin`/`cos` here: measured over 1M and 2M arguments, every
one differed, while scalar `sin`/`cos` matched numpy on all 3M. So the
elementwise kernel cannot be vectorised either, for the same reason the GPU path
cannot be `float64`.

`from_latlon` is about 3 flops per byte on paper, so "below roughly 2 flops per
byte, keep it on the CPU" is not the deciding rule either - it would point the
other way. The deciding rule is that the kernel is five transcendental calls per
point, and this library's correctness property is that those calls are glibc's.


## Coverage

Ported, and the whole public surface of upstream is here:

| upstream name | ported as |
| --- | --- |
| `from_latlon` | `ported.from_latlon` = `ported.from_latlon_prepare` + `ported.from_latlon_kernel` |
| `to_latlon` | `ported.to_latlon` = `ported.to_latlon_prepare` + `ported.to_latlon_kernel` |
| `latlon_to_zone_number` | `ported.latlon_to_zone_number` |
| `latitude_to_zone_letter` | `ported.latitude_to_zone_letter` |
| `zone_number_to_central_longitude` | `ported.zone_number_to_central_longitude` |
| `zone_letter_to_central_latitude` | `ported.zone_letter_to_central_latitude` |
| `check_valid_zone` | `ported.check_valid_zone` |
| `OutOfRangeError` | `mojoutm.error.OutOfRangeError`, a `ValueError` subclass |

Upstream's private helpers - `in_bounds`, `check_valid_zone_letter`,
`check_valid_zone_number`, `mixed_signs`, `mod_angle` and every ellipsoid
constant from `K0` to `P5` - are ported too and stay private, which is where
upstream keeps them. Three names are not upstream's: `reduce_minmax` replaces
the pair of `np.min` / `np.max` calls upstream spells, `hemisphere` is the
"rebind `northern` from the zone letter" line written once because the kernel
split means both halves need it, and `python_mod` reproduces Python's `%` (see
[Divergences](#divergences)).

No upstream function is missing: every name in the table above is exercised by
`tests/test_parity.py` against the real `utm` package, and `test_exports_match_upstream`
asserts the public surface is the same set. Every `@export` in `src/` is also
called by name in `tests/test_kernels.py` and asserted on directly, so no
symbol ships unmeasured; `tests/test_numerics.py` pins the bit-identity claims
above against upstream rather than against stored constants.

### Divergences

Every one of these is a place where Mojo forced a different shape. None of them
change a result anywhere, including outside the domain `to_latlon` documents:
both entry points are bit-identical to upstream over the whole tested range.

- **`%` had to be spelled out.** This is the one that cost something. Upstream's
  `mod_angle` is `(value + pi) % (2*pi) - pi`, and Python's float `%` is an
  exact `fmod` (truncated division) followed by a sign correction. Mojo's `%`
  is `a - b*floor(a/b)`, which folds the rounding error of the quotient into the
  result, and `std.math` has no `fmod` to call instead. The two spellings agree
  on exact arithmetic and drift by an ulp once the quotient grows - and every
  longitude that wraps in `to_latlon` has a quotient of a few hundred, so the
  effect is not theoretical. Before this was found, a sweep of 60 zones by 20
  bands found 799 points whose longitude differed from upstream, the worst by
  2.4e-10 degrees, including points inside the documented band.
  `python_mod` now calls libc's `fmod` and applies the same sign correction, which
  is what CPython does, so it reproduces upstream's `%` exactly. Both of
  upstream's own backends spell it `%`, so matching it matches whichever one is
  in use. After the change the same sweep finds zero differences, and a
  past-the-pole sweep over 360,000 points across all 60 zones and 20 bands finds
  zero too - including the regions where upstream itself returns latitudes of
  1e19 degrees, which this now reproduces bit for bit rather than merely
  agreeing to a tolerance. The cost is one libc call per point in `to_latlon`
  and none in `from_latlon`, which never calls `mod_angle`.
- **Raises became status codes.** An `abi("C")` function cannot raise, so every
  Mojo function returns an `Int` and `python/mojoutm/conversion.py` maps a nonzero
  value back to upstream's exception class and message. The checks themselves
  still happen in Mojo, in upstream's order, at upstream's points in the
  functions. A status code nobody recognises would be a `KeyError`, which cannot
  happen today because the codes are defined in one place.
- **Zone letters cross as ASCII codes.** `zone_letter` is compared against `'C'`,
  `'N'` and `'X'` in upstream; in Mojo it is the character's code, and the
  uppercase is done in the Python wrapper. Upstream compares the letter as a
  whole string, so a multi-character one is judged by, and then used as, its
  first character; the wrapper does the same, which is why `zone_letter='VV'`
  behaves here as it does there and `'XY'` is out of range in both.
  `force_zone_letter` is returned verbatim, lowercase included, which is what
  upstream returns.
- **`None` crosses as a sentinel.** An unset `zone_letter` is 0 and an unset
  `northern` / `force_northern` is -1; upstream tests `is None`. A zone number
  crosses as `float64` rather than being narrowed, because upstream carries a
  float zone number through its arithmetic untouched.
- **An array `zone_number` is not broadcast.** Upstream would broadcast it and
  return arrays. This library takes `flat[0]`, the same rule upstream already
  applies to an array latitude in `latlon_to_zone_number`. The common case, a
  scalar zone number, is unchanged.
- **Inputs are broadcast before the call.** Upstream relies on numpy broadcasting
  inside its arithmetic and on `np.min` for its bounds checks, so it only
  partly supports a scalar mixed with an array. This library broadcasts first
  and therefore supports the combinations uniformly. Same numbers.
- **Zero-length arrays return zero-length arrays.** Upstream's `np.min` on an
  empty array raises `ValueError`; there is no `n` for the Mojo loop to run, so
  the call is skipped.

None of the above changes a number. The `%` rewrite did change numbers before it
was found; that is the first bullet above, and it is the reason the claim above
is bit-identity over the whole tested range rather than inside a band.

Three files make up `tests/`. `test_parity.py` walks the public API case by case
against upstream. `test_kernels.py` calls each `@export` by name and asserts on
its raw output, so a kernel that was wired up correctly but computed the wrong
answer cannot pass unnoticed. `test_numerics.py` pins the bit-identity claims on
bulk data. None of the three can pass without the shared library: they load it
at import and a build failure is an error, not a skip.

`utm` also falls back to the `math` module when numpy is absent. `mojoutm`
requires numpy: the Mojo side is over flat buffers, and making a scalar-only
path work would mean a second, differently-shaped API.

## How it works

```
python/mojoutm/conversion.py   upstream's signatures; None -> sentinel, status -> exception
        |  ctypes: one call for a small array, prepare + one call per chunk above
src/capi.mojo                  @export ... abi("C") wrappers, Int addresses, nothing else
src/ported.mojo                the numerics, over raw pointers, in upstream's source order
```

Three rules hold throughout:

- **Nothing in Mojo allocates.** Every routine writes into buffers the caller
  owns, so lifetimes are Python's problem and there is nothing to leak.
- **Contiguous float64 arrays are never copied.** They cross as an address and a
  length. A non-contiguous or non-float64 input is converted once, in numpy,
  where that is cheap. Inputs are copied only when broadcasting forces a new
  array.
- **Buffers cross as `Int` addresses.** `@export` refuses parametric functions
  and a pointer with an inferred origin is parametric, so `src/capi.mojo`
  rebuilds each pointer inside the wrapper with an explicit
  `AnyOrigin[mut=True]`.

### Memory layout

Everything in and out of Mojo is a flat, C-contiguous `float64` buffer, element
`i` holding point `i`. The kernel walks it once with `while i < n`; there are no
structs, no strides and no views to keep alive. A multi-dimensional input is
flattened in the wrapper, because a chunk is a slice of the flat buffer and not
of an axis.

Outputs come from one allocation per call, not one per array. `_out_buffers(n)`
returns a single `np.empty(2 * n + 2)` laid out as
`[output0, output1, zone_number, zone_letter]`, and the wrapper hands Mojo
`base + k * n * 8`. Reading an address is the expensive part of the crossing -
`a.ctypes.data` builds a fresh `_ctypes` proxy every time, about 1.3 us - so one
lookup and a fixed stride replaces one lookup per buffer. The two outputs the
caller gets back are views into that allocation, which is why the stride
arithmetic is asserted in `tests/test_parity.py`: an output pointer one stride
too far would write past the end of the buffer while still returning the right
answer.

The one structure that is not pointwise is the strict bounds check, which
upstream spells `lower <= np.min(x) and np.max(x) < upper` - a pair of
reductions over the whole array before any element is converted. That is
`reduce_minmax` in `ported.mojo`, in the same place, and it is the only
vectorised loop in the port: `min` and `max` on `float64` are exact, so the
`simd_width_of[DType.float64]()` lanes and the scalar tail give the same answer
as one element at a time.

One thing that pass has to handle explicitly is NaN. numpy's `np.min` and
`np.max` propagate it, so one NaN anywhere makes the bounds check fail and
upstream raises. Mojo's `min` and `max` are plain comparisons and drop it -
`min(1.0, nan)` is `1.0` - so relying on them alone would let a NaN through a
check upstream rejects and hand back a NaN coordinate. `reduce_minmax`
therefore watches for a NaN alongside the reduction and returns `NAN` for both
bounds when it sees one, which makes the comparisons in `in_bounds` fail
exactly as they do under numpy. `tests/test_parity.py` pins this across the
vector width and the chunking threshold, in the vector body and in the tail.

### Floating point

The build passes `--fp-mode contract=off`. The compiler default is
`contract=fast`, which fuses `a + b*c` into an FMA across statements; numpy does
not contract, so with contraction on every result would differ from upstream in
the last bit. With it off the arithmetic is the same operation sequence as
upstream's, and both entry points are bit-identical rather than merely close.
That leaves the two operators Mojo spells differently from Python: `%`, which
goes through libc's `fmod` so it matches CPython's definition exactly, and
`sin` / `cos` / `sqrt`, which come from `std.math` and are bit-identical to
glibc's over every input tested here (a 300,000-case sweep of `fmod` found no
disagreement).

## Related

- [mojo-sklearn](https://github.com/lee101/mojo-sklearn) - scikit-learn's core
  estimators in Mojo
- [mojo-plotly](https://github.com/lee101/mojo-plotly) - plotly figures with
  Mojo kernels
- [mojo-numba](https://github.com/lee101/mojo-numba) - numpy kernels in Mojo

## License

MIT
