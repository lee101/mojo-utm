"""`mojoutm` - the public API, mirroring `utm` 0.9.0.

Every name, argument order, default and return type here is upstream's, so this
module is a drop-in replacement for the covered subset. The numerics live in
`src/ported.mojo`; what is left here is argument marshalling: `None` becomes a
sentinel, a status code becomes an exception, and numpy arrays cross the C ABI
by address without a copy. An array long enough to be worth splitting is cut
into contiguous ranges and converted by several calls at once; the ranges are
disjoint and the kernel reads nothing outside its own, so the answer does not
depend on where they fall.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mojoutm import _lib
from mojoutm._lib import addr, letter_code, letter_code_exact, bool_flag, lib
from mojoutm.error import OutOfRangeError

__all__ = [
    "to_latlon",
    "from_latlon",
    "latlon_to_zone_number",
    "latitude_to_zone_letter",
    "check_valid_zone",
    "zone_number_to_central_longitude",
    "zone_letter_to_central_latitude",
    "OutOfRangeError",
]


# The elementwise kernels are pure: point `i` of the output depends only on
# point `i` of the input. Splitting the range across cores therefore cannot
# change a single bit of the answer - ctypes drops the GIL around a foreign
# call, so the chunks really do run at the same time.
#
# Below this many elements per chunk the fan-out costs more than it saves, so
# small arrays stay on one core. 8192 is where the curve turns, and 48 is the
# cap: at 1M points the fan-out runs 13.4x faster than the same kernel on one
# core, and past 48 chunks the extra crossings outweigh the extra workers.
MIN_ELEMENTS_PER_CHUNK = 8192
MAX_CHUNKS = min(48, os.cpu_count() or 1)

_executor = None


def _chunk_bounds(n: int) -> list[tuple[int, int]]:
    """Split `[0, n)` into contiguous ranges, or return `[]` to stay serial."""
    count = min(MAX_CHUNKS, n // MIN_ELEMENTS_PER_CHUNK)
    if count < 2:
        return []
    step = -(-n // count)
    return [(start, min(start + step, n)) for start in range(0, n, step)]


def _fan_out(job, bounds) -> None:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(MAX_CHUNKS, thread_name_prefix="mojoutm")
    # `map` is drained here so a failure inside a chunk surfaces to the caller.
    for _ in _executor.map(job, bounds):
        pass


def _pairs(a, b):
    """Broadcast two inputs the way numpy broadcasts inside upstream's arithmetic.

    Returns them flattened, because the kernels address one flat contiguous
    buffer and a chunk is a slice of that buffer, not a slice of an axis.

    Two `float64` C-contiguous arrays of the same shape - what a caller who
    already has numpy data has - are the common case and are passed through by
    a reshape, which is a view. Everything else, including a broadcast against a
    scalar, a non-contiguous input, an integer array and a nested list, goes
    through `broadcast_arrays` and is materialised exactly as before.
    """
    if (
        type(a) is np.ndarray
        and type(b) is np.ndarray
        and a.dtype == np.float64
        and b.dtype == np.float64
        and a.shape == b.shape
        and a.flags.c_contiguous
        and b.flags.c_contiguous
    ):
        return a.reshape(-1), b.reshape(-1), a.shape
    x, y = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    return (
        np.ascontiguousarray(x).reshape(-1),
        np.ascontiguousarray(y).reshape(-1),
        x.shape,
    )


# Reading an array's address costs about 1.3 us - `a.ctypes` builds a fresh
# `_ctypes` proxy on every attribute access - which is more than the conversion
# of a single point. Every buffer a conversion needs therefore lives in one
# allocation, so a call looks up one address and slices it: the base plus a
# fixed stride instead of a lookup per array.
#
# The addresses the wrappers hand over are `address + k * (n * 8)`, computed
# from the one address they looked up, so a chunk never needs an `addr()` of its
# own either.
def _out_buffers(n: int):
    """One allocation holding the outputs, the zone number and the zone letter.

    Laid out `[out0, out1, zone_number, zone_letter]`: the two `float64` outputs,
    then one `float64` for the zone number and one `int64` for the zone letter.
    The two returned views are exactly `n` elements each, so they reshape and
    index like ordinary outputs.
    """
    buf = np.empty(2 * n + 2, dtype=np.float64)
    return buf, buf[:n], buf[n:2 * n]


def _is_scalar(a, b) -> bool:
    # `np.ndim` is 2 us per pair here, which is a tenth of a scalar call, and
    # the answer for two floats is decided by their type alone. Anything that is
    # not exactly `float` on both sides falls through to `np.ndim`, so an
    # integer, a numpy scalar, a 0-d array and a list all take the slow path and
    # are classified exactly as before.
    if type(a) is float and type(b) is float:
        return True
    return np.ndim(a) == 0 and np.ndim(b) == 0


def _scalar_zone_number(zone_number) -> float:
    # Upstream would broadcast an array zone number and return arrays; this
    # takes the first element instead, matching the `flat[0]` rule upstream
    # already applies to array latitudes and longitudes.
    if isinstance(zone_number, np.ndarray):
        zone_number = zone_number.flat[0]
    # float, not int: upstream carries a float zone number through its
    # arithmetic untouched, so narrowing here would answer for a different
    # zone than the caller asked about. `nan` stays `nan` and is rejected by
    # the range check, as upstream rejects it.
    return float(zone_number)


def to_latlon(easting, northing, zone_number, zone_letter=None, northern=None, strict=True):
    """This function converts UTM coordinates to Latitude and Longitude

        Parameters
        ----------
        easting: int or NumPy array
            Easting value of UTM coordinates

        northing: int or NumPy array
            Northing value of UTM coordinates

        zone_number: int
            Zone number is represented with global map numbers of a UTM zone
            numbers map. For more information see utmzones [1]_

        zone_letter: str
            Zone letter can be represented as string values.  UTM zone
            designators can be seen in [1]_

        northern: bool
            You can set True (North) or False (South) as an alternative to
            providing a zone letter. Default is None

        strict: bool
            Raise an OutOfRangeError if outside of bounds

        Returns
        -------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)

        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).


       .. _[1]: http://www.jaworski.ca/utmzones.htm

    """
    scalar = _is_scalar(easting, northing)
    e, no, shape = _pairs(easting, northing)
    n = e.size
    buf, lat, lon = _out_buffers(n)
    if n:
        handle = lib()
        zone = _scalar_zone_number(zone_number)
        letter = letter_code(zone_letter)
        north = bool_flag(northern)
        base = addr(buf)
        stride = n * 8
        bounds = _chunk_bounds(n)
        if not bounds:
            # One crossing, not two: at this size the crossing costs more than
            # the arithmetic.
            status = handle.mu_to_latlon(
                addr(e), addr(no), n, zone, letter, north,
                1 if strict else 0, base, base + stride,
            )
            if status:
                _lib.raise_status(status)
        else:
            status = handle.mu_to_latlon_prepare(
                addr(e), addr(no), n, zone, letter, north, 1 if strict else 0,
            )
            if status:
                _lib.raise_status(status)
            kernel = handle.mu_to_latlon_kernel

            def job(chunk):
                start, stop = chunk
                offset = start * 8
                count = stop - start
                kernel(
                    addr(e[start:stop]), addr(no[start:stop]), count,
                    zone, letter, north,
                    base + offset, base + stride + offset,
                )

            _fan_out(job, bounds)
    if scalar:
        return lat[0], lon[0]
    return lat.reshape(shape), lon.reshape(shape)


def from_latlon(latitude, longitude, force_zone_number=None, force_zone_letter=None, force_northern=None):
    """This function converts Latitude and Longitude to UTM coordinate

        Parameters
        ----------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)

        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).

        force_zone_number: int
            Zone number is represented by global map numbers of an UTM zone
            numbers map. You may force conversion to be included within one
            UTM zone number.  For more information see utmzones [1]_

        force_zone_letter: str
            You may force conversion to be included within one UTM zone
            letter.  For more information see utmzones [1]_

        force_northern: bool
            You can set True (North) or False (South) as an alternative to
            forcing with a zone letter. When set, the returned zone_letter will
            be None. Default is None

        Returns
        -------
        easting: float or NumPy array
            Easting value of UTM coordinates

        northing: float or NumPy array
            Northing value of UTM coordinates

        zone_number: int
            Zone number is represented by global map numbers of an UTM zone
            numbers map. More information see utmzones [1]_

        zone_letter: str
            Zone letter is represented as a string value. UTM zone designators
            can be accessed in [1]_


       .. _[1]: http://www.jaworski.ca/utmzones.htm
    """
    scalar = _is_scalar(latitude, longitude)
    lat, lon, shape = _pairs(latitude, longitude)
    n = lat.size
    # float64 throughout: upstream returns a float zone number verbatim when one
    # was forced, and the zone letter as a character, which is exactly the int64
    # code `chr` turns back into a string below. One allocation holds all four.
    buf, easting, northing = _out_buffers(n)
    zone_number = buf[2 * n:2 * n + 1]
    zone_letter = buf[2 * n + 1:2 * n + 2].view(np.int64)
    if n:
        handle = lib()
        # -1.0 is the kernel's "no force_zone_number" sentinel, not a zone.
        fzn = (
            float(_lib.UNSET_BOOL)
            if force_zone_number is None
            else _scalar_zone_number(force_zone_number)
        )
        fzl = letter_code(force_zone_letter)
        fnorth = bool_flag(force_northern)
        base = addr(buf)
        stride = n * 8
        bounds = _chunk_bounds(n)
        if not bounds:
            # One crossing, not two: at this size the crossing costs more than
            # the arithmetic.
            status = handle.mu_from_latlon(
                addr(lat), addr(lon), n, fzn, fzl, fnorth,
                base, base + stride, base + 2 * stride, base + 2 * stride + 8,
            )
            if status:
                _lib.raise_status(status)
        else:
            status = handle.mu_from_latlon_prepare(
                addr(lat), addr(lon), n, fzn, fzl, fnorth,
                base + 2 * stride, base + 2 * stride + 8,
            )
            if status:
                _lib.raise_status(status)
            zone = float(zone_number[0])
            letter = int(zone_letter[0])
            kernel = handle.mu_from_latlon_kernel

            def job(chunk):
                start, stop = chunk
                offset = start * 8
                count = stop - start
                kernel(
                    addr(lat[start:stop]), addr(lon[start:stop]), count,
                    zone, letter, fnorth,
                    base + offset, base + stride + offset,
                )

            _fan_out(job, bounds)
    # upstream returns the forced zone number verbatim and the computed one as
    # an int, so the type of what the caller passed is preserved here.
    zn = force_zone_number if force_zone_number is not None else int(zone_number[0])
    if force_zone_letter is not None:
        # upstream returns the forced letter verbatim, case included; the
        # kernel only ever sees the uppercased form the comparisons use.
        zl = force_zone_letter
    else:
        zl = chr(int(zone_letter[0])) if zone_letter[0] else None
    if scalar:
        return easting[0], northing[0], zn, zl
    return easting.reshape(shape), northing.reshape(shape), zn, zl


def latitude_to_zone_letter(latitude):
    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if isinstance(latitude, np.ndarray):
        latitude = latitude.flat[0]

    code = lib().mu_latitude_to_zone_letter(float(latitude))
    return chr(code) if code else None


_ndarray = np.ndarray
_mu_latlon_to_zone_number = None


def latlon_to_zone_number(latitude, longitude):
    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if isinstance(latitude, _ndarray):
        latitude = latitude.flat[0]
    if isinstance(longitude, _ndarray):
        longitude = longitude.flat[0]

    # This is the only entry point a caller can drive one point at a time, so
    # the wrapper is kept to the bone: the bound ctypes function is cached in a
    # module global rather than re-fetched from the CDLL, and `isinstance` reads
    # a module global rather than an attribute of `numpy`. The call itself
    # costs about 0.8 us and is not reducible from Python.
    kernel = _mu_latlon_to_zone_number
    if kernel is None:
        kernel = _bind_latlon_to_zone_number()
    return kernel(float(latitude), float(longitude))


def _bind_latlon_to_zone_number():
    global _mu_latlon_to_zone_number
    _mu_latlon_to_zone_number = lib().mu_latlon_to_zone_number
    return _mu_latlon_to_zone_number


def check_valid_zone(zone_number, zone_letter):
    status = lib().mu_check_valid_zone(_scalar_zone_number(zone_number), letter_code(zone_letter))
    if status:
        _lib.raise_status(status)


def zone_number_to_central_longitude(zone_number):
    # float64 out: upstream's answer is float for a float zone number, and
    # this buffer is where it comes back.
    out = np.zeros(1, dtype=np.float64)
    status = lib().mu_zone_number_to_central_longitude(_scalar_zone_number(zone_number), addr(out))
    if status:
        _lib.raise_status(status)
    # upstream's arithmetic is int-in-int-out and float-in-float-out, so the
    # returned type follows the argument's.
    value = float(out[0])
    return value if isinstance(zone_number, float) else int(value)


def zone_letter_to_central_latitude(zone_letter):
    out = np.zeros(1, dtype=np.int64)
    status = lib().mu_zone_letter_to_central_latitude(letter_code_exact(zone_letter), addr(out))
    if status:
        _lib.raise_status(status)
    return int(out[0])
