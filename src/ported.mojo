"""A port of `utm/conversion.py` (utm 0.9.0, Tobias Bergstrom).

Read this file top to bottom next to `utm/conversion.py`: the constants, the
helper functions, their signatures and their branch structure follow upstream's
source order. Upstream expresses the elementwise arithmetic with numpy arrays;
Mojo has no implicit array type, so each of those array expressions becomes the
body of one `while i < n` loop and `n` is carried explicitly.

`to_latlon` and `from_latlon` are each two functions rather than one. The
`prepare` half answers every question that needs the whole array - the bounds
checks, the zone scalars, the mixed-sign rule - and the `kernel` half converts
one contiguous range and touches nothing else. The ranges are independent, so
`python/mojoutm/conversion.py` can run them on several cores at once; a small
array just runs one. Both halves stay in upstream's order.

The other structural change is error handling. Upstream raises; an `abi("C")`
function cannot. Every ported function that can fail therefore returns an `Int`
status, 0 for success, and `python/mojoutm/conversion.py` turns a nonzero status
back into upstream's exception type and message. The check itself still happens
here, in upstream's order.
"""


from std.math import sqrt, sin, cos, pi, nan
from std.sys import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int, AnyOrigin[mut=True]]

# Python's float `%` is an exact `fmod` (truncated division) followed by a sign
# correction. Mojo's is `a - b*floor(a/b)`, which carries the rounding error of
# the quotient into the result, and `std.math` has no `fmod`. Both spellings
# agree on exact arithmetic and drift by one ulp once the quotient is large
# enough - which is every longitude that wraps in `to_latlon`. Since the two
# implementations of upstream that this port must match (`numpy` when it is
# importable, `math` when it is not) both use the `%` operator, libc's `fmod`
# plus the same sign correction reproduces either one exactly.
@extern("fmod")
def libc_fmod(x: Float64, y: Float64) abi("C") -> Float64:
    ...


def python_mod(x: Float64, y: Float64) -> Float64:
    # CPython's `float_rem`: `fmod`, then add y back if the result is negative,
    # which is what makes `-3 % 4` a positive 1 rather than -3.
    var r = libc_fmod(x, y)
    if r < 0.0:
        r += y
    return r

# Lanes for the reductions. `min`/`max` on float64 is exact, so vectorising
# them changes no result, and the elementwise kernels stay scalar to keep the
# arithmetic in the same order upstream evaluates it.
comptime W = simd_width_of[DType.float64]()
comptime F = SIMD[DType.float64, W]
comptime FMAX = F(1.7976931348623157e308)
comptime FMIN = F(-1.7976931348623157e308)

# For most use cases in this module, numpy is indistinguishable from math,
# except it also works on numpy arrays. (upstream lines 3-11)

comptime K0 = 0.9996

comptime E = 0.00669438
comptime E2 = E * E
comptime E3 = E2 * E
comptime E_P2 = E / (1 - E)

comptime SQRT_E = sqrt(1 - E)
comptime _E = (1 - SQRT_E) / (1 + SQRT_E)
comptime _E2 = _E * _E
comptime _E3 = _E2 * _E
comptime _E4 = _E3 * _E
comptime _E5 = _E4 * _E

# numpy's `np.min` / `np.max` propagate NaN, so a NaN anywhere in the array
# makes the bounds check fail. Mojo's `min` / `max` do not: they are plain
# comparisons, so `min(1.0, nan)` is `1.0` and the NaN is dropped silently.
# A NaN would therefore pass every bounds check and reach the arithmetic,
# where it produces a NaN coordinate instead of upstream's OutOfRangeError.
# `NAN` is what `reduce_minmax` returns once it has seen one, so the
# comparisons in `in_bounds` fail exactly as they do under numpy.
comptime NAN = nan[DType.float64]()

comptime M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
comptime M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
comptime M3 = (15 * E2 / 256 + 45 * E3 / 1024)
comptime M4 = (35 * E3 / 3072)

comptime P2 = (3 / 2 * _E - 27 / 32 * _E3 + 269 / 512 * _E5)
comptime P3 = (21 / 16 * _E2 - 55 / 32 * _E4)
comptime P4 = (151 / 96 * _E3 - 417 / 128 * _E5)
comptime P5 = (1097 / 512 * _E4)

comptime R = 6378137.0

comptime ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"

# numpy's `deg2rad` / `rad2deg`; upstream's `mathlib.radians` / `degrees` are
# these same constants applied to the argument.
comptime DEG2RAD = pi / 180.0
comptime RAD2DEG = 180.0 / pi

# Upstream compares single-character strings. The comparisons it spells as
# `zone_letter >= 'N'` and `'C' <= zone_letter <= 'X'` are done on ASCII codes.
comptime LETTER_C = 67  # 'C'
comptime LETTER_I = 73  # 'I'
comptime LETTER_N = 78  # 'N'
comptime LETTER_O = 79  # 'O'
comptime LETTER_X = 88  # 'X'

# Status codes replacing upstream's raises. `python/mojoutm/conversion.py`
# holds the message each one maps to.
comptime OK = 0
comptime ERR_EASTING_RANGE = 1
comptime ERR_NORTHING_RANGE = 2
comptime ERR_ZONE_NUMBER_RANGE = 3
comptime ERR_ZONE_LETTER_RANGE = 4
comptime ERR_LATITUDE_RANGE = 5
comptime ERR_LONGITUDE_RANGE = 6
comptime ERR_MIXED_SIGNS = 7
comptime ERR_NEEDS_LETTER_OR_NORTHERN = 8
comptime ERR_LETTER_XOR_NORTHERN = 9
comptime ERR_FORCE_LETTER_XOR_NORTHERN = 10

# Sentinels for upstream's `None`: an unset zone letter is 0 and an unset
# `northern` / `force_northern` is -1. Anything upstream tests with `is None`
# is tested with `< 0` (or `== UNSET_LETTER`) here.
comptime UNSET_LETTER = 0
comptime UNSET_BOOL = -1


def zone_letter_at(index: Int) -> Int:
    # `ZONE_LETTERS[index]`. Mojo string indexing yields a one-character String
    # rather than a `str`, so `ord` recovers the code the letter comparisons and
    # the C ABI use. Every caller indexes a letter it has already range-checked.
    return ord(ZONE_LETTERS[codepoint=index])


def reduce_minmax(x: Ptr, n: Int) -> Tuple[Float64, Float64]:
    # upstream's `np.min(x)` and `np.max(x)` are two passes over the array;
    # every caller here wants both, so one vectorised pass produces both.
    #
    # `lo`/`hi` are seeded with the infinities and reduced in SIMD, then the
    # tail - the `n % W` elements the vector loop cannot cover - is walked one
    # at a time. `i` is the index the vector loop stopped at, so the tail
    # starts exactly where it left off and every element is seen once.
    #
    # A NaN in the array is tracked separately, because `min`/`max` drop it.
    # numpy propagates it into both bounds, which makes every comparison in
    # `in_bounds` false and raises; returning `NAN` reproduces that. SIMD
    # `!=` reduces to a single `Bool` here, so one test per vector is enough.
    if n <= 0:
        return (0.0, 0.0)
    var lo = FMAX
    var hi = FMIN
    var saw_nan = False
    var i = 0
    while i + W <= n:
        var v = x.unsafe_load[width=W](i)
        lo = min(lo, v)
        hi = max(hi, v)
        saw_nan = saw_nan or (v != v)
        i += W
    var lo_s = lo.reduce_min()
    var hi_s = hi.reduce_max()
    while i < n:
        var s = x.unsafe_load(i)
        lo_s = min(lo_s, s)
        hi_s = max(hi_s, s)
        if s != s:
            saw_nan = True
        i += 1
    if saw_nan:
        return (NAN, NAN)
    return (lo_s, hi_s)


def in_bounds(x: Ptr, n: Int, lower: Float64, upper: Float64, upper_strict: Bool = False) -> Bool:
    var bounds = reduce_minmax(x, n)
    if upper_strict:
        return lower <= bounds[0] and bounds[1] < upper
    return lower <= bounds[0] and bounds[1] <= upper


def check_valid_zone_letter(zone_letter: Int) -> Int:
    if not (LETTER_C <= zone_letter <= LETTER_X) or zone_letter == LETTER_I or zone_letter == LETTER_O:
        return ERR_ZONE_LETTER_RANGE
    return OK


# Upstream takes the zone number as whatever the caller passed. A float zone
# number is legal there and is carried through the arithmetic untouched, so
# these are `Float64` throughout: narrowing to `Int` here would silently
# answer for a different zone.
def check_valid_zone_number(zone_number: Float64) -> Int:
    if not 1.0 <= zone_number <= 60.0:
        return ERR_ZONE_NUMBER_RANGE
    return OK


def check_valid_zone(zone_number: Float64, zone_letter: Int) -> Int:
    var status = check_valid_zone_number(zone_number)
    if status != OK:
        return status
    if zone_letter != UNSET_LETTER:
        return check_valid_zone_letter(zone_letter)
    return OK


def mixed_signs(bounds: Tuple[Float64, Float64]) -> Bool:
    return bounds[0] < 0 and bounds[1] >= 0


def mod_angle(value: Float64) -> Float64:
    """Returns angle in radians to be between -pi and pi"""
    # `python_mod` is upstream's `%` verbatim; see its note above. Spelling this
    # `value % (2*pi)` would be the floor form, which is not what Python means.
    return python_mod(value + pi, 2 * pi) - pi


def hemisphere(zone_letter: Int, flag: Int) -> Int:
    # upstream rebinds `northern` / `is_northern` from the zone letter when the
    # letter is the one that was given. The split into `prepare` and `kernel`
    # makes the kernel need the letter as well as the flag, so the rule lives
    # here once and both halves call it.
    if zone_letter != UNSET_LETTER:
        return 1 if zone_letter >= LETTER_N else 0
    return flag


def to_latlon_prepare(
    easting: Ptr,
    northing: Ptr,
    n: Int,
    zone_number: Float64,
    zone_letter: Int,
    northern: Int,
    strict: Int,
) -> Int:
    """Everything `to_latlon` decides before it converts a single point.

    Only the strict bounds check and the zone scalars depend on the whole
    array; the conversion itself is elementwise, so the caller can run it
    over as many independent chunks as it has cores for.
    """
    if zone_letter == UNSET_LETTER and northern == UNSET_BOOL:
        return ERR_NEEDS_LETTER_OR_NORTHERN
    elif zone_letter != UNSET_LETTER and northern != UNSET_BOOL:
        return ERR_LETTER_XOR_NORTHERN

    if strict != 0:
        if not in_bounds(easting, n, 100000.0, 1000000.0, True):
            return ERR_EASTING_RANGE
        if not in_bounds(northing, n, 0.0, 10000000.0):
            return ERR_NORTHING_RANGE

    return check_valid_zone(zone_number, zone_letter)


def to_latlon_kernel(
    easting: Ptr,
    northing: Ptr,
    n: Int,
    zone_number: Float64,
    zone_letter: Int,
    northern: Int,
    out_latitude: Ptr,
    out_longitude: Ptr,
) -> None:
    """The elementwise conversion, with every whole-array decision already
    made by `to_latlon_prepare`. Each call converts one contiguous range, so
    the ranges can be run concurrently without changing any result."""
    # upstream: x = easting - 500000 ; y = northing if northern else northing - 10000000
    var y_offset = 0.0 if hemisphere(zone_letter, northern) != 0 else -10000000.0
    var central_lon_rad = DEG2RAD * zone_number_to_central_longitude(zone_number)

    var x: Float64
    var y: Float64

    var i = 0
    while i < n:
        x = easting.unsafe_load(i) - 500000.0
        y = northing.unsafe_load(i) + y_offset

        var m = y / K0
        var mu = m / (R * M1)

        var p_rad = (mu +
                     P2 * sin(2 * mu) +
                     P3 * sin(4 * mu) +
                     P4 * sin(6 * mu) +
                     P5 * sin(8 * mu))

        var p_sin = sin(p_rad)
        var p_sin2 = p_sin * p_sin

        var p_cos = cos(p_rad)

        var p_tan = p_sin / p_cos
        var p_tan2 = p_tan * p_tan
        var p_tan4 = p_tan2 * p_tan2

        var ep_sin = 1 - E * p_sin2
        var ep_sin_sqrt = sqrt(1 - E * p_sin2)

        var n_rad = R / ep_sin_sqrt
        var r = (1 - E) / ep_sin

        var c = E_P2 * p_cos**2
        var c2 = c * c

        var d = x / (n_rad * K0)
        var d2 = d * d
        var d3 = d2 * d
        var d4 = d3 * d
        var d5 = d4 * d
        var d6 = d5 * d

        var latitude = p_rad - (p_tan / r) * (
                     d2 / 2 -
                     d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2) +
                     d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2))

        var longitude = (d -
                     d3 / 6 * (1 + 2 * p_tan2 + c) +
                     d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4)) / p_cos

        longitude = mod_angle(longitude + central_lon_rad)

        out_latitude.unsafe_store(i, RAD2DEG * latitude)
        out_longitude.unsafe_store(i, RAD2DEG * longitude)
        i += 1


def to_latlon(
    easting: Ptr,
    northing: Ptr,
    n: Int,
    zone_number: Float64,
    zone_letter: Int,
    northern: Int,
    strict: Int,
    out_latitude: Ptr,
    out_longitude: Ptr,
) -> Int:
    """Converts UTM coordinates to Latitude and Longitude, in one crossing.

    An array small enough to stay on one core comes through here: a second FFI
    crossing costs more than the whole conversion does at that size.
    """
    var status = to_latlon_prepare(
        easting, northing, n, zone_number, zone_letter, northern, strict
    )
    if status != OK:
        return status
    to_latlon_kernel(
        easting, northing, n, zone_number, zone_letter, northern,
        out_latitude, out_longitude,
    )
    return OK



def from_latlon_prepare(
    latitude: Ptr,
    longitude: Ptr,
    n: Int,
    force_zone_number: Float64,
    force_zone_letter: Int,
    force_northern: Int,
    out_zone_number: Ptr,
    out_zone_letter: IPtr,
) -> Int:
    """Everything `from_latlon` decides before it converts a single point.

    The three whole-array questions upstream asks - the latitude band, the
    longitude band and the mixed-sign rule - are all answered by the same
    `min`/`max` pair over the latitudes, so one fused pass covers them, in
    upstream's order.
    """
    var lat_bounds = reduce_minmax(latitude, n)
    if not (-80.0 <= lat_bounds[0] and lat_bounds[1] <= 84.0):
        return ERR_LATITUDE_RANGE
    if not in_bounds(longitude, n, -180.0, 180.0):
        return ERR_LONGITUDE_RANGE
    if force_zone_letter != UNSET_LETTER and force_northern != UNSET_BOOL:
        return ERR_FORCE_LETTER_XOR_NORTHERN
    if force_zone_number != Float64(UNSET_BOOL):
        var status = check_valid_zone(force_zone_number, force_zone_letter)
        if status != OK:
            return status

    var zone_number: Float64
    if force_zone_number == Float64(UNSET_BOOL):
        zone_number = Float64(latlon_to_zone_number(latitude.unsafe_load(0), longitude.unsafe_load(0)))
    else:
        zone_number = force_zone_number

    var zone_letter: Int
    if force_zone_letter == UNSET_LETTER and force_northern == UNSET_BOOL:
        zone_letter = latitude_to_zone_letter(latitude.unsafe_load(0))
    else:
        zone_letter = force_zone_letter

    var check_signs = force_northern == UNSET_BOOL and force_zone_letter == UNSET_LETTER
    if check_signs and mixed_signs(lat_bounds):
        return ERR_MIXED_SIGNS

    out_zone_number.unsafe_store(0, zone_number)
    out_zone_letter.unsafe_store(0, zone_letter)
    return OK


def from_latlon_kernel(
    latitude: Ptr,
    longitude: Ptr,
    n: Int,
    zone_number: Float64,
    zone_letter: Int,
    force_northern: Int,
    out_easting: Ptr,
    out_northing: Ptr,
) -> None:
    """The elementwise conversion, with every whole-array decision already
    made by `from_latlon_prepare`. Each call converts one contiguous range, so
    the ranges can be run concurrently without changing any result.

    The southern-hemisphere offset upstream applies as a second pass over
    `northing` is folded into the value here, which saves reading and
    rewriting the whole array.
    """
    var central_lon_rad = DEG2RAD * zone_number_to_central_longitude(zone_number)
    var hemi_offset = 0.0 if hemisphere(zone_letter, force_northern) != 0 else 10000000.0

    var i = 0
    while i < n:
        var lat_rad = DEG2RAD * latitude.unsafe_load(i)
        var lat_sin = sin(lat_rad)
        var lat_cos = cos(lat_rad)

        var lat_tan = lat_sin / lat_cos
        var lat_tan2 = lat_tan * lat_tan
        var lat_tan4 = lat_tan2 * lat_tan2

        var lon_rad = DEG2RAD * longitude.unsafe_load(i)

        var n_rad = R / sqrt(1 - E * lat_sin**2)
        var c = E_P2 * lat_cos**2

        var a = lat_cos * mod_angle(lon_rad - central_lon_rad)
        var a2 = a * a
        var a3 = a2 * a
        var a4 = a3 * a
        var a5 = a4 * a
        var a6 = a5 * a

        var m = R * (M1 * lat_rad -
                 M2 * sin(2 * lat_rad) +
                 M3 * sin(4 * lat_rad) -
                 M4 * sin(6 * lat_rad))

        var easting = K0 * n_rad * (a +
                           a3 / 6 * (1 - lat_tan2 + c) +
                           a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

        var northing = K0 * (m + n_rad * lat_tan * (a2 / 2 +
                                            a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                            a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))
        out_easting.unsafe_store(i, easting)
        out_northing.unsafe_store(i, northing + hemi_offset)
        i += 1


def from_latlon(
    latitude: Ptr,
    longitude: Ptr,
    n: Int,
    force_zone_number: Float64,
    force_zone_letter: Int,
    force_northern: Int,
    out_easting: Ptr,
    out_northing: Ptr,
    out_zone_number: Ptr,
    out_zone_letter: IPtr,
) -> Int:
    """Converts Latitude and Longitude to UTM coordinates, in one crossing.

    An array small enough to stay on one core comes through here: a second FFI
    crossing costs more than the whole conversion does at that size.
    """
    var status = from_latlon_prepare(
        latitude, longitude, n, force_zone_number, force_zone_letter,
        force_northern, out_zone_number, out_zone_letter,
    )
    if status != OK:
        return status
    from_latlon_kernel(
        latitude, longitude, n, out_zone_number.unsafe_load(0),
        out_zone_letter.unsafe_load(0), force_northern,
        out_easting, out_northing,
    )
    return OK


def latitude_to_zone_letter(latitude: Float64) -> Int:
    # Returns 0 where upstream returns None.
    if -80.0 <= latitude <= 84.0:
        return zone_letter_at(Int(latitude + 80) >> 3)
    else:
        return UNSET_LETTER


def latlon_to_zone_number(latitude: Float64, longitude: Float64) -> Int:
    # Normalize longitude to be in the range [-180, 180)
    var lon = (longitude % 360.0 + 540.0) % 360.0 - 180.0

    # Special zone for Norway
    if 56.0 <= latitude < 64.0 and 3.0 <= lon < 12.0:
        return 32
    # Special zones for Svalbard
    if 72.0 <= latitude <= 84.0 and lon >= 0.0:
        if lon < 9.0:
            return 31
        elif lon < 21.0:
            return 33
        elif lon < 33.0:
            return 35
        elif lon < 42.0:
            return 37

    return Int((lon + 180.0) / 6.0) + 1


def zone_number_to_central_longitude(zone_number: Float64) -> Float64:
    # upstream calls check_valid_zone_number first; every caller here does.
    return (zone_number - 1) * 6 - 180 + 3


def zone_letter_to_central_latitude(zone_letter: Int) -> Int:
    # upstream calls check_valid_zone_letter first; every caller here does.
    if zone_letter == LETTER_X:
        return 78
    else:
        var i = 0
        while i < 21:
            if zone_letter_at(i) == zone_letter:
                return -76 + (i * 8)
            i += 1
        return 0
