"""Parity of `mojoutm` against the real `utm` 0.9.0 package.

Every test compares against upstream rather than against a stored value, except
`test_published_doc_vectors`, which pins the numbers upstream documents in its
own README.

One documented divergence runs through all of this: Mojo's float `%` is
`a - b*floor(a/b)`, where Python's is an exact `fmod` followed by a sign
correction. Inside the [-80, 84] degree band that `to_latlon` documents the two
agree; past the pole the quotient grows and they stop agreeing, but so does the
answer - upstream returns latitudes of 1e17 degrees there. `from_latlon` never
touches a large quotient and is bit-exact everywhere.
"""

from __future__ import annotations

import gc
import inspect

import numpy as np
import pytest

import mojoutm
import utm

EXPORTS = [
    "to_latlon",
    "from_latlon",
    "latlon_to_zone_number",
    "latitude_to_zone_letter",
    "check_valid_zone",
    "zone_number_to_central_longitude",
    "zone_letter_to_central_latitude",
    "OutOfRangeError",
]

# Worst longitude disagreement observed anywhere in the documented band, in
# degrees. 1.3e-11 deg is 1.4 mm; see the module docstring.
LON_TOL_DEG = 1e-9


def outcome(fn, *args, **kwargs):
    """Either the return value or upstream's exact exception type and message."""
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - comparing failures is the point
        return ("raise", type(exc), str(exc))


def assert_same_failure(mine, theirs):
    """The two modules define separate `OutOfRangeError` classes, so compare the
    class by name rather than by identity; the message must be identical."""
    assert mine[0] == theirs[0], f"{mine} vs {theirs}"
    if mine[0] == "raise":
        assert mine[1].__name__ == theirs[1].__name__, f"{mine} vs {theirs}"
        assert mine[1].__mro__[1].__name__ == theirs[1].__mro__[1].__name__, f"{mine} vs {theirs}"
        assert mine[2] == theirs[2], f"{mine} vs {theirs}"
    else:
        assert mine[1] == theirs[1], f"{mine} vs {theirs}"


def random_points(n, rng, northern=True):
    # upstream refuses a latitude array that mixes signs, so every bulk case
    # stays on one side of the equator.
    lat = rng.uniform(1e-3, 84.0, n) if northern else rng.uniform(-80.0, -1e-3, n)
    lat = np.round(lat, 6)
    lon = np.round(rng.uniform(-179.999999, 179.999999, n), 6)
    return lat, lon


# --------------------------------------------------------------------------
# API shape
# --------------------------------------------------------------------------


def test_exports_match_upstream():
    public = {n for n in dir(utm) if not n.startswith("_")}
    assert public == {n for n in dir(mojoutm) if not n.startswith("_") and n != "__version__"}
    for name in EXPORTS:
        assert hasattr(mojoutm, name), name
        assert hasattr(utm, name), name


def test_signatures_match_upstream():
    for name in EXPORTS[:-1]:
        mine = inspect.signature(getattr(mojoutm, name))
        theirs = inspect.signature(getattr(utm, name))
        assert mine == theirs, f"{name}: {mine} != {theirs}"


def test_error_type_matches_upstream():
    assert mojoutm.OutOfRangeError is not utm.OutOfRangeError
    assert issubclass(mojoutm.OutOfRangeError, ValueError)
    assert issubclass(utm.OutOfRangeError, ValueError)


# --------------------------------------------------------------------------
# published vectors (upstream's own README)
# --------------------------------------------------------------------------


def test_published_doc_vectors():
    # Upstream's README rounds to 13 significant digits; compare to that, then
    # pin the exact value upstream's code produces.
    easting, northing, zone_number, zone_letter = mojoutm.from_latlon(51.2, 7.5)
    assert zone_number == 32 and zone_letter == "U"
    assert easting == pytest.approx(395201.3103811303, rel=1e-12)
    assert northing == pytest.approx(5673135.241182375, rel=1e-12)
    assert (easting, northing, zone_number, zone_letter) == utm.from_latlon(51.2, 7.5)

    # Upstream's README quotes (51.51852098408468, 6.693872395145327); the
    # installed 0.9.0 returns values a few e-5 degrees away, so the README
    # numbers are pinned loosely and the real output is pinned exactly.
    assert mojoutm.to_latlon(340000, 5710000, 32, "U") == pytest.approx(
        (51.51852098408468, 6.693872395145327), rel=1e-4
    )
    assert mojoutm.to_latlon(340000, 5710000, 32, "U") == utm.to_latlon(340000, 5710000, 32, "U")

    easting, northing, zone_number, zone_letter = mojoutm.from_latlon(
        np.array([51.2, 49.0]), np.array([7.5, 8.4])
    )
    # The README prints these at 8 significant digits; upstream's own array
    # output carries the rest.
    np.testing.assert_allclose(easting, np.array([395201.31038113, 456114.59586214]), rtol=1e-12)
    np.testing.assert_allclose(northing, np.array([5673135.24118237, 5427629.20426126]), rtol=1e-12)
    assert (zone_number, zone_letter) == (32, "U")
    theirs = utm.from_latlon(np.array([51.2, 49.0]), np.array([7.5, 8.4]))
    assert np.array_equal(easting, theirs[0])
    assert np.array_equal(northing, theirs[1])


# --------------------------------------------------------------------------
# from_latlon
# --------------------------------------------------------------------------


@pytest.mark.parametrize("northern", [True, False])
def test_from_latlon_scalar_parity(northern):
    rng = np.random.default_rng(12345)
    lat, lon = random_points(400, rng, northern=northern)
    for la, lo in zip(lat, lon):
        assert mojoutm.from_latlon(float(la), float(lo)) == utm.from_latlon(float(la), float(lo))


def test_from_latlon_array_parity_north():
    rng = np.random.default_rng(7)
    lat, lon = random_points(200_000, rng, northern=True)
    mine = mojoutm.from_latlon(lat, lon)
    theirs = utm.from_latlon(lat, lon)
    assert mine[2:] == theirs[2:]
    assert np.array_equal(mine[0], theirs[0]), "easting must be bit-identical"
    assert np.array_equal(mine[1], theirs[1]), "northing must be bit-identical"


def test_from_latlon_array_parity_south():
    rng = np.random.default_rng(8)
    lat, lon = random_points(200_000, rng, northern=False)
    mine = mojoutm.from_latlon(lat, lon)
    theirs = utm.from_latlon(lat, lon)
    assert mine[2:] == theirs[2:]
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


def test_from_latlon_preserves_shape():
    lat = np.linspace(1.0, 40.0, 20_000).reshape(100, 200)
    lon = np.linspace(-10.0, 10.0, 20_000).reshape(100, 200)
    easting, northing, _, _ = mojoutm.from_latlon(lat, lon)
    assert easting.shape == (100, 200)
    assert northing.shape == (100, 200)
    te, tn, _, _ = utm.from_latlon(lat, lon)
    assert np.array_equal(easting, te)
    assert np.array_equal(northing, tn)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"force_zone_number": 33},
        {"force_zone_letter": "V"},
        {"force_zone_letter": "N"},
        {"force_northern": True},
        {"force_northern": False},
        {"force_zone_number": 30, "force_zone_letter": "S"},
        {"force_zone_number": 30, "force_northern": True},
    ],
)
def test_from_latlon_forced_parity(kwargs):
    rng = np.random.default_rng(99)
    lat, lon = random_points(50_000, rng)
    mine = mojoutm.from_latlon(lat, lon, **kwargs)
    theirs = utm.from_latlon(lat, lon, **kwargs)
    assert mine[2:] == theirs[2:]
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


def test_from_latlon_lowercase_zone_letter_parity():
    mine = mojoutm.from_latlon(51.2, 7.5, force_zone_letter="u")
    theirs = utm.from_latlon(51.2, 7.5, force_zone_letter="u")
    assert mine == theirs
    assert mine[3] == "u"


# --------------------------------------------------------------------------
# to_latlon
# --------------------------------------------------------------------------


def test_to_latlon_scalar_parity():
    """Round-tripped through `from_latlon`, so every point is inside the
    documented [-80, 84] band where the inverse series converges and the
    comparison is bit-exact rather than merely close."""
    rng = np.random.default_rng(4242)
    for lo, hi in ((-79.9, -1e-3), (1e-3, 83.9)):
        lat = rng.uniform(lo, hi, 250)
        lon = rng.uniform(-0.5, 0.5, 250)
        easting, northing, zone_number, zone_letter = mojoutm.from_latlon(lat, lon)
        for e, n in zip(easting, northing):
            assert mojoutm.to_latlon(e, n, zone_number, zone_letter=zone_letter) == \
                utm.to_latlon(e, n, zone_number, zone_letter=zone_letter)


def test_to_latlon_array_parity_in_domain():
    rng = np.random.default_rng(2024)
    lat = np.round(rng.uniform(1e-3, 84.0, 300_000), 6)
    lon = np.round(rng.uniform(-0.5, 0.5, 300_000), 6)
    easting, northing, zone_number, zone_letter = mojoutm.from_latlon(lat, lon)
    mine = mojoutm.to_latlon(easting, northing, zone_number, zone_letter=zone_letter)
    theirs = utm.to_latlon(easting, northing, zone_number, zone_letter=zone_letter)
    assert np.array_equal(mine[0], theirs[0]), "latitude must be bit-identical"
    assert np.array_equal(mine[1], theirs[1]), "longitude must be bit-identical"
    assert np.max(np.abs(mine[0] - lat)) < 1e-6
    assert np.max(np.abs(mine[1] - lon)) < 1e-6


def test_to_latlon_array_parity_full_sweep():
    """Every easting in the zone and every northing from equator to pole.
    Latitude stays bit-exact; longitude stays inside LON_TOL_DEG while the
    result is a real place, and the whole sweep is asserted against that."""
    rng = np.random.default_rng(555)
    n = 200_000
    easting = rng.integers(100_000, 1_000_000, n).astype(np.float64)
    northing = rng.integers(0, 10_000_001, n).astype(np.float64)
    for northern in (True, False):
        mine_lat, mine_lon = mojoutm.to_latlon(easting, northing, 33, northern=northern)
        their_lat, their_lon = utm.to_latlon(easting, northing, 33, northern=northern)
        assert np.array_equal(mine_lat, their_lat), "latitude must be bit-identical"
        in_band = np.abs(their_lat) <= 84.0
        assert np.max(np.abs(mine_lon[in_band] - their_lon[in_band])) < LON_TOL_DEG


def test_to_latlon_zone_letter_and_northern_agree():
    easting = np.full(1000, 500000.0)
    northing = np.full(1000, 5000000.0)
    by_letter = mojoutm.to_latlon(easting, northing, 33, zone_letter="N")
    by_flag = mojoutm.to_latlon(easting, northing, 33, northern=True)
    assert np.array_equal(by_letter[0], by_flag[0])
    assert np.array_equal(by_letter[1], by_flag[1])


def test_to_latlon_strict_false_parity():
    easting = np.array([50_000.0, 1_000_000.0, 2_000_000.0])
    northing = np.array([5_000_000.0, 5_000_000.0, 5_000_000.0])
    mine = mojoutm.to_latlon(easting, northing, 33, northern=True, strict=False)
    theirs = utm.to_latlon(easting, northing, 33, northern=True, strict=False)
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


def test_to_latlon_array_zone_number_is_accepted():
    easting = np.array([500000.0, 500000.0])
    northing = np.array([5000000.0, 5000000.0])
    got = mojoutm.to_latlon(easting, northing, np.array([33, 33]), northern=True)
    want = mojoutm.to_latlon(easting, northing, 33, northern=True)
    assert np.array_equal(got[0], want[0])
    assert np.array_equal(got[1], want[1])


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_round_trip_is_stable():
    rng = np.random.default_rng(31337)
    lat = np.round(rng.uniform(1e-3, 84.0, 100_000), 6)
    lon = np.round(rng.uniform(-0.5, 0.5, 100_000), 6)
    easting, northing, zone_number, zone_letter = mojoutm.from_latlon(lat, lon)
    back_lat, back_lon = mojoutm.to_latlon(easting, northing, zone_number, zone_letter=zone_letter)
    assert np.max(np.abs(back_lat - lat)) < 1e-6
    assert np.max(np.abs(back_lon - lon)) < 1e-6
    their_lat, their_lon = utm.to_latlon(easting, northing, zone_number, zone_letter=zone_letter)
    assert np.array_equal(back_lat, their_lat)
    assert np.array_equal(back_lon, their_lon)



# --------------------------------------------------------------------------
# chunked kernels
# --------------------------------------------------------------------------


# `MIN_ELEMENTS_PER_CHUNK` in `mojoutm.conversion` decides where an array stops
# running on one core. Both sides of that boundary, and the sizes that make the
# per-chunk SIMD reduction leave a scalar tail, are checked here against the
# serial path and against upstream.
CHUNKED_SIZES = [
    1,
    4,
    7,
    3 * 8 + 1,          # one SIMD vector plus a five-element tail
    8 * 8 - 1,
    8 * 8,
    8 * 8 + 1,
    mojoutm.conversion.MIN_ELEMENTS_PER_CHUNK - 1,
    mojoutm.conversion.MIN_ELEMENTS_PER_CHUNK,
    mojoutm.conversion.MIN_ELEMENTS_PER_CHUNK * 2 + 3,
    300_000,            # several chunks, and not a round number of them
]


@pytest.mark.parametrize("n", CHUNKED_SIZES)
def test_from_latlon_chunk_boundaries_match_upstream(n):
    rng = np.random.default_rng(1000 + n)
    lat, lon = random_points(n, rng)
    mine = mojoutm.from_latlon(lat, lon)
    theirs = utm.from_latlon(lat, lon)
    assert mine[2:] == theirs[2:]
    assert np.array_equal(mine[0], theirs[0]), "easting must be bit-identical"


@pytest.mark.parametrize("n", CHUNKED_SIZES)
def test_to_latlon_chunk_boundaries_match_upstream(n):
    rng = np.random.default_rng(2000 + n)
    # longitudes near a zone meridian, so every easting lands inside the band
    # `to_latlon` checks for in strict mode
    lat = np.round(rng.uniform(1e-3, 84.0, n), 6)
    lon = np.round(rng.uniform(-0.5, 0.5, n), 6)
    easting, northing, zone_number, zone_letter = mojoutm.from_latlon(lat, lon)
    mine = mojoutm.to_latlon(easting, northing, zone_number, zone_letter=zone_letter)
    theirs = utm.to_latlon(easting, northing, zone_number, zone_letter=zone_letter)
    assert np.array_equal(mine[0], theirs[0]), "latitude must be bit-identical"
    assert np.array_equal(mine[1], theirs[1]), "longitude must be bit-identical"


@pytest.mark.parametrize("n", CHUNKED_SIZES)
def test_southern_hemisphere_offset_survives_chunking(n):
    # The +10000000 offset is folded into the kernel rather than applied in a
    # second pass, so it has to land on the last element of the last chunk too.
    rng = np.random.default_rng(3000 + n)
    lat, lon = random_points(n, rng, northern=False)
    mine = mojoutm.from_latlon(lat, lon)
    theirs = utm.from_latlon(lat, lon)
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


def test_chunked_and_serial_agree_bit_for_bit():
    """The fan-out splits a flat buffer into ranges; the answer must not depend
    on where the splits fall."""
    rng = np.random.default_rng(4242)
    n = mojoutm.conversion.MIN_ELEMENTS_PER_CHUNK * 5 + 17
    lat, lon = random_points(n, rng)
    bounds = mojoutm.conversion._chunk_bounds(n)
    assert len(bounds) >= 2
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    for (_, stop), (start, _) in zip(bounds, bounds[1:]):
        assert stop == start, "chunks must tile the input with no gap or overlap"

    fan = mojoutm.from_latlon(lat, lon)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mojoutm.conversion, "_chunk_bounds", lambda n: [])
        serial = mojoutm.from_latlon(lat, lon)
    assert np.array_equal(fan[0], serial[0]), "easting must not depend on the split"
    assert np.array_equal(fan[1], serial[1]), "northing must not depend on the split"
    assert fan[2:] == serial[2:]


def test_small_arrays_stay_on_one_chunk():
    assert mojoutm.conversion._chunk_bounds(0) == []
    assert mojoutm.conversion._chunk_bounds(1) == []
    n = mojoutm.conversion.MIN_ELEMENTS_PER_CHUNK * 2
    assert len(mojoutm.conversion._chunk_bounds(n)) == 2
    big = mojoutm.conversion.MAX_CHUNKS * mojoutm.conversion.MIN_ELEMENTS_PER_CHUNK
    assert len(mojoutm.conversion._chunk_bounds(big)) == mojoutm.conversion.MAX_CHUNKS


# --------------------------------------------------------------------------
# the single output allocation
# --------------------------------------------------------------------------

# Reading a buffer's address costs about 1.3 us, which is more than the
# conversion of a single point, so every buffer a call needs comes from one
# allocation and the call looks up one address and slices it by a fixed stride.
# That makes the stride arithmetic a correctness concern, not just a saving:
# an output pointed at past the end of the allocation is a heap overflow that
# returns the right answer and corrupts whatever follows.


def test_out_buffers_are_adjacent_views_of_one_allocation():
    for n in (0, 1, 2, 7, 1000):
        buf, a, b = mojoutm.conversion._out_buffers(n)
        assert a.dtype == b.dtype == np.float64
        assert a.shape == b.shape == (n,)
        assert a.ctypes.data == buf.ctypes.data
        assert b.ctypes.data == buf.ctypes.data + n * 8
        # the zone number and the zone letter are the two scalars past them
        assert buf[2 * n:2 * n + 1].ctypes.data == buf.ctypes.data + 2 * n * 8
        letter = buf[2 * n + 1:2 * n + 2].view(np.int64)
        assert letter.ctypes.data == buf.ctypes.data + (2 * n + 1) * 8


def test_outputs_do_not_overlap_each_other():
    """The two outputs sit side by side, so a kernel that wrote one element too
    far would land in the other instead of corrupting unrelated memory."""
    rng = np.random.default_rng(4711)
    for n in (1, 7, 8, 33, 8192, 8193, 20_000):
        lat, lon = random_points(n, rng)
        easting, northing, zone_number, zone_letter = mojoutm.from_latlon(lat, lon)
        assert easting.shape == northing.shape == (n,)
        assert not np.shares_memory(easting, northing)
        their_e, their_n, _, _ = utm.from_latlon(lat, lon)
        assert np.array_equal(easting, their_e)
        assert np.array_equal(northing, their_n)
        assert (zone_number, zone_letter) == (utm.from_latlon(lat, lon)[2],
                                             utm.from_latlon(lat, lon)[3])


def test_outputs_survive_the_returning_frame_going_away():
    """The views are returned rather than the allocation, so the allocation has
    to outlive them. Dropping every local reference must not free the buffer
    the caller is holding a window into."""
    rng = np.random.default_rng(99)
    n = 30_000
    lat, lon = random_points(n, rng)
    easting, northing, _, _ = mojoutm.from_latlon(lat, lon)
    their_e, their_n, _, _ = utm.from_latlon(lat, lon)
    gc.collect()
    assert np.array_equal(easting, their_e)
    assert np.array_equal(northing, their_n)
    # writing through the view must still reach the allocation it came from
    easting[0] = 1234.5
    assert easting[0] == 1234.5


def test_scalar_call_returns_the_same_type_as_upstream():
    """A scalar in gives a scalar out. Upstream indexes a 1-element numpy array
    and gets a `np.float64` back, so that is the type to match, not `float`."""
    theirs = utm.from_latlon(51.2, 7.5)
    mine = mojoutm.from_latlon(51.2, 7.5)
    assert type(mine[0]) is type(theirs[0])
    assert type(mine[1]) is type(theirs[1])
    assert isinstance(mine[0], float) and np.ndim(mine[0]) == 0
    assert mine[2:] == theirs[2:]
    their_lat, their_lon = utm.to_latlon(theirs[0], theirs[1], theirs[2],
                                          zone_letter=theirs[3])
    lat, lon = mojoutm.to_latlon(mine[0], mine[1], mine[2], zone_letter=mine[3])
    assert type(lat) is type(their_lat)
    assert np.ndim(lat) == 0 and np.ndim(lon) == 0


def test_to_latlon_outputs_do_not_overlap():
    rng = np.random.default_rng(2024)
    for n in (1, 7, 8, 33, 8192, 8193, 20_000):
        easting = rng.integers(100_000, 1_000_000, n).astype(np.float64)
        northing = rng.integers(0, 10_000_001, n).astype(np.float64)
        mine_lat, mine_lon = mojoutm.to_latlon(easting, northing, 33, northern=True)
        their_lat, their_lon = utm.to_latlon(easting, northing, 33, northern=True)
        assert mine_lat.shape == mine_lon.shape == (n,)
        assert not np.shares_memory(mine_lat, mine_lon)
        assert np.array_equal(mine_lat, their_lat)
        assert np.array_equal(mine_lon, their_lon)


def test_prepare_reports_errors_before_any_chunk_runs():
    # A point that is out of range has to raise even though the conversion
    # itself would happily compute a number for it.
    n = mojoutm.conversion.MIN_ELEMENTS_PER_CHUNK * 4
    lat = np.full(n, 10.0)
    lon = np.full(n, 10.0)
    lat[-1] = 90.0
    assert_same_failure(
        outcome(mojoutm.from_latlon, lat, lon),
        outcome(utm.from_latlon, lat, lon),
    )
    easting = np.full(n, 500000.0)
    northing = np.full(n, 5000000.0)
    easting[0] = 10.0
    assert_same_failure(
        outcome(mojoutm.to_latlon, easting, northing, 33, northern=True),
        outcome(utm.to_latlon, easting, northing, 33, northern=True),
    )

# --------------------------------------------------------------------------
# zone helpers
# --------------------------------------------------------------------------


def test_latlon_to_zone_number_parity():
    rng = np.random.default_rng(2024)
    lat = np.round(rng.uniform(-80.0, 84.0, 5000), 6)
    lon = np.round(rng.uniform(-360.0, 360.0, 5000), 6)
    for la, lo in zip(lat, lon):
        assert mojoutm.latlon_to_zone_number(float(la), float(lo)) == \
            utm.latlon_to_zone_number(float(la), float(lo))


def test_latlon_to_zone_number_special_zones():
    # Norway, Svalbard, the antimeridian, and both edges of each special band.
    cases = [
        (60.0, 5.0, 32),
        (55.999999, 5.0, 31),
        (64.0, 5.0, 31),
        (56.0, 2.999999, 31),
        (56.0, 12.0, 33),
        (79.0, 5.0, 31),
        (79.0, 15.0, 33),
        (79.0, 25.0, 35),
        (79.0, 38.0, 37),
        (79.0, 45.0, 38),
        (71.999999, 5.0, 31),
        (79.0, -0.000001, 30),
    ]
    for latitude, longitude, expected in cases:
        assert mojoutm.latlon_to_zone_number(latitude, longitude) == expected
        assert utm.latlon_to_zone_number(latitude, longitude) == expected
        assert mojoutm.latlon_to_zone_number(latitude, longitude + 360) == \
            utm.latlon_to_zone_number(latitude, longitude + 360)
        assert mojoutm.latlon_to_zone_number(latitude, longitude - 360) == \
            utm.latlon_to_zone_number(latitude, longitude - 360)


def test_latitude_to_zone_letter_parity():
    for i in range(-80_000, 84_001):
        latitude = i / 1000.0
        assert mojoutm.latitude_to_zone_letter(latitude) == utm.latitude_to_zone_letter(latitude)
    assert mojoutm.latitude_to_zone_letter(-80.0001) is None
    assert mojoutm.latitude_to_zone_letter(84.0001) is None


def test_latitude_to_zone_letter_array_uses_first_element():
    lat = np.array([10.0, 70.0, -79.0])
    assert mojoutm.latitude_to_zone_letter(lat) == utm.latitude_to_zone_letter(lat) == "P"


def test_latlon_to_zone_number_array_uses_first_element():
    lat = np.array([10.0, 70.0])
    lon = np.array([20.0, 5.0])
    assert mojoutm.latlon_to_zone_number(lat, lon) == utm.latlon_to_zone_number(lat, lon)


def test_zone_number_to_central_longitude_parity():
    for zone_number in range(1, 61):
        assert mojoutm.zone_number_to_central_longitude(zone_number) == \
            utm.zone_number_to_central_longitude(zone_number)
    for zone_number in (0, 61, -1, 100):
        assert_same_failure(
            outcome(mojoutm.zone_number_to_central_longitude, zone_number),
            outcome(utm.zone_number_to_central_longitude, zone_number),
        )


def test_zone_letter_to_central_latitude_parity():
    for letter in "CDEFGHJKLMNPQRSTUVWX":
        assert mojoutm.zone_letter_to_central_latitude(letter) == \
            utm.zone_letter_to_central_latitude(letter)
    for letter in ("A", "I", "O", "Y", "Z"):
        assert_same_failure(
            outcome(mojoutm.zone_letter_to_central_latitude, letter),
            outcome(utm.zone_letter_to_central_latitude, letter),
        )
    assert mojoutm.zone_letter_to_central_latitude("X") == 78


def test_check_valid_zone_parity():
    for zone_number in range(0, 62):
        for letter in (None, "C", "I", "X", "N"):
            assert_same_failure(
                outcome(mojoutm.check_valid_zone, zone_number, letter),
                outcome(utm.check_valid_zone, zone_number, letter),
            )
    assert mojoutm.check_valid_zone(33, "N") is None


# --------------------------------------------------------------------------
# error parity
# --------------------------------------------------------------------------


BAD_CASES = [
    ("from_latlon latitude too high", mojoutm.from_latlon, utm.from_latlon, (85.0, 0.0), {}),
    ("from_latlon latitude too low", mojoutm.from_latlon, utm.from_latlon, (-81.0, 0.0), {}),
    ("from_latlon longitude too high", mojoutm.from_latlon, utm.from_latlon, (0.0, 181.0), {}),
    ("from_latlon longitude too low", mojoutm.from_latlon, utm.from_latlon, (0.0, -181.0), {}),
    ("from_latlon forced zone out of range", mojoutm.from_latlon, utm.from_latlon,
     (0.0, 0.0), {"force_zone_number": 61}),
    ("from_latlon forced letter and northern", mojoutm.from_latlon, utm.from_latlon,
     (0.0, 0.0), {"force_zone_letter": "N", "force_northern": True}),
    ("to_latlon no letter no northern", mojoutm.to_latlon, utm.to_latlon,
     (500000.0, 5000000.0, 33), {}),
    ("to_latlon letter and northern", mojoutm.to_latlon, utm.to_latlon,
     (500000.0, 5000000.0, 33), {"zone_letter": "N", "northern": True}),
    ("to_latlon easting too low", mojoutm.to_latlon, utm.to_latlon,
     (99_999.0, 5000000.0, 33), {"northern": True}),
    ("to_latlon easting too high", mojoutm.to_latlon, utm.to_latlon,
     (1_000_000.0, 5000000.0, 33), {"northern": True}),
    ("to_latlon northing negative", mojoutm.to_latlon, utm.to_latlon,
     (500000.0, -1.0, 33), {"northern": True}),
    ("to_latlon northing too high", mojoutm.to_latlon, utm.to_latlon,
     (500000.0, 10_000_001.0, 33), {"northern": True}),
    ("to_latlon bad zone number", mojoutm.to_latlon, utm.to_latlon,
     (500000.0, 5000000.0, 61), {"northern": True}),
    ("to_latlon bad zone letter", mojoutm.to_latlon, utm.to_latlon,
     (500000.0, 5000000.0, 33), {"zone_letter": "A"}),
]


@pytest.mark.parametrize("name,mine,theirs,args,kwargs", BAD_CASES,
                         ids=[c[0] for c in BAD_CASES])
def test_error_parity(name, mine, theirs, args, kwargs):
    assert_same_failure(outcome(mine, *args, **kwargs), outcome(theirs, *args, **kwargs))


def test_strict_bound_edge_parity():
    # upstream's easting bound is closed at the bottom and half-open at the top;
    # its northing bound is closed at both ends.
    assert_same_failure(
        outcome(mojoutm.to_latlon, 100_000.0, 0.0, 33, northern=True),
        outcome(utm.to_latlon, 100_000.0, 0.0, 33, northern=True),
    )
    # The top of the range is the pole, where the inverse series has run out;
    # both implementations agree on latitude exactly and on longitude to
    # within a small tolerance there.
    mine = outcome(mojoutm.to_latlon, 999_999.0, 10_000_000.0, 33, northern=True)
    theirs = outcome(utm.to_latlon, 999_999.0, 10_000_000.0, 33, northern=True)
    assert mine[0] == theirs[0] == "ok"
    assert mine[1][0] == theirs[1][0]
    assert abs(mine[1][1] - theirs[1][1]) < 1e-3
    assert_same_failure(
        outcome(mojoutm.to_latlon, 1_000_000.0, 10_000_000.0, 33, northern=True),
        outcome(utm.to_latlon, 1_000_000.0, 10_000_000.0, 33, northern=True),
    )
    assert_same_failure(
        outcome(mojoutm.to_latlon, 100_000.0, 10_000_001.0, 33, northern=True),
        outcome(utm.to_latlon, 100_000.0, 10_000_001.0, 33, northern=True),
    )
    assert_same_failure(
        outcome(mojoutm.to_latlon, 100_000.0, -1.0, 33, northern=True),
        outcome(utm.to_latlon, 100_000.0, -1.0, 33, northern=True),
    )


def test_strict_false_ignores_bounds():
    easting = np.array([50_000.0, 1_000_000.0])
    northing = np.array([-5.0, 20_000_000.0])
    mine = mojoutm.to_latlon(easting, northing, 33, northern=True, strict=False)
    theirs = utm.to_latlon(easting, northing, 33, northern=True, strict=False)
    assert np.array_equal(mine[0], theirs[0])
    assert np.allclose(mine[1], theirs[1], rtol=0, atol=1e-6)


def test_array_bound_error_parity():
    easting = np.array([500000.0, 50000.0])
    northing = np.full(2, 5000000.0)
    assert_same_failure(
        outcome(mojoutm.to_latlon, easting, northing, 33, northern=True),
        outcome(utm.to_latlon, easting, northing, 33, northern=True),
    )
    lat = np.array([10.0, 90.0])
    lon = np.array([10.0, 10.0])
    assert_same_failure(
        outcome(mojoutm.from_latlon, lat, lon),
        outcome(utm.from_latlon, lat, lon),
    )


def test_mixed_signs_error_parity():
    lat = np.array([-10.0, 10.0])
    lon = np.array([10.0, 10.0])
    assert_same_failure(
        outcome(mojoutm.from_latlon, lat, lon),
        outcome(utm.from_latlon, lat, lon),
    )


def test_mixed_signs_allowed_when_forced():
    lat = np.array([-10.0, 10.0])
    lon = np.array([10.0, 10.0])
    mine = mojoutm.from_latlon(lat, lon, force_northern=True)
    theirs = utm.from_latlon(lat, lon, force_northern=True)
    assert np.array_equal(mine[0], theirs[0]), "easting must be bit-identical"
    assert np.allclose(mine[1], theirs[1], rtol=0, atol=1e-6), \
        "the -10000000 hemisphere offset must not move a micrometre"


def test_latitude_exactly_at_zone_band_edges():
    lat = np.array([0.0, 84.0, 1.0, 7.999999, 8.0])
    lon = np.full(5, 5.0)
    mine = mojoutm.from_latlon(lat, lon)
    theirs = utm.from_latlon(lat, lon)
    assert mine[2:] == theirs[2:]
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 4, 5, 8, 9, 8191, 8192, 8193, 20_000])
def test_nan_is_rejected_like_upstream(n):
    """numpy propagates NaN through `np.min`/`np.max`, so a NaN anywhere
    fails the bounds check. Mojo's `min`/`max` drop it, so the reduction has
    to notice it separately; the sizes above straddle the SIMD vector width
    and the chunking threshold, so the vector loop, its scalar tail and both
    chunked paths are all covered."""
    easting = np.full(n, 500000.0)
    easting[n - 1] = np.nan
    northing = np.full(n, 1000.0)
    assert_same_failure(
        outcome(mojoutm.to_latlon, easting, northing, 31, zone_letter="V"),
        outcome(utm.to_latlon, easting, northing, 31, zone_letter="V"),
    )

    lat = np.full(n, 51.2)
    lat[0] = np.nan
    lon = np.full(n, 7.5)
    assert_same_failure(
        outcome(mojoutm.from_latlon, lat, lon),
        outcome(utm.from_latlon, lat, lon),
    )

    lat2 = np.full(n, 51.2)
    lon2 = np.full(n, 7.5)
    lon2[n - 1] = np.nan
    assert_same_failure(
        outcome(mojoutm.from_latlon, lat2, lon2),
        outcome(utm.from_latlon, lat2, lon2),
    )


def test_nan_passes_when_no_bounds_check_runs():
    """`strict=False` skips the only whole-array check, so a NaN must reach
    the arithmetic and come back as a NaN coordinate, not be rejected."""
    easting = np.array([500000.0, np.nan])
    northing = np.array([1000.0, 2000.0])
    mine = mojoutm.to_latlon(easting, northing, 31, zone_letter="V", strict=False)
    theirs = utm.to_latlon(easting, northing, 31, zone_letter="V", strict=False)
    assert np.isnan(mine[0][1]) and np.isnan(theirs[0][1])
    assert np.isnan(mine[1][1]) and np.isnan(theirs[1][1])


@pytest.mark.parametrize("zone_number", [1.5, 32.7, 59.5, 60.0, 1.0])
def test_float_zone_number_is_not_narrowed(zone_number):
    """Upstream carries a float zone number through its arithmetic untouched.
    Narrowing it to an int would silently answer for a different zone."""
    mine = mojoutm.zone_number_to_central_longitude(zone_number)
    theirs = utm.zone_number_to_central_longitude(zone_number)
    assert mine == theirs
    assert type(mine) is type(theirs)

    assert_same_failure(
        outcome(mojoutm.check_valid_zone, zone_number, "U"),
        outcome(utm.check_valid_zone, zone_number, "U"),
    )
    easting, northing, zn, _ = mojoutm.from_latlon(51.2, 7.5, force_zone_number=zone_number)
    teasting, tnorthing, tzn, _ = utm.from_latlon(51.2, 7.5, force_zone_number=zone_number)
    assert easting == teasting and northing == tnorthing
    assert zn == tzn


def test_integer_zone_number_still_returns_an_integer():
    # upstream's arithmetic is int-in-int-out, so the wrapper must not hand
    # back a float for an int argument.
    assert isinstance(mojoutm.zone_number_to_central_longitude(32), int)
    assert isinstance(utm.zone_number_to_central_longitude(32), int)


def test_nan_zone_number_is_out_of_range():
    assert_same_failure(
        outcome(mojoutm.from_latlon, 51.2, 7.5, force_zone_number=np.nan),
        outcome(utm.from_latlon, 51.2, 7.5, force_zone_number=np.nan),
    )


@pytest.mark.parametrize("zone_number", [0.0, 60.5, -1.5, 100.0])
def test_out_of_range_float_zone_number(zone_number):
    assert_same_failure(
        outcome(mojoutm.check_valid_zone, zone_number, "U"),
        outcome(utm.check_valid_zone, zone_number, "U"),
    )


# Upstream compares zone letters as whole strings, so a multi-character
# string is in range exactly when its first character is, and is then used
# by that character. `'XY'` is out of range, `'VV'` is in range and northern,
# and `'I'`/`'O'` are excluded only as whole strings.
@pytest.mark.parametrize(
    "zone_letter", ["VV", "IX", "XY", "IV", "OO", "I", "O", "C", "X", "m", "n"]
)
def test_multi_character_zone_letter_matches_upstream(zone_letter):
    assert_same_failure(
        outcome(mojoutm.check_valid_zone, 31, zone_letter),
        outcome(utm.check_valid_zone, 31, zone_letter),
    )
    mine = outcome(mojoutm.to_latlon, 500000.0, 1000.0, 31, zone_letter=zone_letter)
    theirs = outcome(utm.to_latlon, 500000.0, 1000.0, 31, zone_letter=zone_letter)
    if mine[0] == "raise":
        assert_same_failure(mine, theirs)
    else:
        assert np.array_equal(mine[1][0], theirs[1][0])
        assert np.array_equal(mine[1][1], theirs[1][1])
    assert_same_failure(
        outcome(mojoutm.zone_letter_to_central_latitude, zone_letter),
        outcome(utm.zone_letter_to_central_latitude, zone_letter),
    )


def test_zone_number_is_validated_before_the_letter():
    # upstream's `check_valid_zone` checks the number first; a wrapper that
    # rejected the letter while marshalling would report the other one.
    assert_same_failure(
        outcome(mojoutm.check_valid_zone, 99, "XY"),
        outcome(utm.check_valid_zone, 99, "XY"),
    )
    assert_same_failure(
        outcome(mojoutm.to_latlon, 500000.0, 1000.0, 99, zone_letter="XY"),
        outcome(utm.to_latlon, 500000.0, 1000.0, 99, zone_letter="XY"),
    )


@pytest.mark.parametrize("zone_letter", [None, ""])
def test_falsy_zone_letter_matches_upstream(zone_letter):
    # upstream's `check_valid_zone` skips a falsy letter, and
    # `zone_letter_to_central_latitude` dies on `None.upper()` before it can
    # validate anything. Both are quirks, but they are the observable ones.
    assert_same_failure(
        outcome(mojoutm.zone_letter_to_central_latitude, zone_letter),
        outcome(utm.zone_letter_to_central_latitude, zone_letter),
    )
    assert_same_failure(
        outcome(mojoutm.check_valid_zone, 31, zone_letter),
        outcome(utm.check_valid_zone, 31, zone_letter),
    )
    assert_same_failure(
        outcome(mojoutm.to_latlon, 500000.0, 1000.0, 31, zone_letter=zone_letter),
        outcome(utm.to_latlon, 500000.0, 1000.0, 31, zone_letter=zone_letter),
    )
