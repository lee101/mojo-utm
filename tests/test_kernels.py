"""Direct tests of the exported C ABI, one per `@export` in `src/capi.mojo`.

`tests/test_parity.py` reaches the kernels only through `mojoutm`'s Python API.
That is enough to prove the port is right, but not enough to prove each symbol
is right: a kernel that is wired up correctly and computes the wrong answer
still ships green, because nothing ever calls it under its own name and looks at
its raw output. Every `@export` in `src/` has a test here that calls that symbol
directly and asserts on the values it returns.

The shared library is loaded once, at import, and a missing library is a hard
error rather than a skip: a skip here would let every test in this file go
unrun while the suite still reported green. Build it with `pixi run build`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import utm

from mojoutm import _lib

# Raises rather than skips: an unbuilt library means nothing in this file was
# actually exercised, and reporting that as "passed" is exactly the failure mode
# these tests exist to prevent.
lib = _lib.lib()
addr = _lib.addr

OK = 0
ERR_EASTING_RANGE = 1
ERR_NORTHING_RANGE = 2
ERR_ZONE_NUMBER_RANGE = 3
ERR_ZONE_LETTER_RANGE = 4
ERR_LATITUDE_RANGE = 5
ERR_LONGITUDE_RANGE = 6
ERR_MIXED_SIGNS = 7
ERR_NEEDS_LETTER_OR_NORTHERN = 8
ERR_LETTER_XOR_NORTHERN = 9
ERR_FORCE_LETTER_XOR_NORTHERN = 10

UNSET_LETTER = 0
UNSET_BOOL = -1

# Sentinels from the Python side of the same split.
NONE_ZONE_NUMBER = float(_lib.UNSET_BOOL)


def f64(*values):
    """A contiguous float64 buffer the kernel may write into."""
    return np.array(values, dtype=np.float64)


def i64(*values):
    return np.array(values, dtype=np.int64)


def test_shared_library_is_the_compiled_one():
    """Guard the premise of this file: these tests run against `dist/*.so`.

    `_lib.lib()` is loaded at import time and raises if the library cannot be
    produced, so the tests above already ran against real machine code. This
    pins down *which* file was loaded: `MOJO_UTM_LIB` overrides it, so the path
    is checked rather than assumed.
    """
    assert _lib.lib().mu_to_latlon
    assert _lib.LIB.endswith(".so")
    assert os.path.isfile(_lib.LIB), _lib.LIB
    assert os.path.getsize(_lib.LIB) > 0


# --------------------------------------------------------------------------
# mu_latlon_to_zone_number
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latitude, longitude, expected",
    [
        (51.2, 7.5, 32),
        (0.0, 0.0, 31),
        (0.0, 180.0, 1),
        (0.0, -180.0, 1),
        (-33.8688, 151.2093, 56),
        # the Norway exception
        (56.0, 3.0, 32),
        (63.999, 11.999, 32),
        (55.999, 5.0, 31),
        (64.0, 5.0, 31),
        # the four Svalbard exceptions
        (72.0, 0.0, 31),
        (75.0, 8.999, 31),
        (75.0, 9.0, 33),
        (75.0, 20.999, 33),
        (75.0, 21.0, 35),
        (75.0, 32.999, 35),
        (75.0, 33.0, 37),
        (75.0, 41.999, 37),
        (75.0, 42.0, 38),
        (71.999, 5.0, 31),
        (84.0, 5.0, 31),
    ],
)
def test_mu_latlon_to_zone_number(latitude, longitude, expected):
    assert lib.mu_latlon_to_zone_number(latitude, longitude) == expected
    assert lib.mu_latlon_to_zone_number(latitude, longitude) == utm.latlon_to_zone_number(
        latitude, longitude
    )


def test_mu_latlon_to_zone_number_sweeps_upstream():
    rng = np.random.default_rng(20240501)
    for _ in range(20000):
        latitude = float(rng.uniform(-80.0, 84.0))
        longitude = float(rng.uniform(-360.0, 360.0))
        assert lib.mu_latlon_to_zone_number(latitude, longitude) == utm.latlon_to_zone_number(
            latitude, longitude
        )


# --------------------------------------------------------------------------
# mu_latitude_to_zone_letter
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latitude, expected",
    [
        (-80.0, "C"),
        (-79.999, "C"),
        (-72.001, "C"),
        (-72.0, "D"),
        (-76.0, "C"),
        (0.0, "N"),
        (7.999, "N"),
        (8.0, "P"),
        (51.2, "U"),
        (83.999, "X"),
        (84.0, "X"),
    ],
)
def test_mu_latitude_to_zone_letter(latitude, expected):
    # upstream is `ZONE_LETTERS[int(latitude + 80) >> 3]`, a band of 8 degrees
    # starting at -80, so -76 is still inside the first band.
    code = lib.mu_latitude_to_zone_letter(latitude)
    assert chr(code) == expected
    assert code == ord(expected)
    assert expected == utm.latitude_to_zone_letter(latitude)


@pytest.mark.parametrize("latitude", [84.0000001, 85.0, 90.0, 1000.0, -80.0000001])
def test_mu_latitude_to_zone_letter_out_of_band_is_unset(latitude):
    # upstream returns None here; the kernel spells that 0, and the wrapper
    # turns it back into None.
    assert lib.mu_latitude_to_zone_letter(latitude) == UNSET_LETTER
    assert utm.latitude_to_zone_letter(latitude) is None


def test_mu_latitude_to_zone_letter_covers_every_band():
    seen = {chr(lib.mu_latitude_to_zone_letter(lat)) for lat in np.arange(-80.0, 84.0, 1.0)}
    assert seen == set("CDEFGHJKLMNPQRSTUVWX")


# --------------------------------------------------------------------------
# mu_check_valid_zone_number
# --------------------------------------------------------------------------


@pytest.mark.parametrize("zone_number", [1, 2, 31, 32, 59, 60, 32.5, 1.0])
def test_mu_check_valid_zone_number_accepts(zone_number):
    assert lib.mu_check_valid_zone_number(float(zone_number)) == OK
    utm.zone_number_to_central_longitude(zone_number)  # must not raise


@pytest.mark.parametrize("zone_number", [0, 0.999, 60.001, 61, -1, -1000, 1e9])
def test_mu_check_valid_zone_number_rejects(zone_number):
    assert lib.mu_check_valid_zone_number(float(zone_number)) == ERR_ZONE_NUMBER_RANGE
    with pytest.raises(utm.OutOfRangeError):
        utm.zone_number_to_central_longitude(zone_number)


def test_mu_check_valid_zone_number_rejects_nan():
    # `not 1.0 <= nan <= 60.0` is true, so the kernel rejects; upstream's
    # `not 1 <= nan <= 60` is the same chain and raises too.
    assert lib.mu_check_valid_zone_number(float("nan")) == ERR_ZONE_NUMBER_RANGE
    with pytest.raises(utm.OutOfRangeError):
        utm.zone_number_to_central_longitude(float("nan"))


# --------------------------------------------------------------------------
# mu_check_valid_zone_letter
# --------------------------------------------------------------------------


@pytest.mark.parametrize("letter", "CDEFGHJKLMNPQRSTUVWX")
def test_mu_check_valid_zone_letter_accepts(letter):
    assert lib.mu_check_valid_zone_letter(ord(letter)) == OK
    utm.zone_letter_to_central_latitude(letter)  # must not raise


@pytest.mark.parametrize("letter", ["A", "B", "I", "O", "Y", "Z", "a"])
def test_mu_check_valid_zone_letter_rejects(letter):
    assert lib.mu_check_valid_zone_letter(ord(letter)) == ERR_ZONE_LETTER_RANGE
    with pytest.raises(utm.OutOfRangeError):
        utm.zone_letter_to_central_latitude(letter)


def test_mu_check_valid_zone_letter_rejects_below_c():
    assert lib.mu_check_valid_zone_letter(ord("B")) == ERR_ZONE_LETTER_RANGE
    assert lib.mu_check_valid_zone_letter(0) == ERR_ZONE_LETTER_RANGE


# --------------------------------------------------------------------------
# mu_check_valid_zone
# --------------------------------------------------------------------------


def test_mu_check_valid_zone_accepts():
    for letter in "CUNX":
        assert lib.mu_check_valid_zone(32.0, ord(letter)) == OK
    # an unset letter is skipped, exactly as upstream's `if zone_letter:`
    assert lib.mu_check_valid_zone(32.0, UNSET_LETTER) == OK


def test_mu_check_valid_zone_validates_number_before_letter():
    # upstream calls check_valid_zone_number first, so a bad number is
    # reported even when the letter is also bad.
    assert lib.mu_check_valid_zone(0.0, ord("A")) == ERR_ZONE_NUMBER_RANGE
    assert lib.mu_check_valid_zone(61.0, ord("A")) == ERR_ZONE_NUMBER_RANGE
    assert lib.mu_check_valid_zone(32.0, ord("A")) == ERR_ZONE_LETTER_RANGE


def test_mu_check_valid_zone_matches_upstream_over_the_cross_product():
    number_message = "zone number out of range (must be between 1 and 60)"
    letter_message = "zone letter out of range (must be between C and X)"
    # 'U' is a legal band, so only 'I', 'O' and 'Z' are invalid letters here
    rejected = set()
    for zone_number in (0, 1, 32, 60, 61, 32.5):
        for letter in "CIXOUZ":
            mine = lib.mu_check_valid_zone(float(zone_number), ord(letter))
            try:
                utm.check_valid_zone(zone_number, letter)
                theirs = OK
            except utm.OutOfRangeError as exc:
                theirs = next(
                    code for code, (_, message) in _lib.ERRORS.items() if message == str(exc)
                )
            assert mine == theirs, f"{zone_number}/{letter}"
            if mine != OK:
                assert _lib.ERRORS[mine][1] in (number_message, letter_message)
                rejected.add((zone_number, letter))
    # the rule, stated rather than enumerated: a bad number fails whatever the
    # letter, and a good number still fails on an excluded band
    bad_number = {0, 61}
    bad_letter = set("IOZ")
    assert rejected == {
        (zone_number, letter)
        for zone_number in (0, 1, 32, 60, 61, 32.5)
        for letter in "CIXOUZ"
        if zone_number in bad_number or letter in bad_letter
    }


# --------------------------------------------------------------------------
# mu_zone_number_to_central_longitude
# --------------------------------------------------------------------------


def test_mu_zone_number_to_central_longitude():
    out = f64(0.0)
    for zone_number in (1, 2, 31, 32, 33, 59, 60):
        assert lib.mu_zone_number_to_central_longitude(float(zone_number), addr(out)) == OK
        assert out[0] == utm.zone_number_to_central_longitude(zone_number)


def test_mu_zone_number_to_central_longitude_endpoints():
    out = f64(0.0)
    assert lib.mu_zone_number_to_central_longitude(1.0, addr(out)) == OK
    assert out[0] == -177.0
    assert lib.mu_zone_number_to_central_longitude(60.0, addr(out)) == OK
    assert out[0] == 177.0


def test_mu_zone_number_to_central_longitude_fractional_zone():
    # upstream's arithmetic is float-in-float-out, so 32.5 lands mid-band.
    out = f64(0.0)
    assert lib.mu_zone_number_to_central_longitude(32.5, addr(out)) == OK
    assert out[0] == utm.zone_number_to_central_longitude(32.5)


@pytest.mark.parametrize("zone_number", [0.0, 61.0, -1.0, float("nan")])
def test_mu_zone_number_to_central_longitude_rejects(zone_number):
    out = f64(0.0)
    assert (
        lib.mu_zone_number_to_central_longitude(zone_number, addr(out))
        == ERR_ZONE_NUMBER_RANGE
    )


def test_mu_zone_number_to_central_longitude_does_not_write_on_error():
    # the wrapper checks first and returns before the store, so a rejected zone
    # leaves the caller's buffer exactly as it found it.
    out = f64(-12345.0)
    assert lib.mu_zone_number_to_central_longitude(0.0, addr(out)) == ERR_ZONE_NUMBER_RANGE
    assert out[0] == -12345.0


# --------------------------------------------------------------------------
# mu_zone_letter_to_central_latitude
# --------------------------------------------------------------------------


def test_mu_zone_letter_to_central_latitude():
    out = i64(0)
    for index, letter in enumerate("CDEFGHJKLMNPQRSTUVWX"):
        assert lib.mu_zone_letter_to_central_latitude(ord(letter), addr(out)) == OK
        assert out[0] == utm.zone_letter_to_central_latitude(letter)
        if letter == "X":
            assert out[0] == 78
        else:
            assert out[0] == -76 + index * 8


def test_mu_zone_letter_to_central_latitude_x_is_the_exception():
    out = i64(0)
    assert lib.mu_zone_letter_to_central_latitude(ord("X"), addr(out)) == OK
    assert out[0] == 78


@pytest.mark.parametrize("letter", ["I", "O", "A", "B", "Y", "Z"])
def test_mu_zone_letter_to_central_latitude_rejects(letter):
    out = i64(0)
    assert (
        lib.mu_zone_letter_to_central_latitude(ord(letter), addr(out))
        == ERR_ZONE_LETTER_RANGE
    )


def test_mu_zone_letter_to_central_latitude_does_not_write_on_error():
    out = i64(-999)
    assert (
        lib.mu_zone_letter_to_central_latitude(ord("I"), addr(out))
        == ERR_ZONE_LETTER_RANGE
    )
    assert out[0] == -999


# --------------------------------------------------------------------------
# mu_to_latlon_prepare
# --------------------------------------------------------------------------


def _to_latlon_prepare(easting, northing, zone_number, zone_letter, northern, strict):
    return lib.mu_to_latlon_prepare(
        addr(easting), addr(northing), easting.size,
        float(zone_number), zone_letter, northern, strict,
    )


def test_mu_to_latlon_prepare_accepts_a_valid_input():
    status = _to_latlon_prepare(f64(500000.0), f64(5000000.0), 32, ord("U"), UNSET_BOOL, 1)
    assert status == OK


def test_mu_to_latlon_prepare_requires_letter_or_northern():
    assert (
        _to_latlon_prepare(f64(500000.0), f64(5000000.0), 32, UNSET_LETTER, UNSET_BOOL, 1)
        == ERR_NEEDS_LETTER_OR_NORTHERN
    )


def test_mu_to_latlon_prepare_rejects_letter_and_northern_together():
    assert (
        _to_latlon_prepare(f64(500000.0), f64(5000000.0), 32, ord("U"), 1, 1)
        == ERR_LETTER_XOR_NORTHERN
    )


def test_mu_to_latlon_prepare_checks_easting_upper_bound_strictly():
    # 1000000 is excluded: upstream's in_bounds uses upper_strict here.
    assert _to_latlon_prepare(f64(1000000.0), f64(5000000.0), 32, ord("U"), UNSET_BOOL, 1) == (
        ERR_EASTING_RANGE
    )
    assert _to_latlon_prepare(f64(999999.0), f64(5000000.0), 32, ord("U"), UNSET_BOOL, 1) == OK


def test_mu_to_latlon_prepare_checks_easting_lower_bound():
    assert _to_latlon_prepare(f64(99999.0), f64(5000000.0), 32, ord("U"), UNSET_BOOL, 1) == (
        ERR_EASTING_RANGE
    )
    assert _to_latlon_prepare(f64(100000.0), f64(5000000.0), 32, ord("U"), UNSET_BOOL, 1) == OK


def test_mu_to_latlon_prepare_checks_northing_inclusively():
    assert _to_latlon_prepare(f64(500000.0), f64(-1.0), 32, ord("U"), UNSET_BOOL, 1) == (
        ERR_NORTHING_RANGE
    )
    assert _to_latlon_prepare(f64(500000.0), f64(10000001.0), 32, ord("U"), UNSET_BOOL, 1) == (
        ERR_NORTHING_RANGE
    )
    assert _to_latlon_prepare(f64(500000.0), f64(10000000.0), 32, ord("U"), UNSET_BOOL, 1) == OK


def test_mu_to_latlon_prepare_validates_the_zone_last():
    # upstream checks bounds, then the zone; the status must report that order.
    assert _to_latlon_prepare(f64(0.0), f64(5000000.0), 99, ord("A"), UNSET_BOOL, 1) == (
        ERR_EASTING_RANGE
    )
    assert _to_latlon_prepare(f64(500000.0), f64(5000000.0), 99, ord("A"), UNSET_BOOL, 1) == (
        ERR_ZONE_NUMBER_RANGE
    )
    assert _to_latlon_prepare(f64(500000.0), f64(5000000.0), 32, ord("A"), UNSET_BOOL, 1) == (
        ERR_ZONE_LETTER_RANGE
    )


def test_mu_to_latlon_prepare_skips_bounds_when_not_strict():
    # strict=0 must accept coordinates the strict path rejects.
    assert _to_latlon_prepare(f64(0.0), f64(-5000000.0), 32, ord("U"), UNSET_BOOL, 0) == OK


def test_mu_to_latlon_prepare_sees_nan():
    # numpy's np.min/np.max propagate NaN so the bounds check fails; the
    # kernel must agree even though min()/max() would drop it.
    assert _to_latlon_prepare(f64(np.nan), f64(5000000.0), 32, ord("U"), UNSET_BOOL, 1) == (
        ERR_EASTING_RANGE
    )
    assert _to_latlon_prepare(f64(500000.0), f64(np.nan), 32, ord("U"), UNSET_BOOL, 1) == (
        ERR_NORTHING_RANGE
    )


def test_mu_to_latlon_prepare_sees_nan_beyond_the_vector_width():
    # past simd_width_of lanes, so the NaN can only be found by the scalar tail.
    size = 4096
    easting = np.full(size, 500000.0)
    easting[size - 1] = np.nan
    assert _to_latlon_prepare(easting, np.full(size, 5000000.0), 32, ord("U"), UNSET_BOOL, 1) == (
        ERR_EASTING_RANGE
    )


# --------------------------------------------------------------------------
# mu_to_latlon
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "zone_number, zone_letter",
    [(32, "U"), (31, "N"), (33, "S"), (60, "X"), (1, "C")],
)
def test_mu_to_latlon_is_bit_exact_with_upstream(zone_number, zone_letter):
    easting = f64(340000.0)
    northing = f64(5710000.0)
    latitude = f64(0.0)
    longitude = f64(0.0)
    status = lib.mu_to_latlon(
        addr(easting), addr(northing), 1, float(zone_number),
        ord(zone_letter), UNSET_BOOL, 1, addr(latitude), addr(longitude),
    )
    assert status == OK
    ref_lat, ref_lon = utm.to_latlon(340000.0, 5710000.0, zone_number, zone_letter)
    assert latitude[0] == ref_lat
    assert longitude[0] == ref_lon


def test_mu_to_latlon_honours_the_northern_flag():
    easting = f64(340000.0)
    northing = f64(5710000.0)
    lat_n = f64(0.0)
    lat_s = f64(0.0)
    lon = f64(0.0)
    assert lib.mu_to_latlon(
        addr(easting), addr(northing), 1, 32.0, UNSET_LETTER, 1, 1, addr(lat_n), addr(lon)
    ) == OK
    assert lib.mu_to_latlon(
        addr(easting), addr(northing), 1, 32.0, UNSET_LETTER, 0, 1, addr(lat_s), addr(lon)
    ) == OK
    # upstream subtracts 10000000 from the northing when northern is false, so
    # the same northing names two points 10000000 m apart on the meridian. At
    # this latitude that is about 90.26 degrees, not exactly 90.
    assert lat_n[0] == utm.to_latlon(340000.0, 5710000.0, 32, northern=True)[0]
    assert lat_s[0] == utm.to_latlon(340000.0, 5710000.0, 32, northern=False)[0]
    assert lat_n[0] - lat_s[0] == pytest.approx(90.26, abs=0.1)


def test_mu_to_latlon_zone_letter_decides_the_hemisphere():
    # upstream reads the hemisphere off the letter when one is given, so a
    # northern band and a southern band must disagree about the same northing
    # even though the northern flag is left unset in both.
    easting = f64(400000.0)
    northing = f64(4000000.0)
    north = f64(0.0)
    south = f64(0.0)
    lon = f64(0.0)
    lib.mu_to_latlon(
        addr(easting), addr(northing), 1, 32.0, ord("N"), UNSET_BOOL, 1, addr(north), addr(lon)
    )
    lib.mu_to_latlon(
        addr(easting), addr(northing), 1, 32.0, ord("C"), UNSET_BOOL, 1, addr(south), addr(lon)
    )
    assert north[0] == utm.to_latlon(400000.0, 4000000.0, 32, "N")[0]
    assert south[0] == utm.to_latlon(400000.0, 4000000.0, 32, "C")[0]
    assert north[0] > 0.0
    assert south[0] < 0.0
    # bands within the same hemisphere must not differ
    p_band = f64(0.0)
    lib.mu_to_latlon(
        addr(easting), addr(northing), 1, 32.0, ord("P"), UNSET_BOOL, 1, addr(p_band), addr(lon)
    )
    assert p_band[0] == north[0]


def test_mu_to_latlon_returns_the_status_and_writes_nothing_on_error():
    easting = f64(0.0)  # out of range
    northing = f64(5000000.0)
    latitude = f64(-999.0)
    longitude = f64(-999.0)
    status = lib.mu_to_latlon(
        addr(easting), addr(northing), 1, 32.0, ord("U"), UNSET_BOOL, 1,
        addr(latitude), addr(longitude),
    )
    assert status == ERR_EASTING_RANGE
    assert latitude[0] == -999.0
    assert longitude[0] == -999.0


def test_mu_to_latlon_is_bit_exact_across_a_bulk_array():
    rng = np.random.default_rng(987654)
    size = 50000
    easting = rng.uniform(150000.0, 850000.0, size)
    northing = rng.uniform(100000.0, 9300000.0, size)
    latitude = np.empty(size)
    longitude = np.empty(size)
    status = lib.mu_to_latlon(
        addr(easting), addr(northing), size, 32.0, ord("U"), UNSET_BOOL, 1,
        addr(latitude), addr(longitude),
    )
    assert status == OK
    ref_lat, ref_lon = utm.to_latlon(easting, northing, 32, "U")
    assert np.array_equal(latitude, ref_lat)
    assert np.array_equal(longitude, ref_lon)


# --------------------------------------------------------------------------
# mu_to_latlon_kernel
# --------------------------------------------------------------------------


def test_mu_to_latlon_kernel_reproduces_mu_to_latlon_bit_for_bit():
    rng = np.random.default_rng(13579)
    size = 4096
    easting = rng.uniform(150000.0, 850000.0, size)
    northing = rng.uniform(100000.0, 9300000.0, size)

    lat_whole = np.empty(size)
    lon_whole = np.empty(size)
    assert lib.mu_to_latlon_prepare(
        addr(easting), addr(northing), size, 32.0, ord("U"), UNSET_BOOL, 1
    ) == OK
    lib.mu_to_latlon_kernel(
        addr(easting), addr(northing), size, 32.0, ord("U"), UNSET_BOOL,
        addr(lat_whole), addr(lon_whole),
    )

    # the same data through the one-shot entry point
    lat_single = np.empty(size)
    lon_single = np.empty(size)
    assert lib.mu_to_latlon(
        addr(easting), addr(northing), size, 32.0, ord("U"), UNSET_BOOL, 1,
        addr(lat_single), addr(lon_single),
    ) == OK
    assert np.array_equal(lat_whole, lat_single)
    assert np.array_equal(lon_whole, lon_single)

    # and through disjoint chunks, which is what the Python wrapper does above
    # the chunk threshold
    lat_chunked = np.empty(size)
    lon_chunked = np.empty(size)
    for start in range(0, size, 1000):
        stop = min(start + 1000, size)
        lib.mu_to_latlon_kernel(
            addr(easting[start:stop]), addr(northing[start:stop]), stop - start,
            32.0, ord("U"), UNSET_BOOL,
            addr(lat_chunked[start:stop]), addr(lon_chunked[start:stop]),
        )
    assert np.array_equal(lat_chunked, lat_whole)
    assert np.array_equal(lon_chunked, lon_whole)


def test_mu_to_latlon_kernel_matches_upstream_bit_for_bit():
    rng = np.random.default_rng(24680)
    size = 4096
    easting = rng.uniform(150000.0, 850000.0, size)
    northing = rng.uniform(100000.0, 9300000.0, size)
    latitude = np.empty(size)
    longitude = np.empty(size)
    lib.mu_to_latlon_kernel(
        addr(easting), addr(northing), size, 32.0, ord("U"), UNSET_BOOL,
        addr(latitude), addr(longitude),
    )
    ref_lat, ref_lon = utm.to_latlon(easting, northing, 32, "U")
    assert np.array_equal(latitude, ref_lat)
    assert np.array_equal(longitude, ref_lon)


def test_mu_to_latlon_kernel_honours_the_northern_flag():
    easting = np.full(4, 340000.0)
    northing = np.full(4, 5710000.0)
    lat_north = np.empty(4)
    lat_south = np.empty(4)
    lon = np.empty(4)
    lib.mu_to_latlon_kernel(
        addr(easting), addr(northing), 4, 32.0, UNSET_LETTER, 1, addr(lat_north), addr(lon)
    )
    lib.mu_to_latlon_kernel(
        addr(easting), addr(northing), 4, 32.0, UNSET_LETTER, 0, addr(lat_south), addr(lon)
    )
    assert lat_north[0] == utm.to_latlon(340000.0, 5710000.0, 32, northern=True)[0]
    assert lat_south[0] == utm.to_latlon(340000.0, 5710000.0, 32, northern=False)[0]


def test_mu_to_latlon_kernel_does_not_validate():
    # The kernel half makes no whole-array decision, so an out-of-range
    # easting is converted rather than rejected. That is the split's contract:
    # only `prepare` enforces anything.
    easting = f64(0.0)
    northing = f64(5000000.0)
    latitude = f64(0.0)
    longitude = f64(0.0)
    assert lib.mu_to_latlon_kernel(
        addr(easting), addr(northing), 1, 32.0, ord("U"), UNSET_BOOL,
        addr(latitude), addr(longitude),
    ) == OK
    assert np.isfinite(latitude[0]) and np.isfinite(longitude[0])


# --------------------------------------------------------------------------
# mu_from_latlon_prepare
# --------------------------------------------------------------------------


def _from_latlon_prepare(latitude, longitude, force_zone_number, force_zone_letter, force_northern):
    zone_number = f64(0.0)
    zone_letter = i64(0)
    status = lib.mu_from_latlon_prepare(
        addr(latitude), addr(longitude), latitude.size,
        float(force_zone_number), force_zone_letter, force_northern,
        addr(zone_number), addr(zone_letter),
    )
    return status, zone_number, zone_letter


def test_mu_from_latlon_prepare_computes_the_zone():
    status, zone_number, zone_letter = _from_latlon_prepare(
        f64(51.2), f64(7.5), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL
    )
    assert status == OK
    assert zone_number[0] == 32.0
    assert chr(zone_letter[0]) == "U"


def test_mu_from_latlon_prepare_rejects_a_mixed_array_before_reporting_a_zone():
    # upstream runs the mixed-sign check with whatever latitude array it was
    # given and only then computes the zone, so a second point south of the
    # equator fails even though the first point is northern and valid. The
    # kernel must reject in the same order and leave the zone buffers alone.
    latitude = f64(51.2, -33.8688)
    longitude = f64(7.5, 151.2093)
    zone_number = f64(-1.0)
    zone_letter = i64(-1)
    status = lib.mu_from_latlon_prepare(
        addr(latitude), addr(longitude), 2,
        NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL,
        addr(zone_number), addr(zone_letter),
    )
    assert status == ERR_MIXED_SIGNS
    assert zone_number[0] == -1.0
    assert zone_letter[0] == -1
    with pytest.raises(ValueError, match="latitudes must all have the same sign"):
        utm.from_latlon(latitude, longitude)


def test_mu_from_latlon_prepare_uses_the_first_point_only():
    # one hemisphere throughout, so the zone really is the first point's and
    # not the second's
    latitude = f64(51.2, 49.0)
    longitude = f64(7.5, 8.4)
    zone_number = f64(0.0)
    zone_letter = i64(0)
    status = lib.mu_from_latlon_prepare(
        addr(latitude), addr(longitude), 2,
        NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL,
        addr(zone_number), addr(zone_letter),
    )
    assert status == OK
    assert zone_number[0] == 32.0
    assert chr(zone_letter[0]) == "U"
    # the second point is in zone 32 as well but a different letter band, and
    # upstream reports the first point's letter
    assert zone_letter[0] == ord(utm.latitude_to_zone_letter(51.2))


def test_mu_from_latlon_prepare_honours_a_forced_zone_number():
    status, zone_number, _ = _from_latlon_prepare(
        f64(51.2), f64(7.5), 33.0, UNSET_LETTER, UNSET_BOOL
    )
    assert status == OK
    assert zone_number[0] == 33.0


def test_mu_from_latlon_prepare_rejects_a_bad_forced_zone_number():
    assert _from_latlon_prepare(f64(51.2), f64(7.5), 99.0, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_ZONE_NUMBER_RANGE
    )
    assert _from_latlon_prepare(f64(51.2), f64(7.5), 0.0, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_ZONE_NUMBER_RANGE
    )


def test_mu_from_latlon_prepare_checks_latitude_band():
    assert _from_latlon_prepare(f64(84.0001), f64(7.5), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_LATITUDE_RANGE
    )
    assert _from_latlon_prepare(f64(-80.0001), f64(7.5), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_LATITUDE_RANGE
    )
    assert _from_latlon_prepare(f64(84.0), f64(7.5), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == OK
    assert _from_latlon_prepare(f64(-80.0), f64(7.5), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == OK


def test_mu_from_latlon_prepare_checks_longitude_band():
    assert _from_latlon_prepare(f64(51.2), f64(180.001), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_LONGITUDE_RANGE
    )
    assert _from_latlon_prepare(f64(51.2), f64(-180.001), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_LONGITUDE_RANGE
    )
    assert _from_latlon_prepare(f64(51.2), f64(180.0), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == OK
    assert _from_latlon_prepare(f64(51.2), f64(-180.0), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == OK


def test_mu_from_latlon_prepare_rejects_letter_and_northern_together():
    assert _from_latlon_prepare(f64(51.2), f64(7.5), NONE_ZONE_NUMBER, ord("U"), 1)[0] == (
        ERR_FORCE_LETTER_XOR_NORTHERN
    )


def test_mu_from_latlon_prepare_rejects_mixed_signs():
    latitude = f64(-1.0, 1.0)
    longitude = f64(1.0, 1.0)
    assert _from_latlon_prepare(latitude, longitude, NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_MIXED_SIGNS
    )


def test_mu_from_latlon_prepare_allows_mixed_signs_when_forced():
    latitude = f64(-1.0, 1.0)
    longitude = f64(1.0, 1.0)
    assert _from_latlon_prepare(latitude, longitude, NONE_ZONE_NUMBER, UNSET_LETTER, True)[0] == OK
    assert _from_latlon_prepare(latitude, longitude, NONE_ZONE_NUMBER, ord("P"), UNSET_BOOL)[0] == OK


def test_mu_from_latlon_prepare_mixed_signs_beyond_the_vector_width():
    # a negative at the far end of a long array, so the reduction has to see it
    size = 100000
    latitude = np.linspace(-80.0, 84.0, size)
    longitude = np.zeros(size)
    assert _from_latlon_prepare(latitude, longitude, NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_MIXED_SIGNS
    )


def test_mu_from_latlon_prepare_sees_nan():
    assert _from_latlon_prepare(f64(np.nan), f64(7.5), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_LATITUDE_RANGE
    )
    assert _from_latlon_prepare(f64(51.2), f64(np.nan), NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL)[0] == (
        ERR_LONGITUDE_RANGE
    )


def test_mu_from_latlon_prepare_does_not_write_the_zone_on_error():
    zone_number = f64(-1.0)
    zone_letter = i64(-1)
    latitude = f64(90.0)
    longitude = f64(7.5)
    status = lib.mu_from_latlon_prepare(
        addr(latitude), addr(longitude), 1,
        NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL,
        addr(zone_number), addr(zone_letter),
    )
    assert status == ERR_LATITUDE_RANGE
    assert zone_number[0] == -1.0
    assert zone_letter[0] == -1


# --------------------------------------------------------------------------
# mu_from_latlon
# --------------------------------------------------------------------------


def test_mu_from_latlon_is_bit_exact_with_upstream():
    latitude = f64(51.2)
    longitude = f64(7.5)
    easting = f64(0.0)
    northing = f64(0.0)
    zone_number = f64(0.0)
    zone_letter = i64(0)
    status = lib.mu_from_latlon(
        addr(latitude), addr(longitude), 1, NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL,
        addr(easting), addr(northing), addr(zone_number), addr(zone_letter),
    )
    assert status == OK
    ref = utm.from_latlon(51.2, 7.5)
    assert easting[0] == ref[0]
    assert northing[0] == ref[1]
    assert zone_number[0] == ref[2]
    assert chr(zone_letter[0]) == ref[3]


def test_mu_from_latlon_returns_the_forced_zone_verbatim():
    latitude = f64(51.2)
    longitude = f64(7.5)
    easting = f64(0.0)
    northing = f64(0.0)
    zone_number = f64(0.0)
    zone_letter = i64(0)
    status = lib.mu_from_latlon(
        addr(latitude), addr(longitude), 1, 33.5, UNSET_LETTER, UNSET_BOOL,
        addr(easting), addr(northing), addr(zone_number), addr(zone_letter),
    )
    assert status == OK
    assert zone_number[0] == 33.5


def test_mu_from_latlon_returns_the_forced_zone_letter():
    latitude = f64(51.2)
    longitude = f64(7.5)
    easting = f64(0.0)
    northing = f64(0.0)
    zone_number = f64(0.0)
    zone_letter = i64(0)
    status = lib.mu_from_latlon(
        addr(latitude), addr(longitude), 1, NONE_ZONE_NUMBER, ord("P"), UNSET_BOOL,
        addr(easting), addr(northing), addr(zone_number), addr(zone_letter),
    )
    assert status == OK
    assert chr(zone_letter[0]) == "P"


def test_mu_from_latlon_returns_the_status_and_writes_nothing_on_error():
    easting = f64(-1.0)
    northing = f64(-1.0)
    zone_number = f64(-1.0)
    zone_letter = i64(-1)
    latitude = f64(90.0)
    longitude = f64(7.5)
    status = lib.mu_from_latlon(
        addr(latitude), addr(longitude), 1,
        NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL,
        addr(easting), addr(northing), addr(zone_number), addr(zone_letter),
    )
    assert status == ERR_LATITUDE_RANGE
    assert easting[0] == -1.0
    assert northing[0] == -1.0
    assert zone_number[0] == -1.0
    assert zone_letter[0] == -1


def test_mu_from_latlon_is_bit_exact_across_a_bulk_array():
    rng = np.random.default_rng(31415)
    size = 50000
    latitude = np.round(rng.uniform(0.001, 84.0, size), 6)
    longitude = np.round(rng.uniform(-179.999999, 179.999999, size), 6)
    easting = np.empty(size)
    northing = np.empty(size)
    zone_number = f64(0.0)
    zone_letter = i64(0)
    status = lib.mu_from_latlon(
        addr(latitude), addr(longitude), size, NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL,
        addr(easting), addr(northing), addr(zone_number), addr(zone_letter),
    )
    assert status == OK
    ref_easting, ref_northing, _, _ = utm.from_latlon(latitude, longitude)
    assert np.array_equal(easting, ref_easting)
    assert np.array_equal(northing, ref_northing)


# --------------------------------------------------------------------------
# mu_from_latlon_kernel
# --------------------------------------------------------------------------


def _zone_scalars(latitude, longitude):
    zone_number = f64(0.0)
    zone_letter = i64(0)
    assert lib.mu_from_latlon_prepare(
        addr(latitude), addr(longitude), latitude.size,
        NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL,
        addr(zone_number), addr(zone_letter),
    ) == OK
    return zone_number[0], int(zone_letter[0])


def test_mu_from_latlon_kernel_reproduces_mu_from_latlon_bit_for_bit():
    rng = np.random.default_rng(271828)
    size = 4096
    latitude = np.round(rng.uniform(0.001, 84.0, size), 6)
    longitude = np.round(rng.uniform(-5.0, 15.0, size), 6)
    zone_number, zone_letter = _zone_scalars(latitude, longitude)

    easting_whole = np.empty(size)
    northing_whole = np.empty(size)
    lib.mu_from_latlon_kernel(
        addr(latitude), addr(longitude), size, zone_number, zone_letter, UNSET_BOOL,
        addr(easting_whole), addr(northing_whole),
    )

    easting_single = np.empty(size)
    northing_single = np.empty(size)
    assert lib.mu_from_latlon(
        addr(latitude), addr(longitude), size, NONE_ZONE_NUMBER, UNSET_LETTER, UNSET_BOOL,
        addr(easting_single), addr(northing_single), addr(f64(0.0)), addr(i64(0)),
    ) == OK
    assert np.array_equal(easting_whole, easting_single)
    assert np.array_equal(northing_whole, northing_single)

    easting_chunked = np.empty(size)
    northing_chunked = np.empty(size)
    for start in range(0, size, 999):  # not a divisor of 4096, to hit the tail
        stop = min(start + 999, size)
        lib.mu_from_latlon_kernel(
            addr(latitude[start:stop]), addr(longitude[start:stop]), stop - start,
            zone_number, zone_letter, UNSET_BOOL,
            addr(easting_chunked[start:stop]), addr(northing_chunked[start:stop]),
        )
    assert np.array_equal(easting_chunked, easting_whole)
    assert np.array_equal(northing_chunked, northing_whole)


def test_mu_from_latlon_kernel_matches_upstream_bit_for_bit():
    rng = np.random.default_rng(141421)
    size = 4096
    # southern latitudes, so the zone letter is a southern band and the
    # false-northing offset is exercised
    latitude = np.round(rng.uniform(-80.0, -0.001, size), 6)
    longitude = np.round(rng.uniform(3.0, 15.0, size), 6)
    zone_number, zone_letter = _zone_scalars(latitude, longitude)
    easting = np.empty(size)
    northing = np.empty(size)
    lib.mu_from_latlon_kernel(
        addr(latitude), addr(longitude), size, zone_number, zone_letter, UNSET_BOOL,
        addr(easting), addr(northing),
    )
    ref_easting, ref_northing, _, _ = utm.from_latlon(latitude, longitude)
    assert np.array_equal(easting, ref_easting)
    assert np.array_equal(northing, ref_northing)
    assert np.all(northing > 0.0)


def test_mu_from_latlon_kernel_applies_the_southern_offset():
    latitude = np.full(4, -33.8688)
    longitude = np.full(4, 151.2093)
    easting = np.empty(4)
    northing = np.empty(4)
    # 'J' is a southern band, so upstream adds the false-northing offset.
    lib.mu_from_latlon_kernel(
        addr(latitude), addr(longitude), 4, 56.0, ord("J"), UNSET_BOOL,
        addr(easting), addr(northing),
    )
    ref_easting, ref_northing, _, _ = utm.from_latlon(latitude, longitude)
    assert np.array_equal(easting, ref_easting)
    assert np.array_equal(northing, ref_northing)
    assert np.all(northing > 5000000.0)


def test_mu_from_latlon_kernel_honours_force_northern():
    latitude = np.full(4, -33.8688)
    longitude = np.full(4, 151.2093)
    south = np.empty(4)
    north = np.empty(4)
    easting = np.empty(4)
    lib.mu_from_latlon_kernel(
        addr(latitude), addr(longitude), 4, 56.0, UNSET_LETTER, 0, addr(easting), addr(south)
    )
    lib.mu_from_latlon_kernel(
        addr(latitude), addr(longitude), 4, 56.0, UNSET_LETTER, 1, addr(easting), addr(north)
    )
    ref_south = utm.from_latlon(latitude, longitude, force_northern=False)[1]
    ref_north = utm.from_latlon(latitude, longitude, force_northern=True)[1]
    assert np.array_equal(south, ref_south)
    assert np.array_equal(north, ref_north)
    # upstream adds 10000000 to the northing when northern is false
    assert np.allclose(south - north, 10000000.0)


def test_mu_from_latlon_kernel_does_not_validate():
    latitude = f64(90.0)
    longitude = f64(7.5)
    easting = f64(0.0)
    northing = f64(0.0)
    assert lib.mu_from_latlon_kernel(
        addr(latitude), addr(longitude), 1, 32.0, ord("U"), UNSET_BOOL,
        addr(easting), addr(northing),
    ) == OK
    assert np.isfinite(easting[0]) and np.isfinite(northing[0])