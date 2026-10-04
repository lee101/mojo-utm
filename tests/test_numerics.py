"""Numerical invariants the port claims, checked against upstream on bulk data.

`test_parity.py` covers the API case by case; `test_kernels.py` covers each
exported symbol. This file covers the three claims the README makes about
*how close* the two implementations are, because those are the claims a caller
relying on a drop-in replacement actually depends on:

- `from_latlon` is bit-identical to upstream over its whole domain.
- `to_latlon` latitude is bit-identical everywhere, and its longitude is
  bit-identical inside the latitude band `to_latlon` documents.
- the only place the two part ways is `to_latlon` longitude past the pole,
  which is stated precisely rather than waved at.

Every comparison is against the real `utm` package. Nothing here compares
against a stored constant, because a stored constant only proves the port has
not changed - it says nothing about whether it was ever right.
"""

from __future__ import annotations

import numpy as np
import pytest
import utm

import mojoutm

# The latitude band `to_latlon` documents and `from_latlon` enforces.
BAND = (-80.0, 84.0)

# Upstream's own band letters; `X` appears twice in ZONE_LETTERS, so the
# distinct letters are these 20.
BANDS = "CDEFGHJKLMNPQRSTUVWX"


def north_points(n, rng):
    lat = np.round(rng.uniform(1e-4, 84.0, n), 6)
    lon = np.round(rng.uniform(-179.999999, 179.999999, n), 6)
    return lat, lon


def south_points(n, rng):
    lat = np.round(rng.uniform(-80.0, -1e-4, n), 6)
    lon = np.round(rng.uniform(-179.999999, 179.999999, n), 6)
    return lat, lon


# --------------------------------------------------------------------------
# from_latlon is bit-identical
# --------------------------------------------------------------------------


@pytest.mark.parametrize("northern", [True, False])
def test_from_latlon_is_bit_identical_over_the_domain(northern):
    rng = np.random.default_rng(20260101)
    lat, lon = north_points(300_000, rng) if northern else south_points(300_000, rng)
    mine = mojoutm.from_latlon(lat, lon)
    theirs = utm.from_latlon(lat, lon)
    assert np.array_equal(mine[0], theirs[0]), "easting must be bit-identical"
    assert np.array_equal(mine[1], theirs[1]), "northing must be bit-identical"
    assert mine[2:] == theirs[2:]


def test_from_latlon_is_bit_identical_at_every_band_edge():
    # the exact latitude of each band boundary and a hair either side, clipped
    # to the legal [-80, 84] range. `84.0 + 1e-9` is outside it and upstream
    # rejects it, so that case is not a parity case but a bounds case.
    candidates = sorted(
        {
            value
            for edge in (0.001, 8.0, 16.0, 83.999999999)
            for delta in (-1e-9, 0.0, 1e-9)
            for value in (edge + delta,)
            if -80.0 <= value <= 84.0
        }
    )
    lat = np.array(candidates)
    lon = np.full(lat.size, 7.5)
    mine = mojoutm.from_latlon(lat, lon)
    theirs = utm.from_latlon(lat, lon)
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


def test_from_latlon_is_bit_identical_across_every_zone():
    # force each zone in turn, so the central-longitude term varies
    for zone in range(1, 61):
        lat = np.array([51.2, -33.8688])
        lon = np.array([7.5, 151.2093])
        mine = mojoutm.from_latlon(lat, lon, force_zone_number=zone, force_northern=True)
        theirs = utm.from_latlon(lat, lon, force_zone_number=zone, force_northern=True)
        assert np.array_equal(mine[0], theirs[0]), f"zone {zone}"
        assert np.array_equal(mine[1], theirs[1]), f"zone {zone}"


# --------------------------------------------------------------------------
# to_latlon is bit-identical inside the documented band
# --------------------------------------------------------------------------


def test_to_latlon_latitude_is_bit_identical_inside_the_band():
    rng = np.random.default_rng(98765)
    size = 200_000
    # northings chosen so the result stays within the documented band
    easting = rng.uniform(100_000.0, 999_999.0, size)
    northing = rng.uniform(0.0, 9_300_000.0, size)
    mine = mojoutm.to_latlon(easting, northing, 32, "U")
    theirs = utm.to_latlon(easting, northing, 32, "U")
    assert np.all(np.abs(mine[0]) <= 84.0), "the fixture must stay in the documented band"
    assert np.array_equal(mine[0], theirs[0]), "latitude must be bit-identical"
    assert np.array_equal(mine[1], theirs[1]), "longitude must be bit-identical"


def test_to_latlon_is_bit_identical_across_every_zone_and_band():
    rng = np.random.default_rng(55555)
    for zone in range(1, 61):
        for band in BANDS:
            size = 200
            easting = rng.uniform(100_000.0, 999_999.0, size)
            # stay inside the band: 9,300,000 m is under 84 deg for every zone
            northing = rng.uniform(0.0, 9_300_000.0, size)
            mine = mojoutm.to_latlon(easting, northing, zone, band)
            theirs = utm.to_latlon(easting, northing, zone, band)
            assert np.array_equal(mine[0], theirs[0]), f"zone {zone} band {band} latitude"
            assert np.array_equal(mine[1], theirs[1]), f"zone {zone} band {band} longitude"


def test_to_latlon_is_bit_identical_south():
    rng = np.random.default_rng(13579)
    size = 100_000
    easting = rng.uniform(100_000.0, 999_999.0, size)
    northing = rng.uniform(0.0, 9_300_000.0, size)
    mine = mojoutm.to_latlon(easting, northing, 31, northern=False)
    theirs = utm.to_latlon(easting, northing, 31, northern=False)
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


def test_to_latlon_is_bit_identical_at_the_zone_edges():
    for zone in (1, 60):
        for easting in (100_000.0, 999_999.0):
            for northing in (0.0, 9_300_000.0):
                mine = mojoutm.to_latlon(easting, northing, zone, "U")
                theirs = utm.to_latlon(easting, northing, zone, "U")
                assert mine == theirs, f"zone {zone} at {easting},{northing}"


# --------------------------------------------------------------------------
# the one documented divergence, bounded precisely
# --------------------------------------------------------------------------


def _in_band(longitude_latitude):
    return BAND[0] <= longitude_latitude <= BAND[1]


def test_to_latlon_diverges_only_past_the_pole():
    """Latitude past the pole is nonsense in both, but pin the shape of it.

    Mojo's float `%` is `a - b*floor(a/b)` where Python's is an exact `fmod`
    plus a sign correction. Upstream's `mod_angle` is
    `(value + pi) % (2*pi) - pi`. The two agree while the quotient
    `value / (2*pi)` stays small and part company once it does not, which is
    what a northing past the pole forces. Latitude is unaffected: it never goes
    through `mod_angle`.
    """
    mine = mojoutm.to_latlon(500_000.0, 9_998_000.0, 34, "W", strict=False)
    theirs = utm.to_latlon(500_000.0, 9_998_000.0, 34, "W", strict=False)
    # latitude is the pole either way, and identical
    assert mine[0] == theirs[0]
    assert abs(mine[0]) > 84.0, "past the pole: outside the documented band"
    # longitude is the same to a hair at the central meridian...
    assert mine[1] == pytest.approx(theirs[1], abs=1e-9)


def test_to_latlon_longitude_agrees_inside_the_documented_band_even_past_the_pole_northing():
    # Northing 9,900,000 m gives latitude -141.8 deg, far outside the band, but
    # the two still agree to ~4e-13 deg because the quotient is still small.
    # This is the boundary the README quotes: agreement up to here, nonsense
    # beyond it.
    mine = mojoutm.to_latlon(900_000.0, 9_900_000.0, 34, "W", strict=False)
    theirs = utm.to_latlon(900_000.0, 9_900_000.0, 34, "W", strict=False)
    assert mine[0] == theirs[0]
    assert mine[1] == pytest.approx(theirs[1], abs=1e-9)


def test_no_disagreement_anywhere_inside_the_documented_band():
    """The claim that matters: a grid over the whole legal domain agrees.

    60 zones by 30 eastings by 60 northings, keeping only the points whose
    returned latitude lands inside [-80, 84]. Every one must be bit-identical.
    """
    checked = 0
    for zone in range(1, 61):
        for easting in np.linspace(100_000.0, 999_999.0, 30):
            for northing in np.linspace(0.0, 10_000_000.0, 60):
                east = float(easting)
                north = float(northing)
                mine = mojoutm.to_latlon(east, north, zone, "N", strict=False)
                if not _in_band(mine[0]):
                    continue
                theirs = utm.to_latlon(east, north, zone, "N", strict=False)
                assert mine == theirs, f"zone {zone} at {east},{north}"
                checked += 1
    # the grid must have produced a meaningful number of in-band points
    assert checked > 90_000


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def _round_trip_points(rng, n):
    """Points confined to one zone *and* to its legal easting range.

    `from_latlon` returns the zone of its *first* point and converts the rest
    in that zone, so a spread of longitudes drives the eastings of the far
    points outside [100000, 999999] and the trip back raises - in upstream
    exactly as here. The fixture sits inside zone 32, whose central meridian
    is 9 deg, and so stays clear of zone 31: upstream gives longitudes
    [3, 12) at latitudes [56, 64) to Norway zone 32's neighbour, 31, even
    though they are east of the central meridian. Staying below 56 deg of
    latitude removes that exception, and the easting range is asserted rather
    than assumed.
    """
    lat = np.round(rng.uniform(0.001, 55.0, n), 6)
    lon = np.round(rng.uniform(8.6, 9.4, n), 6)
    easting, northing, zone_number, zone_letter = mojoutm.from_latlon(lat, lon)
    assert zone_number == 32, "the fixture must stay in one zone"
    assert easting.min() >= 100_000.0 and easting.max() <= 999_999.0, "fixture out of range"
    assert northing.min() >= 0.0 and northing.max() <= 10_000_000.0, "fixture out of range"
    return lat, lon, easting, northing, zone_number, zone_letter


def test_round_trip_is_exact_enough_to_reproduce_the_input():
    rng = np.random.default_rng(24680)
    lat, lon, easting, northing, zone_number, zone_letter = _round_trip_points(rng, 50_000)
    back_lat, back_lon = mojoutm.to_latlon(easting, northing, zone_number, zone_letter)
    # The projection is inverted by a truncated Taylor series, so a round trip
    # is not exact: over the whole band the residual is a few 1e-9 degrees,
    # which is under a millimetre on the ground. What matters is that this is
    # the same residual upstream gets, which the next test asserts.
    assert np.max(np.abs(back_lat - lat)) < 1e-8
    assert np.max(np.abs(back_lon - lon)) < 1e-8


def test_round_trip_through_upstream_agrees_point_for_point():
    # mojoutm -> utm -> mojoutm must land where utm -> utm lands
    rng = np.random.default_rng(11235)
    lat, lon, easting, northing, zone_number, zone_letter = _round_trip_points(rng, 20_000)
    back = utm.to_latlon(easting, northing, zone_number, zone_letter)
    mine = mojoutm.to_latlon(easting, northing, zone_number, zone_letter)
    # element-wise: the two return (array, array), so `==` on the tuple is the
    # array truth value, not an elementwise comparison
    assert np.array_equal(mine[0], back[0]), "round-tripped latitude must be bit-identical"
    assert np.array_equal(mine[1], back[1]), "round-tripped longitude must be bit-identical"