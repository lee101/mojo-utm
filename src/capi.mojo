"""The C ABI the Python bindings call through.

Buffers cross as `Int` addresses: `@export` refuses parametric functions, and a
pointer with an inferred origin is parametric. Nothing here allocates and
nothing here decides anything: each wrapper turns upstream's `None`-able
arguments into the sentinels `ported.mojo` uses and hands the call straight on.
"""

from ported import (
    IPtr,
    OK,
    Ptr,
    check_valid_zone,
    check_valid_zone_letter,
    check_valid_zone_number,
    from_latlon,
    from_latlon_kernel,
    from_latlon_prepare,
    latitude_to_zone_letter,
    latlon_to_zone_number,
    to_latlon,
    to_latlon_kernel,
    to_latlon_prepare,
    zone_letter_to_central_latitude,
    zone_number_to_central_longitude,
)


# `to_latlon` and `from_latlon` are each split in two: a `prepare` that makes
# every whole-array decision and returns a status, and a `kernel` that converts
# one contiguous range and touches nothing else. A caller with a big enough
# array runs the kernel over disjoint ranges at the same time; a caller with a
# small one uses the single-crossing `mu_to_latlon` / `mu_from_latlon` and never
# pays for a second crossing. Every form produces identical bytes.


@export("mu_to_latlon")
def mu_to_latlon(
    easting: Int,
    northing: Int,
    n: Int,
    zone_number: Float64,
    zone_letter: Int,
    northern: Int,
    strict: Int,
    out_latitude: Int,
    out_longitude: Int,
) abi("C") -> Int:
    return to_latlon(
        Ptr(unsafe_from_address=easting),
        Ptr(unsafe_from_address=northing),
        n,
        zone_number,
        zone_letter,
        northern,
        strict,
        Ptr(unsafe_from_address=out_latitude),
        Ptr(unsafe_from_address=out_longitude),
    )


@export("mu_to_latlon_prepare")
def mu_to_latlon_prepare(
    easting: Int,
    northing: Int,
    n: Int,
    zone_number: Float64,
    zone_letter: Int,
    northern: Int,
    strict: Int,
) abi("C") -> Int:
    return to_latlon_prepare(
        Ptr(unsafe_from_address=easting),
        Ptr(unsafe_from_address=northing),
        n,
        zone_number,
        zone_letter,
        northern,
        strict,
    )


@export("mu_to_latlon_kernel")
def mu_to_latlon_kernel(
    easting: Int,
    northing: Int,
    n: Int,
    zone_number: Float64,
    zone_letter: Int,
    northern: Int,
    out_latitude: Int,
    out_longitude: Int,
) abi("C") -> Int:
    to_latlon_kernel(
        Ptr(unsafe_from_address=easting),
        Ptr(unsafe_from_address=northing),
        n,
        zone_number,
        zone_letter,
        northern,
        Ptr(unsafe_from_address=out_latitude),
        Ptr(unsafe_from_address=out_longitude),
    )
    return OK


@export("mu_from_latlon")
def mu_from_latlon(
    latitude: Int,
    longitude: Int,
    n: Int,
    force_zone_number: Float64,
    force_zone_letter: Int,
    force_northern: Int,
    out_easting: Int,
    out_northing: Int,
    out_zone_number: Int,
    out_zone_letter: Int,
) abi("C") -> Int:
    return from_latlon(
        Ptr(unsafe_from_address=latitude),
        Ptr(unsafe_from_address=longitude),
        n,
        force_zone_number,
        force_zone_letter,
        force_northern,
        Ptr(unsafe_from_address=out_easting),
        Ptr(unsafe_from_address=out_northing),
        Ptr(unsafe_from_address=out_zone_number),
        IPtr(unsafe_from_address=out_zone_letter),
    )


@export("mu_from_latlon_prepare")
def mu_from_latlon_prepare(
    latitude: Int,
    longitude: Int,
    n: Int,
    force_zone_number: Float64,
    force_zone_letter: Int,
    force_northern: Int,
    out_zone_number: Int,
    out_zone_letter: Int,
) abi("C") -> Int:
    return from_latlon_prepare(
        Ptr(unsafe_from_address=latitude),
        Ptr(unsafe_from_address=longitude),
        n,
        force_zone_number,
        force_zone_letter,
        force_northern,
        Ptr(unsafe_from_address=out_zone_number),
        IPtr(unsafe_from_address=out_zone_letter),
    )


@export("mu_from_latlon_kernel")
def mu_from_latlon_kernel(
    latitude: Int,
    longitude: Int,
    n: Int,
    zone_number: Float64,
    zone_letter: Int,
    force_northern: Int,
    out_easting: Int,
    out_northing: Int,
) abi("C") -> Int:
    from_latlon_kernel(
        Ptr(unsafe_from_address=latitude),
        Ptr(unsafe_from_address=longitude),
        n,
        zone_number,
        zone_letter,
        force_northern,
        Ptr(unsafe_from_address=out_easting),
        Ptr(unsafe_from_address=out_northing),
    )
    return OK


@export("mu_latlon_to_zone_number")
def mu_latlon_to_zone_number(latitude: Float64, longitude: Float64) abi("C") -> Int:
    return latlon_to_zone_number(latitude, longitude)


@export("mu_latitude_to_zone_letter")
def mu_latitude_to_zone_letter(latitude: Float64) abi("C") -> Int:
    return latitude_to_zone_letter(latitude)


@export("mu_check_valid_zone")
def mu_check_valid_zone(zone_number: Float64, zone_letter: Int) abi("C") -> Int:
    return check_valid_zone(zone_number, zone_letter)


@export("mu_check_valid_zone_number")
def mu_check_valid_zone_number(zone_number: Float64) abi("C") -> Int:
    return check_valid_zone_number(zone_number)


@export("mu_check_valid_zone_letter")
def mu_check_valid_zone_letter(zone_letter: Int) abi("C") -> Int:
    return check_valid_zone_letter(zone_letter)


@export("mu_zone_number_to_central_longitude")
def mu_zone_number_to_central_longitude(zone_number: Float64, out_value: Int) abi("C") -> Int:
    # upstream raises from inside this function; the raise is hoisted into the
    # wrapper so the exported function stays non-raising.
    var status = check_valid_zone_number(zone_number)
    if status != OK:
        return status
    Ptr(unsafe_from_address=out_value).unsafe_store(0, zone_number_to_central_longitude(zone_number))
    return OK


@export("mu_zone_letter_to_central_latitude")
def mu_zone_letter_to_central_latitude(zone_letter: Int, out_value: Int) abi("C") -> Int:
    var status = check_valid_zone_letter(zone_letter)
    if status != OK:
        return status
    IPtr(unsafe_from_address=out_value).unsafe_store(0, zone_letter_to_central_latitude(zone_letter))
    return OK
