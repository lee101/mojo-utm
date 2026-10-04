"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

from mojoutm.error import OutOfRangeError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")

# `MOJO_UTM_LIB` points at an already-built `capi.so`. Two cases need it:
# an ordinary (non-editable) `pip install`, where the package is copied into
# site-packages and `ROOT` no longer names the checkout that holds `src/` and
# `build/`; and any server that must never pay a `mojo build` inside a request
# and would rather ship the kernel next to the deploy. Set it and nothing is
# ever compiled at import or call time.
LIB = os.environ.get("MOJO_UTM_LIB") or os.path.join(ROOT, "build", "capi.so")

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "mu_to_latlon": ([I, I, I, F, I, I, I, I, I], I),
    "mu_to_latlon_prepare": ([I, I, I, F, I, I, I], I),
    "mu_to_latlon_kernel": ([I, I, I, F, I, I, I, I], I),
    "mu_from_latlon": ([I, I, I, F, I, I, I, I, I, I], I),
    "mu_from_latlon_prepare": ([I, I, I, F, I, I, I, I], I),
    "mu_from_latlon_kernel": ([I, I, I, F, I, I, I, I], I),
    "mu_latlon_to_zone_number": ([F, F], I),
    "mu_latitude_to_zone_letter": ([F], I),
    "mu_check_valid_zone": ([F, I], I),
    "mu_check_valid_zone_number": ([F], I),
    "mu_check_valid_zone_letter": ([I], I),
    "mu_zone_number_to_central_longitude": ([F, I], I),
    "mu_zone_letter_to_central_latitude": ([I, I], I),
}

# Sentinel values for `None`, matching `src/ported.mojo`. A zone letter is the
# uppercase ASCII code of the character, or 0 when the argument was not given;
# `northern` / `force_northern` are 0/1, or -1 when not given. A zone number of
# -1 means `force_zone_number` was not given.
UNSET_LETTER = 0
UNSET_BOOL = -1

# ASCII codes of the bands `letter_code` substitutes, and of the two upstream
# excludes. See `letter_code`.
LETTER_I = ord("I")
LETTER_O = ord("O")
LETTER_M = ord("M")
LETTER_P = ord("P")

# One below 'C', so the kernel's own range check rejects it. See `letter_code`.
OUT_OF_RANGE_LETTER = ord("B")

# Status code -> (exception class, message). Every message is upstream's,
# character for character.
ERRORS: dict[int, tuple[type[Exception], str]] = {
    1: (
        OutOfRangeError,
        "easting out of range (must be between 100,000 m and 999,999 m)",
    ),
    2: (
        OutOfRangeError,
        "northing out of range (must be between 0 m and 10,000,000 m)",
    ),
    3: (
        OutOfRangeError,
        "zone number out of range (must be between 1 and 60)",
    ),
    4: (
        OutOfRangeError,
        "zone letter out of range (must be between C and X)",
    ),
    5: (
        OutOfRangeError,
        "latitude out of range (must be between 80 deg S and 84 deg N)",
    ),
    6: (
        OutOfRangeError,
        "longitude out of range (must be between 180 deg W and 180 deg E)",
    ),
    7: (ValueError, "latitudes must all have the same sign"),
    8: (ValueError, "either zone_letter or northern needs to be set"),
    9: (ValueError, "set either zone_letter or northern, but not both"),
    10: (ValueError, "set either force_zone_letter or force_northern, but not both"),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_UTM_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_UTM_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `build/capi.so` if it is missing or stale."""
    if os.environ.get("MOJO_UTM_LIB") and os.path.exists(LIB) and not force:
        # An explicitly supplied library is taken as given: the sources it was
        # built from need not be present at all, so there is nothing to
        # compare mtimes against.
        return LIB
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        # `ROOT` is the directory above the installed package, which is
        # site-packages here, not a checkout - so naming it helps nobody. Say
        # what to do instead.
        raise BuildError(
            "no Mojo sources and no prebuilt library: this is an installed "
            "copy, which cannot compile. Build the library in a checkout "
            "(`pixi run build`) and point MOJO_UTM_LIB at the resulting "
            "capi.so, or run from a checkout where src/ is present."
        )
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        # See build/build.sh: contraction is what keeps us bit-exact with numpy.
        "build", "--emit", "shared-lib", "--fp-mode", "contract=off", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib



def addr(a: np.ndarray) -> int:
    """The buffer address a flat `float64` array hands to the C ABI.

    Reading this is not free: `a.ctypes` builds a fresh `_ctypes` proxy object
    on every attribute access, about 1.3 us, and `__array_interface__` builds
    its dict every time, about 2.0 us. Both are slower than the conversion of a
    single point, so the callers that need several addresses pass one buffer
    that holds all of them rather than paying per array.
    """
    return a.ctypes.data


def raise_status(status: int) -> None:
    exc, message = ERRORS[status]
    raise exc(message)


def letter_code(zone_letter) -> int:
    """Upstream's `None`-able zone letter as the ASCII code the kernel takes.

    Upstream compares the *whole* string (`'C' <= zone_letter.upper() <= 'X'`,
    excluding `'I'` and `'O'`), so a multi-character string is judged as a
    string and then used by its first character: `'VV'` is in range and
    northern, `'XY'` is out of range. The kernel works in single-character
    codes, so the first character decides both the validity and the hemisphere.

    An invalid letter is reported as a code the kernel's own range check
    rejects rather than raised here. `check_valid_zone` validates the number
    before the letter, and raising in this function would report the letter
    first - upstream's order is the opposite. `OUT_OF_RANGE_LETTER` is one
    below `'C'`, so the kernel rejects it with the zone-letter status.

    `'I'` and `'O'` are excluded by upstream only as whole strings, but are
    legal as the first character of a longer one (`'IV'`), where the kernel's
    own check would reject them. They map to the nearest band with the same
    hemisphere, which is all the kernel reads the code for.
    """
    if not zone_letter:
        return UNSET_LETTER
    upper = str(zone_letter).upper()
    if not ("C" <= upper <= "X") or upper in ("I", "O"):
        return OUT_OF_RANGE_LETTER
    code = ord(upper[0])
    if code == LETTER_I:
        return LETTER_M
    if code == LETTER_O:
        return LETTER_P
    return code


def letter_code_exact(zone_letter) -> int:
    """`letter_code` for the callers that need a real band, not a hemisphere.

    `zone_letter_to_central_latitude` looks the letter up in the band table, so
    a multi-character string must fail there the way upstream's
    `ZONE_LETTERS.index` fails - after the range check, which runs first and
    raises `OutOfRangeError` for a string that is out of range.
    """
    if zone_letter is None:
        # upstream calls `zone_letter.upper()` before validating anything, so
        # `None` dies on the attribute, not on the range check.
        raise AttributeError("'NoneType' object has no attribute 'upper'")
    upper = str(zone_letter).upper()
    if not ("C" <= upper <= "X") or upper in ("I", "O"):
        raise OutOfRangeError(
            "zone letter out of range (must be between C and X)"
        )
    if len(upper) != 1:
        raise ValueError("substring not found")
    return ord(upper)



def bool_flag(value) -> int:
    """Upstream's `None`-able boolean as the sentinel triple the kernel takes."""
    if value is None:
        return UNSET_BOOL
    return 1 if value else 0


def main() -> int:
    """`python -m mojoutm._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
