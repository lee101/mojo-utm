#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist" "$repo_dir/build"
# `contract=fast` is the compiler default and it fuses `a + b*c` into an FMA
# across statements, which makes results differ from numpy in the last bit.
# numpy does not contract, so neither does this: parity with upstream is then
# bit-exact rather than merely close.
mojo build --emit shared-lib --fp-mode contract=off -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-utm.so"
# `python/mojoutm/_lib.py` loads `build/capi.so` and rebuilds it when it is
# older than `src/`, so keep that copy fresh instead of paying a second build.
cp "$repo_dir/dist/libmojo-utm.so" "$repo_dir/build/capi.so"
