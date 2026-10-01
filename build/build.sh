#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

MOJO_JOBS="${MOJO_JOBS:-4}"

# src/ported.mojo is the single compilation unit: every libastro routine the
# port covers, followed by the C ABI the ctypes glue calls through. The
# numeric tables are not built here - see build/data.sh for why, and
# `pixi run data` to regenerate them from an ephem sdist.

# MOJO_JOBS caps the compiler's thread pool (`-j 0` = one thread per core).
# The default of 0 over-allocates: `mojo` spawns ~110 threads up front, which
# exceeds the pids.max of container/cgroup sandboxes and aborts the build with
# `LLVM ERROR: pthread_create failed`. Build cost is ~5s regardless of thread
# count (single compilation unit), so a small bound costs nothing.
mojo build --emit shared-lib --num-threads "$MOJO_JOBS" \
    "$repo_dir/src/ported.mojo" \
    -o "$repo_dir/dist/libmojo-pyephem.so"
