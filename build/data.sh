#!/usr/bin/env bash
# Regenerate the numeric tables from an upstream ephem sdist.
#
# They are checked in, so this is only needed when moving to a new upstream
# release. Point EPHEM_SDIST at an unpacked `ephem-X.Y.Z` directory:
#
#     curl -LO https://files.pythonhosted.org/.../ephem-4.2.1.tar.gz
#     tar xzf ephem-4.2.1.tar.gz
#     EPHEM_SDIST=$PWD/ephem-4.2.1 pixi run data
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdist="${1:-${EPHEM_SDIST:-}}"
if [ -z "$sdist" ]; then
    echo "usage: build/data.sh <path-to-ephem-sdist>" >&2
    exit 2
fi
python "$repo_dir/tools/gen_tables.py" "$sdist"
