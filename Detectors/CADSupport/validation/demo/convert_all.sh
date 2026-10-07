#!/bin/bash

# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
# Author: Sandro Wenzel <sandro.wenzel@cern.ch>
# Since: 2026-08

# Stage 1 of the integration demo: convert ExcavatorArm three times.
#
#   excavator_arm_exact  : the cascade  CSG -> exact O2BVHSurfaceSolid -> tessellated fallback
#   excavator_arm_tess   : pure tessellation, same mesh precision as the cascade's fallback
#   excavator_arm_coarse : a deliberately degraded tessellation, so that "the two representations agree"
#                   can be told apart from "the instrument cannot see a difference"
#
# Needs a python3 that can import OCC (for example under `alienv enter pythonOCC/latest`), or
# PYOCC set to one.
#
# Usage: convert_all.sh <output-root>
set -u
OUT=${1:?usage: convert_all.sh <output-root>}
GEO=$(cd "$(dirname "$0")/../.." && pwd)
PYOCC=${PYOCC:-python3}
if ! "$PYOCC" -c "import OCC" 2>/dev/null; then
  echo "convert_all.sh: $PYOCC cannot import OCC; load pythonOCC or set PYOCC" >&2
  exit 1
fi

# Mesh precision. --mesh-prec sets linear AND angular deflection to the same value, and
# it behaves as an *angular* knob.
EXCAVATOR_ARM_PREC=${EXCAVATOR_ARM_PREC:-0.1}
COARSE_PREC=${COARSE_PREC:-2.0}

MODEL="$GEO/examples/ExcavatorArm.step"
MATERIALS="$GEO/examples/ExcavatorArm_MATERIALS.csv"
NIST="$GEO/tools/g4_nist_database/G4_NIST_DB.json"
run() {  # run <tag> <args...>
  local tag=$1; shift
  mkdir -p "$OUT/conv/$tag" "$OUT/logs"
  echo "=== $tag ==="
  /usr/bin/time -v "$PYOCC" "$GEO/tools/O2_CADtoTGeo.py" "$@" \
      --output-folder "$OUT/conv/$tag" -o geom.C --g4-nist-json "$NIST" \
      > "$OUT/logs/conv_$tag.log" 2>&1
  echo "  exit=$? -> $OUT/logs/conv_$tag.log"
}

run excavator_arm_exact  "$MODEL" --csg auto --exact-surfaces auto --materials-csv "$MATERIALS"
run excavator_arm_tess   "$MODEL" --mesh --mesh-prec "$EXCAVATOR_ARM_PREC" --materials-csv "$MATERIALS"
run excavator_arm_coarse "$MODEL" --mesh --mesh-prec "$COARSE_PREC" --materials-csv "$MATERIALS"

# The exact-surface macro needs one post-processing step before o2-sim can JIT it; see
# patch_exact_macro.py.
"$PYOCC" "$GEO/validation/demo/patch_exact_macro.py" "$OUT"/conv/*/geom.C
echo "done"
