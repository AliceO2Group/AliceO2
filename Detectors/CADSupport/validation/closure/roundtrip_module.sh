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

# One module through the whole round trip: TGeo -> STEP (+ media sidecar) -> TGeo.
#
#   roundtrip_module.sh <studydir> <MODULE>
#
# Writes <studydir>/cad/<MODULE>/ with o2sim_geometry.root, <MODULE>.step,
# <MODULE>_media.json and conv/geom.C.  The o2-sim step and the converter steps
# run in SEPARATE shells on purpose: the pythonOCC PYTHONPATH prepends segfault
# o2-sim at startup.  Conversions must not be run in parallel -- --csg auto
# defers its emit and two concurrent runs lose shapes.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS=$(cd "$HERE/../../tools" && pwd)

S="$1"; MOD="$2"
D="$S/cad/$MOD"
mkdir -p "$D"

echo "=== $MOD: building the source geometry"
( source "$S/env_o2.sh" >/dev/null 2>&1
  cd "$D" && o2-sim-serial -n 0 -g boxgen -m "$MOD" -o o2sim > geom.log 2>&1 )

# The experiment hall is hollowed out: o2-sim builds cave/barrel/caveRB24 itself
# whatever module list is asked for, so shipping a second copy would put four
# coincident air boxes in the world.  Their structure is kept, so every subtree
# below them still lands at exactly the transform the source geometry gave it.
echo "=== $MOD: TGeo -> STEP + media sidecar"
( source "$S/env_o2.sh" >/dev/null 2>&1
  source "$S/env_converter.sh"
  cd "$D" && "$SW/Python/latest/bin/python3.10" \
      "$TOOLS/O2_TGeoToCAD.py" o2sim_geometry.root "$MOD.step" \
      --report "${MOD}_writer_report.json" --media-json "${MOD}_media.json" \
      --hollow-volume cave --hollow-volume barrel --hollow-volume caveRB24 \
      --hollow-tag "$MOD" \
      > writer.log 2>&1 )
tail -3 "$D/writer.log"

echo "=== $MOD: STEP -> TGeo (csg auto / exact surfaces auto / mesh fallback)"
( source "$S/env_o2.sh" >/dev/null 2>&1
  source "$S/env_converter.sh"
  cd "$D" && "$SW/Python/latest/bin/python3.10" \
      "$TOOLS/O2_CADtoTGeo.py" "$MOD.step" -o geom.C \
      --output-folder conv --csg auto --exact-surfaces auto --mesh \
      --media-json "${MOD}_media.json" > conv.log 2>&1 )
grep -E "tiers:|Media from sidecar|WARN" "$D/conv.log" || true

echo "=== $MOD: where does this module hang itself?"
( source "$S/env_o2.sh" >/dev/null 2>&1
  cd "$D" && python3 "$HERE/module_anchors.py" \
      o2sim_geometry.root --json anchors.json 2>&1 | grep -vE "^Info in|^Warning in" )

echo "=== $MOD: do the media survive?"
( source "$S/env_o2.sh" >/dev/null 2>&1
  cd "$D" && python3 "$HERE/check_media.py" \
      --original o2sim_geometry.root --macro conv/geom.C --rtol 1e-6 --writer-report "${MOD}_writer_report.json" \
      --json media_check.json 2>&1 | grep -vE "^Info in|^Warning in|^Note:" )
