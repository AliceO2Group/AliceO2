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

# Reduce every run's MCStepLogger tree to a text tally, and every geantino run to a per-ray
# material budget. Writes <conv-root>/analysis/{steps_<tag>.txt,matbudget_<tag>.txt}.
#
# Usage: analyse_all.sh <conv-root>
set -u
GEO=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: analyse_all.sh <conv-root>}
mkdir -p "$OUT/analysis"
command -v root >/dev/null || { echo "analyse_all.sh: root not found; load the O2 environment" >&2; exit 1; }
# MCStepLogger is not on the O2 environment's paths; only the analysis needs it.
MCSL=${MCSTEPLOGGER_ROOT:-${O2_ROOT:+$O2_ROOT/../../MCStepLogger/latest}}
[ -d "$MCSL/include" ] || { echo "analyse_all.sh: MCStepLogger not found; set MCSTEPLOGGER_ROOT" >&2; exit 1; }
export LD_LIBRARY_PATH=$MCSL/lib:${LD_LIBRARY_PATH:-}
export ROOT_INCLUDE_PATH=$MCSL/include:${ROOT_INCLUDE_PATH:-}
for d in "$OUT"/runs/*/; do
  tag=$(basename "$d")
  sf="$d/MCStepLoggerOutput.root"
  [ -f "$sf" ] || continue
  root -l -b -q "$GEO/analyse_steps.macro(\"$sf\")" > "$OUT/analysis/steps_$tag.txt" 2>&1
  case "$tag" in
    geantino_*|matfan_*)
      root -l -b -q "$GEO/matbudget.macro(\"$d/o2sim_geometry.root\",\"$sf\",\"$OUT/analysis/matbudget_$tag.txt\")" \
        > "$OUT/analysis/matbudget_$tag.log" 2>&1
      ;;
  esac
  echo "analysed $tag"
done
