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

# The closure test: the same events through the hand-written C++ TGeo geometry
# and through its own STEP round trip, compared on ITS hit positions.
#
#   run_closure.sh <studydir> [nevents] [seed]
#
# Order matters and each step gates the next:
#
#  0. the baseline twice with one seed -- if those are not bit-identical the
#     per-track seeding is not doing what the comparison assumes and nothing
#     below means anything;
#  1. a CAD probe run, only to learn which local medium index the CAD side gives
#     each medium; MaterialManager resolves a loaded cut by (module, local index)
#     and skips a mismatch SILENTLY, so the mapping has to be built, not assumed;
#  2. the baseline's cuts and processes are carried over by medium NAME and the
#     CAD run is repeated with them loaded, dumping its own;
#  3. the two dumps are compared per medium -- a difference means the physics
#     configuration differs and the transport comparison must not be believed;
#  4. only then the hits. Note where they are: under o2-sim-serial an external
#     detector's hits stay in o2sim.root on a branch named after the detector
#     (CITSHit), rather than being split into o2sim_Hits<DetID>.root the way a
#     built-in detector's are.
#
# Bit-identical hits are NOT the acceptance for charged particles in material:
# Geant draws from the RNG per step, so one extra boundary crossing shifts every
# later draw of that track. Per-track seeding contains that to the track; it does
# not remove it. So the hit comparison reports a distribution, and the numbers to
# read are how many tracks survive with the same hit count and how far the rest
# moved.
set -euo pipefail

S="$1"; N="${2:-20}"; SEED="${3:-424242}"
CT=$(cd "$(dirname "$0")" && pwd)
R="$S/run"
mkdir -p "$R"

cadrun () {   # cadrun <outdir> <extra configKeyValues>
  mkdir -p "$R/$1"
  ( source "$S/env_o2.sh" >/dev/null 2>&1
    cd "$R/$1" && o2-sim-serial -n "$N" -g boxgen --seed "$SEED" \
        --detectorList "CADCLOSURE:$S/detectorlist.json" \
        --extGeomFile "$S/externalDetectors.json" \
        --configKeyValues "SimCutParams.trackSeed=true${2:-}" \
        -o o2sim > run.log 2>&1 )
}

echo "############ 0. determinism control: the baseline twice, one seed"
for r in base1 base2; do
  mkdir -p "$R/$r"
  ( source "$S/env_o2.sh" >/dev/null 2>&1
    cd "$R/$r" && o2-sim-serial -n "$N" -g boxgen -m PIPE ITS TPC MAG --seed "$SEED" \
        --configKeyValues "SimCutParams.trackSeed=true;MaterialManagerParam.outputFile=$R/cuts_baseline.json" \
        -o o2sim > run.log 2>&1 )
done
( source "$S/env_o2.sh" >/dev/null 2>&1
  python3 "$CT/compare_hits.py" "$R/base1" "$R/base2" --json "$R/determinism.json" )

echo
echo "############ 1. CAD probe run, to learn its own medium indices"
cadrun cad_probe ";MaterialManagerParam.outputFile=$R/cuts_cad_probe.json"

echo
echo "############ 2. carry the baseline's cuts over by medium name"
python3 "$CT/remap_cuts.py" --baseline "$R/cuts_baseline.json" \
    --cad-dump "$R/cuts_cad_probe.json" --out "$R/cuts_cad_in.json"

echo
echo "############ 3. the CAD run, with those cuts loaded"
cadrun cad ";MaterialManagerParam.inputFile=$R/cuts_cad_in.json;MaterialManagerParam.outputFile=$R/cuts_cad_out.json"
echo "robustness counters (all must be zero):"
for pat in "stuck" "G4Exception" "Navigation Error" "abort"; do
  printf "  %-16s baseline %-5s CAD %-5s\n" "$pat" \
      "$(grep -ic "$pat" "$R/base1/run.log" || true)" \
      "$(grep -ic "$pat" "$R/cad/run.log" || true)"
done
echo "transport size (they should be comparable, not equal):"
for d in base1 cad; do
  printf "  %-6s steps/event %-8s secondaries/event %s\n" "$d" \
      "$(grep -oP 'did \K[0-9]+(?= steps)' "$R/$d/run.log" | awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n}')" \
      "$(grep -oP 'Stack: [0-9]+ out of \K[0-9]+' "$R/$d/run.log" | awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n}')"
done

echo
echo "############ 4. did both sides get the same cuts and processes?"
python3 "$CT/remap_cuts.py" --compare --baseline "$R/cuts_baseline.json" \
    --cad-dump "$R/cuts_cad_out.json" || true

echo
echo "############ 5. the hits"
( source "$S/env_o2.sh" >/dev/null 2>&1
  python3 "$CT/compare_hits.py" "$R/base1" "$R/cad" \
      --file-a o2sim_HitsITS.root --branch-a ITSHit \
      --file-b o2sim.root --branch-b CITSHit \
      --tol 1e-4 --json "$R/hits.json" ) || true
