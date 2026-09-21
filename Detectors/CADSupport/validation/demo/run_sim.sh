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

# One o2-sim run of the integration demo.
#
# Usage: run_sim.sh <conv-root> <exact|tess> <run-tag> [extra o2-sim args...]
#
# Environment knobs:
#   EVENTS=3  SEED=42  GEN=boxgen  PDG=0 (geantino)  NGUN=20  STEPLOG=0|1  NOGEANT=0|1
#   CONFIGKEY="a=1;b=2"   extra --configKeyValues, appended to the box-gun ones
#
# Everything is deterministic: the seed is fixed and the run is single-threaded
# (o2-sim-serial), so two runs differing only in the geometry representation are comparable.
set -u
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
GEO=$(cd "$(dirname "$0")/.." && pwd)
CONV=${1:?usage: run_sim.sh <conv-root> <exact|tess> <run-tag> [args...]}
REP=${2:?}
TAG=${3:?}
shift 3

EVENTS=${EVENTS:-3}
SEED=${SEED:-42}
GEN=${GEN:-boxgen}
PDG=${PDG:-0}
NGUN=${NGUN:-20}
PMIN=${PMIN:-1.0}
PMAX=${PMAX:-1.0}
STEPLOG=${STEPLOG:-0}
CONFIGKEY=${CONFIGKEY:-}
NOGEANT=${NOGEANT:-0}

RUNDIR=$CONV/runs/$TAG
mkdir -p "$RUNDIR"
python3 "$GEO/demo/make_configs.py" "$CONV" "$REP" "$RUNDIR" || exit 1

command -v o2-sim-serial >/dev/null || { echo "run_sim.sh: o2-sim-serial not found; load the O2 environment" >&2; exit 1; }
cd "$RUNDIR" || exit 1

ARGS=(-n "$EVENTS" -g "$GEN" --seed "$SEED"
      --detectorList "EXTCAD:$RUNDIR/detectorlist.json"
      --extGeomFile "$RUNDIR/externalGeometry.json"
      --configKeyValues "BoxGun.number=$NGUN;BoxGun.pdg=$PDG;BoxGun.prange[0]=$PMIN;BoxGun.prange[1]=$PMAX${CONFIGKEY:+;$CONFIGKEY}"
      -o o2sim)
[ "$NOGEANT" = "1" ] && ARGS+=(--noGeant)

if [ "$STEPLOG" = "1" ]; then
  MCSL=${MCSTEPLOGGER_ROOT:-${O2_ROOT:+$O2_ROOT/../../MCStepLogger/latest}}
  [ -f "$MCSL/lib/libMCStepLoggerInterceptSteps.so" ] || { echo "run_sim.sh: MCStepLogger not found; set MCSTEPLOGGER_ROOT" >&2; exit 1; }
  export LD_PRELOAD=$MCSL/lib/libMCStepLoggerInterceptSteps.so
  export MCSTEPLOG_OUTFILE=$RUNDIR/MCStepLoggerOutput.root
  # the per-step ROOT tree (StepLoggerTree) is only written when MCSTEPLOG_TTREE is set;
  # without it MCStepLogger only prints its per-volume summary to the log.
  export MCSTEPLOG_TTREE=1
fi

/usr/bin/time -v o2-sim-serial "${ARGS[@]}" "$@" > "$RUNDIR/sim.log" 2>&1
rc=$?
unset LD_PRELOAD
echo "run $TAG ($REP) exit=$rc -> $RUNDIR/sim.log"
exit $rc
