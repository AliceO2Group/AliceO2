#!/usr/bin/env bash
#
# This is a simulation example showing how to reduce the number of transport
# steps with a geometry hull: a set of cylinders wrapped around the detector,
# outside of which transport of a track is stopped.
#
# The hull is plugged into the per-step decision of O2MCApplicationBase via
# ```
# [SimCutParams]
# stepFilteringMacro=$O2_ROOT/share/Detectors/gconfig/KeepStepCylinders.macro
# ```
# The macro provides a `keepStep()` function returning the predicate that is
# evaluated for every step. It is applied in addition to the built-in z/R cut,
# so it can only remove steps, never add them.
#
# The chain below
#   1. records a reference simulation with the MCStepLogger,
#   2. derives a hull from the geometry of that simulation,
#   3. checks that every recorded hit lies inside the hull,
#   4. replays the reference without cuts and with the hull, and compares
#      the number of steps and the number of hits per detector.
#
# Requires the step logger in the environment:
#   alienv enter O2/latest MCStepLogger/latest

set -x

MODULES=""            # empty: default detector list
EVENTS=10
HERE=$(cd "$(dirname "$0")" && pwd)
TOP=$PWD

# Alignment is irrelevant to step and hit counting and switching it off keeps
# the example from needing a CCDB connection and an alien token.
CONFIG="align-geom.mDetectors=none"

# ---------------------------------------------------------------- 1. reference
mkdir -p ref && cd ref
MCSTEPLOG_TTREE=1 \
LD_PRELOAD=$MCSTEPLOGGER_ROOT/lib/libMCStepLoggerInterceptSteps.so \
o2-sim-serial -n ${EVENTS} -g pythia8pp -e TGeant4 ${MODULES} \
    --configKeyValues "${CONFIG}" > logref 2>&1
cd ..

# ------------------------------------------------------------------- 2. derive
# The defaults scan the full world in 1 cm slices and 128 azimuthal directions
# and approximate the result with 12 cylinders; this takes about half a minute.
( cd ref && root -l -b -q "${HERE}/makeKeepStepCylinders.macro(\
\"o2sim_geometry.root\", \"MCStepLoggerVolMap.dat\", \"MCStepLoggerSenVol.dat\", \
\"${TOP}/KeepStepCylinders.macro\")" )

# -------------------------------------------------------------------- 3. check
# Necessary condition: no hit may sit outside the hull.
( cd ref && root -l -b -q "${HERE}/checkKeepStepCylinders.macro(\
\"${TOP}/KeepStepCylinders.macro\", \"o2sim\")" )

# ------------------------------------------------------------------- 4. replay
# MCReplayParam.allowStopTrack lets the replay act on the StopTrack() calls the
# hull makes; without it a replay reproduces its reference and nothing is cut.
mkdir -p baseline && cd baseline
o2-sim-serial -n ${EVENTS} -e MCReplay -g extkinO2 ${MODULES} \
    --extKinFile ../ref/o2sim_Kine.root -o baseline \
    --configKeyValues "${CONFIG};MCReplayParam.stepFilename=../ref/MCStepLoggerOutput.root" \
    > logbaseline 2>&1
cd ..

mkdir -p hull && cd hull
o2-sim-serial -n ${EVENTS} -e MCReplay -g extkinO2 ${MODULES} \
    --extKinFile ../ref/o2sim_Kine.root -o hull \
    --configKeyValues "${CONFIG};MCReplayParam.stepFilename=../ref/MCStepLoggerOutput.root;\
MCReplayParam.allowStopTrack=true;\
SimCutParams.stepFilteringMacro=${TOP}/KeepStepCylinders.macro" \
    > loghull 2>&1
cd ..

# ------------------------------------------------------------------ 5. compare
# MCReplay prints "Original number, skipped, kept, ..." once per event.
grep -h "Original number" baseline/logbaseline hull/loghull

# Hit counts per detector. FT0 draws a random number while creating hits and
# does not reproduce under replay, so compare it against the baseline replay
# rather than against the reference simulation, and expect it to be noisy.
for d in baseline hull; do
  echo "=== $d"
  root -l -b -q "${HERE}/countHits.macro(\"${d}/${d}\")"
done
