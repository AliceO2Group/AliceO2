#!/usr/bin/env bash
#
# Same event pool roll-over as run.sh, but through the DPL event generator
# i.e. the path used on hyperloop:
#
#   o2-sim-dpl-eventgen | o2-sim-mctracks-to-aod | <analysis task>
#
# Requires the event pool created by run.sh (or any other pool directory given
# via the POOLDIR variable).
#
set -x
set -e

[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1

POOLDIR=${POOLDIR:-${PWD}/eventpool}
[ ! -d "${POOLDIR}" ] && echo "Error: no event pool at ${POOLDIR}; run run.sh first" && exit 2

NEVENTS=9
COMMON="-b --aggregate-timeframe 10 --vertexMode kNoVertex --seed 12345"

# ---------------------------------------------------------------------------
# a) the whole pool as one event stream
# ---------------------------------------------------------------------------
o2-sim-dpl-eventgen ${COMMON} --nEvents ${NEVENTS} --generator evtpool \
    --configKeyValues "GeneratorEventPool.eventPoolPath=${POOLDIR}" |\
  o2-sim-mctracks-to-aod -b |\
  o2-analysis-mctracks-to-aod-simple-task -b

# ---------------------------------------------------------------------------
# b) reuse the pool once it is exhausted (2 full passes here)
# ---------------------------------------------------------------------------
o2-sim-dpl-eventgen ${COMMON} --nEvents $((2 * NEVENTS)) --generator evtpool \
    --configKeyValues "GeneratorEventPool.eventPoolPath=${POOLDIR};GeneratorEventPool.roundRobin=true" |\
  o2-sim-mctracks-to-aod -b |\
  o2-analysis-mctracks-to-aod-simple-task -b

# ---------------------------------------------------------------------------
# c) plain 'extkinO2' with a comma-separated list of files
# ---------------------------------------------------------------------------
FILELIST=$(ls ${POOLDIR}/*/evtpool.root | paste -sd,)
o2-sim-dpl-eventgen ${COMMON} --nEvents ${NEVENTS} --generator extkinO2 \
    --configKeyValues "GeneratorFromO2Kine.fileName=${FILELIST}" |\
  o2-sim-mctracks-to-aod -b |\
  o2-analysis-mctracks-to-aod-simple-task -b

# ---------------------------------------------------------------------------
# d) asking for more events than the pool holds must fail, not hang
# ---------------------------------------------------------------------------
if o2-sim-dpl-eventgen ${COMMON} --nEvents $((NEVENTS + 1)) --generator evtpool \
       --configKeyValues "GeneratorEventPool.eventPoolPath=${POOLDIR}" |\
     o2-sim-mctracks-to-aod -b |\
     o2-analysis-mctracks-to-aod-simple-task -b; then
  echo "ERROR: the workflow should have failed with 'ran out of events'" && exit 1
else
  echo "OK: the workflow failed as expected (not enough events in the pool)"
fi
