#!/usr/bin/env bash
#
# Example showing how a job goes through an event pool made of several files: one
# file is read at a time and the next one is opened only once the events of the
# current one are exhausted.
#
# Stage 1: create a small event pool made of NPOOLFILES files
# Stage 2: read the pool back in its different modes
#
# Note: the example runs with '--noGeant' and a minimal geometry ('-m PIPE'),
# since only the event generation part is of interest here.
#
set -x
set -e

[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1

NPOOLFILES=3        # number of files the pool consists of
NEVENTS_PER_FILE=3  # events per pool file
NEVENTS=9           # events to be read back from the whole pool

POOLDIR=${PWD}/eventpool
COMMON="-j 4 --noGeant --vertexMode kNoVertex"

# ---------------------------------------------------------------------------
# Stage 1: produce the event pool
# ---------------------------------------------------------------------------
# An event pool is a set of kinematics files that are all called 'evtpool.root'
# and that live in separate sub-directories of a common pool directory.
rm -rf ${POOLDIR}
for i in $(seq 0 $((NPOOLFILES - 1))); do
  o2-sim ${COMMON} -n ${NEVENTS_PER_FILE} -g pythia8pp -o poolgen_${i} --seed $((i + 1))
  mkdir -p ${POOLDIR}/00${i}
  mv poolgen_${i}_Kine.root ${POOLDIR}/00${i}/evtpool.root
done

# ---------------------------------------------------------------------------
# Stage 2a: sequential roll-over through the whole pool
# ---------------------------------------------------------------------------
# The generator moves on to the next pool file once the events of the current one
# are used up, so that NPOOLFILES * NEVENTS_PER_FILE events are available in total.
# Only one file is open at a time.
o2-sim ${COMMON} -n ${NEVENTS} -g evtpool -o rollover \
       --configKeyValues "GeneratorEventPool.eventPoolPath=${POOLDIR};GeneratorEventPool.randomize=false;GeneratorEventPool.rngseed=1"

# ---------------------------------------------------------------------------
# Stage 2b: start over with the first file once the pool is exhausted
# ---------------------------------------------------------------------------
o2-sim ${COMMON} -n $((NEVENTS + NEVENTS_PER_FILE)) -g evtpool -o rollover_roundrobin \
       --configKeyValues "GeneratorEventPool.eventPoolPath=${POOLDIR};GeneratorEventPool.roundRobin=true;GeneratorEventPool.randomize=false;GeneratorEventPool.rngseed=1"

# ---------------------------------------------------------------------------
# Stage 2c: randomized access (the event pool default) across the files
# ---------------------------------------------------------------------------
# The entries of each file are served in a random permutation, so every event of
# the pool is still used exactly once.
o2-sim ${COMMON} -n ${NEVENTS} -g evtpool -o rollover_random \
       --configKeyValues "GeneratorEventPool.eventPoolPath=${POOLDIR};GeneratorEventPool.rngseed=1"

# ---------------------------------------------------------------------------
# Stage 2d: asking for more events than the pool holds must fail
# ---------------------------------------------------------------------------
# It is up to the user to provide enough events; running out of them is fatal.
if o2-sim ${COMMON} -n $((NEVENTS + 1)) -g evtpool -o toofew \
       --configKeyValues "GeneratorEventPool.eventPoolPath=${POOLDIR};GeneratorEventPool.rngseed=1"; then
  echo "ERROR: the simulation should have failed with 'ran out of events'" && exit 1
else
  echo "OK: the simulation failed as expected (not enough events in the pool)"
fi

# ---------------------------------------------------------------------------
# Stage 2e: the same mechanism is available for plain 'extkinO2' by giving a
#           comma-separated list of files
# ---------------------------------------------------------------------------
FILELIST=$(ls ${POOLDIR}/*/evtpool.root | paste -sd,)
o2-sim ${COMMON} -n ${NEVENTS} -g extkinO2 -o extkin_rollover --extKinFile "${FILELIST}"
