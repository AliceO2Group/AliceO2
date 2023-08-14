#!/usr/bin/env bash

set -x

NEVENTS=10000
# launch generator process (for 10000 min bias Pythia8 events; no Geant; no geometry)
o2-sim -j 1 -g pythia8pp -n ${NEVENTS} --noDiscOutput --forwardKine --noGeant &> sim.log &
SIMPROC=$!

# launch a DPL process (having the right proxy configuration)
# (Note that the option --o2sim-pid is not strictly necessary when only one o2-sim process is running.
#  The socket will than be auto-determined.)

# --aggregate-timeframe 200 is used to combine 200 generated events into a timeframe that is then converted to AOD tables
# note that if you need special configuration for the analysis tasks, it needs to be passed to proxy and converter as well
o2-sim-mctracks-proxy -b --nevents ${NEVENTS} --o2sim-pid ${SIMPROC} --aggregate-timeframe 200 |\
o2-sim-mctracks-to-aod -b |\
o2-analysis-mctracks-to-aod-simple-task -b &
TRACKANAPROC=$!

wait ${SIMPROC}
wait ${TRACKANAPROC}


# the very same analysis task can also directly run on an AO2D with McCollisions and McParticles:
# o2-analysis-mctracks-to-aod-simple-task -b --aod-file <AO2DFile>
