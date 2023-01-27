#!/usr/bin/env bash

set -x

# launch generator process (for 100 min bias Pythia8 events; no Geant; no geometry)
o2-sim -j 1 -g pythia8pp -n 100 --noDiscOutput --forwardKine --noGeant -m CAVE -e TGeant3 &> sim.log &
SIMPROC=$!

# launch a DPL process (having the right proxy configuration)
# (Note that the option --o2sim-pid is not strictly necessary when only one o2-sim process is running.
#  The socket will than be auto-determined.)
o2-sim-mctracks-proxy --enable-test-consumer --o2sim-pid ${SIMPROC} &> out_mcanalysis.log &
TRACKANAPROC=$!

wait ${SIMPROC}
sleep 5
kill ${TRACKANAPROC}
wait ${TRACKANAPROC}
