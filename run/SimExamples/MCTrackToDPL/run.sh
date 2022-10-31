#!/usr/bin/env bash

set -x

# launch generator process (for 100 min bias Pythia8 events)
o2-sim -j 1 -g pythia8pp -n 100 --noWriteToDisc --forwardKine &> sim.log &
SIMPROC=$!

# launch a DPL process (having the right proxy configuration)
o2-sim-mctracks-proxy --enable-test-consumer &> out_mcanalysis.log &
TRACKANAPROC=$!

wait ${SIMPROC}
sleep 5
wait ${TRACKANAPROC}
