#!/usr/bin/env bash
#
# This is a simulation example showing how to run simulation with Pythia8
# with an external generator that adds custom information to the event header
#
#

set -x

MODULES="PIPE ITS TPC"
EVENTS=100
NWORKERS=8

### generate some events with the external generator that will
### provide some custom information in the event header

o2-sim -j ${NWORKERS} -n ${EVENTS} -g external -m ${MODULES} \
       --configFile sim.ini > log 2>&1

### read the kinematics to print the custom information stored by
### the external generator that we ran before

root -b -q -l "read_event_info.macro(\"o2sim_Kine.root\")"
