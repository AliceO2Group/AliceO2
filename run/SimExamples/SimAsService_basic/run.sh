#!/usr/bin/env bash
#
# This is a basic simulation example showing how to run the simulation as a service,
# and how to interact with the service via the client control script.
#
# Steps demonstrated:
# a) startup a simulation in service mode
# b) ask a 1st batch of events
# c) ask a 2nd batch of events with a different output name


set -x

MODULES="PIPE ITS TPC TOF TRD"
NWORKERS=4

### step 1: Startup the service with some configuration of workers, engines, 
####        physics/geometry settings. No events are asked at this time.

o2-sim-client.py --startup "-j ${NWORKERS} -n 0 -g pythia8 -m ${MODULES} -o simservice" \
                 --block   # <--- return when everything is fully initialized

### step 2: Transport a bunch of pythia8 events; Reconfiguration of engine not possible at this time.
###         Reconfiguration of generator ok (but limited).
o2-sim-client.py --command "-n 10 -g pythia8 -o batch1_pythia8" --block


### step 3: Transport a bunch of pythia8hi events and wait until finished
o2-sim-client.py --command "-n 2 -g pythia8hi -o batch2_pythia8hi" --block


### step 4: ask the service to shutdown (note the additional 1 which is unfortunately needed)
o2-sim-client.py --command "--stop 1"

