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
NWORKERS=6

# helper to make a random file name
rname1=$(hexdump -n 16 -v -e '/1 "%02X"' -e '/16 "\n"' /dev/urandom | head -c 6)

### step 1: Startup the service with some configuration of workers, engines, 
####        physics/geometry settings. No events are asked at this time.

( o2-sim-client.py --startup "-j ${NWORKERS} -n 0 -g pythia8pp -m ${MODULES} -o simservice --logseverity DEBUG --configKeyValues align-geom.mDetectors=none"  \
                  --block ) | tee /tmp/${rname1}   # <--- return when everything is fully initialized
SERVICE1_PID=$(grep "detached as pid" /tmp/${rname1} | awk '//{print $4}')

sleep 2
### step 2: Transport a bunch of pythia8 events; Reconfiguration of engine not possible at this time.
###         Reconfiguration of generator ok (but limited).
o2-sim-client.py --pid ${SERVICE1_PID} --command "-n 10 -g pythia8pp -o batch1_pythia8" --block

sleep 2

### step 3: Transport a bunch of pythia8hi events and wait until finished
o2-sim-client.py --pid ${SERVICE1_PID} --command "-n 2 -g pythia8hi -o batch2_pythia8hi" --block


### step 4: ask the service to shutdown (note the additional 1 which is unfortunately needed)
o2-sim-client.py --pid ${SERVICE1_PID} --command "--stop 1"

