#!/usr/bin/env bash
#
# This is a simulation example showing how to run the simulation as a service,
# and how to interact with the service via the client control script. Here, the
# service is used to achieve event biasing via a feedback control loop, in which an 
# outside "broker" is triggering on the simulated events and launching more simulations
# until a goal is achieved. 
#
# Steps demonstrated:
# a) startup multiple simulations in service mode
# b) ask a batch of events from the crude service 
#    necessary to obtain the bias feature ... until until a certain number of triggered events
#    is reached
# c) continue simulation of the good events for the rest of primaries
#
# NOTE:
# - the example is primarily meant to demonstrate the use of the sim-service in a more complex setting
# - one can do biasing in this way, but alternative implementations (directly inside o2-sim) may
#   be ultimately better/faster (the present example can then be seen as first baseline solution)
# - This script is glueing together executables / ROOT macros / etc. --> A better/future way might be to 
#   use PyROOT so that one can directly access results better and reuse compiled macros.


MODULES="PIPE ITS TPC TOF"
NWORKERS=8
NTRIALEVENTS=20 # number of trial events in each batch
NTRIGGEREDEVENTS=20  # number of targeted triggered events

# helper to make a random file name
rname1=$(hexdump -n 16 -v -e '/1 "%02X"' -e '/16 "\n"' /dev/urandom | head -c 6)

set -x
### step 1: Startup the 1st service with a partial detector config

( o2-sim-client.py --startup "-j ${NWORKERS} -n 0 -g pythia8 -m ${MODULES} -o simservice --configFile sim_step1.ini" \
                   --block ) | tee /tmp/${rname1}  # <--- return when everything is fully initialized
SERVICE1_PID=$(grep "detached as pid" /tmp/${rname1} | awk '//{print $4}')

# a second service is used for the continue features (currently reconfiguration of engines/stacks is limited, otherwise
# the first service could be used as well)
( o2-sim-client.py --startup "-j ${NWORKERS} -n 0 -m ${MODULES} -o simservice2 --configKeyValues GeneratorFromO2Kine.continueMode=true" \
                   --block ) | tee /tmp/${rname1}  # <--- return when everything is fully initialized
SERVICE2_PID=$(grep "detached as pid" /tmp/${rname1} | awk '//{print $4}')

# sleep 20

# BIASING LOOP
biasedcount=0
batch=0
trialevents=0
while (( biasedcount < ${NTRIGGEREDEVENTS} )); do
  ### simulate some events with service 1
  o2-sim-client.py --pid ${SERVICE1_PID} --command "-g pythia8 -n ${NTRIALEVENTS} --configFile sim_step1.ini -o batch${batch}" --block

  ### filter out good events
  ln -nsf simservice_grp.root batch${batch}_grp.root
  command="broker.macro("'"-g extkinO2 --extKinFile batch'${batch}'_Kine.root --trigger external --configFile sim_step2.ini","batch'${batch}'", "filtered"'")"
  ( root -b -l -q "${command}" ) &> /tmp/${rname1}_${batch}

  triggercount=$(grep "TRIGGER-COUNT" /tmp/${rname1}_${batch} | awk '//{print $2}')

  let biasedcount=biasedcount+triggercount
  let batch=batch+1
  let trialevents=trialevents+NTRIALEVENTS
done
echo "========================================"
echo "Transported ${trialevents} trial events."
echo "Triggered on ${biasedcount} events.     "
echo "****************************************"


# SIMULATE REMAINING PRIMARIES FOR GOOD TRIGGERED EVENTS
o2-sim-client.py --pid ${SERVICE2_PID} --command "-g extkinO2 --extKinFile filtered_Kine.root -n ${NTRIGGEREDEVENTS} -o filtered_part2 \
                                                  --configKeyValues GeneratorFromO2Kine.continueMode=true" --block

# bring down the service
o2-sim-client.py --pid ${SERVICE1_PID} --command "--stop 1"
o2-sim-client.py --pid ${SERVICE2_PID} --command "--stop 1"

sleep 1

# just some tmp safety-net to make sure all processes are really gone
. ${O2_ROOT}/share/scripts/jobutils.sh
for p in $(childprocs ${SERVICE1_PID}); do
  kill -9 ${p}
done
. ${O2_ROOT}/share/scripts/jobutils.sh
for p in $(childprocs ${SERVICE2_PID}); do
  kill -9 ${p}
done
