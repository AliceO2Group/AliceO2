#!/usr/bin/env bash
#
# This is a simulation example showing how to run simulation and selectively
# transport particles through the detector geometry according to configurable settings
#
# The simulation uses Pythia8 pp generator with default settings.
# The specific settings of the simulation are defined in the sim.ini file.
#
# Only the parameters for the selective particle transport are adjusted, namely
# ```
# [Stack]
# transportPrimary=external
# transportPrimaryFileName=transportPDG.macro
# transportPrimaryFuncName=transportPDG(321)
# transportPrimaryInvert=false
# ```
#
# `transportPrimary` defines the mode of selection, `external` means that the definition is loaded from a plugin macro
# defined by the function call `transportPrimaryFuncName` from the file `transportPrimaryFileName`.
# `transportPrimaryInvert` allows one to invert the definition, hence to inhibit transport for the selected particles.
#
#
# The second step of this simulation examples shows how to continue the simulation from the first
# step and propagate all the remaining tracks. The process is done only for events that pass a specific
# trigger which requires that both photons from pi0 decay have converted within a given region.
#

set -x

MODULES="PIPE ITS TPC"
NEVENTS1=100
NEVENTS2=1
NWORKERS=1

### step 1

o2-sim -j ${NWORKERS} -n ${NEVENTS1} -g pythia8pp -m ${MODULES} -o step1 \
       --configFile sim_step1.ini --seed 73141128

### step 2

o2-sim -j ${NWORKERS} -n ${NEVENTS2} -g extkinO2 -m ${MODULES} -o step2 \
       --extKinFile step1_Kine.root \
       --configFile sim_step2.ini \
       --trigger external

