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

set -x

MODULES="PIPE ITS TPC"
EVENTS=100
NWORKERS=8

o2-sim -j ${NWORKERS} -n ${EVENTS} -g pythia8pp -m ${MODULES} -o step1 \
       --configFile sim.ini \
       > logstep1 2>&1

