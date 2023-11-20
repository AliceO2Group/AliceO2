#!/usr/bin/env bash

# An example for a DPL Pythia8 event generation (without vertex smearing) and injection into analysis framework

set -x

NEVENTS=1000

# --aggregate-timeframe 10 is used to combine 10 generated events into a timeframe that is then converted to AOD tables
# note that if you need special configuration for the analysis tasks, it needs to be passed to proxy and converter as well

o2-sim-dpl-eventgen -b --nevents ${NEVENTS} --aggregate-timeframe 10 --generator pythia8pp --confKeyValues "${CONFKEY}" --vertexMode kNoVertex |\
o2-sim-mctracks-to-aod -b | o2-analysis-mctracks-to-aod-simple-task -b &> pythia8.log

# the very same analysis task can also directly run on an AO2D with McCollisions and McParticles:
# o2-analysis-mctracks-to-aod-simple-task -b --aod-file <AO2DFile>
