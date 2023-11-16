#!/usr/bin/env bash

set -x

NEVENTS=1000

# --aggregate-timeframe 200 is used to combine 200 generated events into a timeframe that is then converted to AOD tables
# note that if you need special configuration for the analysis tasks, it needs to be passed to proxy and converter as well
CONFKEY="Diamond.position[1]=10.;Diamond.width[1]=1"
o2-sim-dpl-eventgen -b --nevents ${NEVENTS} --aggregate-timeframe 1 --generator pythia8pp --shm-segment-size 4000000000 --confKeyValues "${CONFKEY}" --vertexMode kNoVertex |\
o2-sim-mctracks-to-aod -b | o2-analysis-mctracks-to-aod-simple-task -b

# the very same analysis task can also directly run on an AO2D with McCollisions and McParticles:
# o2-analysis-mctracks-to-aod-simple-task -b --aod-file <AO2DFile>
