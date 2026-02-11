#!/usr/bin/env bash

# An example for a DPL Pythia8 event generation with vertex smearing and a trigger function and subsequent
# injection into analysis framework.

set -x
NEVENTS=1000
## Add --aod-writer-keep dangling to o2-sim-mctracks-to-aod to write the AO2D file to disc (as AnalysisResults_trees.root)
CONFKEY="TriggerExternal.fileName=trigger.macro;TriggerExternal.funcName=trigger()"
o2-sim-dpl-eventgen -b --nEvents ${NEVENTS} --aggregate-timeframe 10 --generator pythia8pp --trigger external \
                    --vertexMode kDiamondParam --configKeyValues "${CONFKEY}" |\
o2-sim-mctracks-to-aod -b | o2-analysis-mctracks-to-aod-simple-task -b

