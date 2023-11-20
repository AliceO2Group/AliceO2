#!/usr/bin/env bash

# Example in which we produce a kinematics file in a first step and
# use it to inject events into analysis framework in a second step

set -x

# Produce events on disc
NEVENTS=1000
# launch generator process (for 10000 min bias Pythia8 events; no Geant; no geometry)
# o2-sim -j 1 -g pythia8pp -n ${NEVENTS} --noGeant --vertexMode kNoVertex &> sim.log

# Option 1) -- use o2-mckine-publisher
[ -f AnalysisResults.root ] && rm AnalysisResults.root
o2-sim-kine-publisher -b --kineFileName o2sim --aggregate-timeframe 10 |\
o2-sim-mctracks-to-aod -b |\
o2-analysis-mctracks-to-aod-simple-task -b &> log1
mv AnalysisResults.root AnalysisResult_1.root

# Option 2) -- use o2-sim-dpl-eventgen + extkinO2 generator (this should be equivalent to Option 1)
[ -f AnalysisResults.root ] && rm AnalysisResults.root
o2-sim-dpl-eventgen -b --nevents ${NEVENTS} --aggregate-timeframe 10 --generator extkinO2 \
                    --confKeyValues "GeneratorFromO2Kine.fileName=o2sim_Kine.root" --vertexMode kNoVertex |\
o2-sim-mctracks-to-aod -b |\
o2-analysis-mctracks-to-aod-simple-task -b &> log2
mv AnalysisResults.root AnalysisResult_2.root
