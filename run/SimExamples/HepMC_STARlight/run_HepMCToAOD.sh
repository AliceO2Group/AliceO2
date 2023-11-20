#!/usr/bin/env bash
#
# This is a simple simulation example showing the following things
#

# a) how to start STARlight event generator to produce HepMC files
# b) how to inject these HepMC files into an O2Physics Analysis task (generator-only analysis)

STARLIGHT_ROOT=$(starlight-config)
if [ ! "${STARLIGHT_ROOT}" ]; then
    echo "This needs STARLIGHT loaded; alienv enter ..."
    exit 1
fi

# PART a)
# ./run-starlight.sh
set -x
# PART b) ... apply vertex smearing on top of HepMC events and perform simple analysis
NEV=$(grep EVENT slight.out | wc -l)

o2-sim-dpl-eventgen -b --nevents ${NEV} --generator hepmc --confKeyValues \
                    "GeneratorFileOrCmd.fileNames=starlight.hepmc;Diamond.position[2]=0.1;Diamond.width[2]=0.05" |\
                    o2-sim-mctracks-to-aod -b | o2-analysis-mctracks-to-aod-simple-task -b

