#!/usr/bin/env bash
#
# This is a simple simulation example showing the following things
#

# a) how to start STARlight event generator
# b) how to run simulation reading from the HepMC file

set -x
STARLIGHT_ROOT=$(starlight-config)
if [ ! "${STARLIGHT_ROOT}" ]; then
    echo "This needs STARLIGHT loaded; alienv enter ..."
    exit 1
fi

export STARLIGHT_ROOT
# PART a)
./run-starlight.sh

# PART b)
NEV=$(grep EVENT slight.out | wc -l)
o2-sim -j 20 -n ${NEV} -g hepmc -m PIPE ITS -o sim \
       --configKeyValues "GeneratorFileOrCmd.fileNames=starlight.hepmc;Diamond.position[2]=0.1;Diamond.width[2]=0.05"
