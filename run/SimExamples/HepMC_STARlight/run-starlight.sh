#! /usr/bin/env bash

echo " --- run-starlight.sh ---"

set -x

# prepare environment
#VERSION=latest
#eval $(alienv printenv -q STARlight/$VERSION)
#STARLIGHT_ROOT=$(starlight-config)

# populate working directory
cp $STARLIGHT_ROOT/config/slight.in .
cp $STARLIGHT_ROOT/HepMC/*.awk .

# run STARlight
starlight slight.in | tee slight.log
cat slight.log slight.out | \
    awk -f pdgMass.awk -f starlight2hepmc.awk > starlight.hepmc
