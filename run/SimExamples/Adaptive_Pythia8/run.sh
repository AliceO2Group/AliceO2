#!/usr/bin/env bash
#
# This is a simple simulation example showing the following things
#

# a) how to run a simple background event simulation with some parameter customization
# b) how to run a simulation producing signal events based on a custom external generator
#    that can adapt its behaviour depending on the characteristics of the background event.

set -x

# PART a)
NBGR=5
o2-sim -j 20 -n ${NBGR} -g pythia8hi -m PIPE ITS -o bkg --configKeyValues \
       "Diamond.position[2]=0.1;Diamond.width[2]=0.05"

# PART b)
# produce pythia8 events generated according to the configuration given in a file 'adaptive_pythia8.macro'.
# the settings as such that pythia8 is initialised using the 'pythia8_inel.cfg' configuration file.
# the customisation allows the user to generator to receive and react to a notification that signals
# the embedding status of the simulation, giving the header of the background event for determination
# of subsequent actions. In this case, the number of pythia8 events to be embedded is calculated according
# to a formula that uses the number of primary particles of the background events
NSGN=10
o2-sim -j 20 -n ${NSGN} -m PIPE ITS \
       -g external --configKeyValues 'GeneratorExternal.fileName=adaptive_pythia8.macro;GeneratorExternal.funcName=adaptive_pythia8("0.002 * x")' \
       --embedIntoFile bkg_Kine.root -o sgn > logsgn 2>&1
