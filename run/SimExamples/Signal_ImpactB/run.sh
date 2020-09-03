#!/usr/bin/env bash
#
# This is a simulation example showing the following things
#

# a) how to run a simple background event simulation with some parameter customization
# b) how to setup and run an event generator that produces signal events based on the 
#    impact parameter of the backround event where it will be embetted into

set -x

# PART a)
NBGR=5
o2-sim -j 20 -n ${NBGR} -g pythia8hi -m PIPE ITS -o bkg --configKeyValues \
       "Diamond.position[2]=0.1;Diamond.width[2]=0.05"

# PART b)
# produce signal events generated according to the configuration given in a file 'signal_impactb.macro'.
# the custom event generator receives and react to a notification that signals
# the embedding status of the simulation, giving the header of the background event for determination
# of subsequent actions. In this case, the impact paramereter from the backgorund event
# is used to calculate the number of particles to be generated as signal
NSGN=10
o2-sim -j 20 -n ${NSGN} -m PIPE ITS \
       -g external --configKeyValues 'GeneratorExternal.fileName=signal_impactb.macro;GeneratorExternal.funcName=signal_impactb(333, "20. / (x + 1.)")' \
       --embedIntoFile bkg_Kine.root -o sgn > logsgn 2>&1
