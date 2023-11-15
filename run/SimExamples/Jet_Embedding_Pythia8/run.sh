#!/usr/bin/env bash
#
# This is a simple simulation example showing the following things
#

# a) how to run a simple background event simulation with some parameter customization
# b) how to run a simulation producing signal events using specific pythia8 settings and user hooks (to accept / reject generated events)
# c) how to run digitization embedding/overlaying background and signal hits
# d) how to access the MC kinematics given MClabels
# e) how to navigate MCtracks

set -x

# PART a)
NBGR=5
o2-sim -j 20 -n ${NBGR} -g pythia8hi -m PIPE ITS TPC -o bkg --configKeyValues \
       "Diamond.position[2]=0.1;Diamond.width[2]=0.05"

# PART b)
# produce hard jets using a pythia8 configuration given in a file 'pythia8_hard.cfg'; event selection is done by a user hook specified
# in file 'pythia8_userhooks_jets.macro' and using same vertex setting as background events (via --embedInto)
NSGN=10
o2-sim -j 20 -n ${NSGN} -g pythia8pp -m PIPE ITS TPC --configKeyValues "GeneratorPythia8.config=pythia8_hard.cfg;GeneratorPythia8.hooksFileName=pythia8_userhooks_jets.macro" --embedIntoFile bkg_Kine.root -o sgn > logsgn 2>&1

# PART c)
# digitization with summation of signal on top of background events
o2-sim-digitizer-workflow --sims bkg,sgn --tpc-lanes 4 -b --run

# PART d)
# Simple analysis: read ITS digits and analyse some properties of digits and MC tracks leaving a digit
root -q -b -l 'exampleMCTrackAnalysis.macro("itsdigits.root")'
