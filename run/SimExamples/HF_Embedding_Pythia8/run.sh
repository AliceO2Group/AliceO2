#!/usr/bin/env bash
#
# This is a simulation example showing the following things
#
# a) how to run a simple background event simulation with some parameter customization
# b) how to generate a pythia8 configuration file replacing values from a template configuration
# c) how to run a simulation producing signal events using several specific pythia8 settings and user hooks
# d) how to run digitization embedding/overlaying background and signal hits
#
# The main focus in this example is on PART c). PART b) just relaces some template values in the Pythia8 configuration file.
#
# An external event generator configuration  `-g external` is loaded from the file defined via `--configKeyValues` setting
# `GeneratorExternal.fileName=GeneratorHF.macro` by running the code defined in the function `GeneratorExternal.funcName="GeneratorHF()"`.
# Special configuration parameters are loaded from the INI file `--configFile o2sim_configuration_sgn.ini`.
#
# Pythia8.config defines the Pythia8 configuration file name.
#
# We configured to bias towards c-cbar processes where we can select them baed on pt-hat bins.
#
# Pythia8.hooksFileName defines the file name where to load the custom Pythia8 hooks
# Pythia8.hooskFuncName defines the function call to be run to retrieve the custom Pythia8 hools.
#
# Hooks are used in this example to speedup the event generation. Event generation is paused at parton level.
# We check if there are the partons of our interest, if not we veto the event. This saves time because we
# do not have to process the full hadronisation step of uninteresting events.
# Look inside `pythia8_userhooks_ccbar.macro` for details on how this is implemented
#
# DecayerPythia8.config[0], [1] and [2] define several configuration files for the external decayer.
# The files are loaded in series. Look inside the corresponding `.cfg` for details.
#
# SimUserDecay.pdglist instructs Geant not to process the decays, but to hand them to the external decayer.
#

set -x

MODULES="PIPE ITS TPC"
BKGEVENTS=5
SIGEVENTS=20
NWORKERS=8

# PART a)

o2-sim -j ${NWORKERS} -n ${BKGEVENTS} -g pythia8hi -m ${MODULES} -o bkg \
       --configFile o2sim_configuration_bkg.ini \
       > logbkg 2>&1

# PART b)

RNDSEED=0    # [default = 0] time-based random seed
PTHATMIN=0.  # [default = 0]
PTHATMAX=-1. # [default = -1]

sed -e "s/\${rndSeed}/$RNDSEED/" \
    -e "s/\${pTHatMin}/$PTHATMIN/" \
    -e "s/\${pTHatMax}/$PTHATMAX/" \
    pythia8_ccbar.template > pythia8.cfg

# PART c)

o2-sim -j ${NWORKERS} -n ${SIGEVENTS} -g external -m ${MODULES} -o sgn \
       --configKeyValues "GeneratorExternal.fileName=GeneratorHF.macro;GeneratorExternal.funcName=GeneratorHF()" \
       --configFile o2sim_configuration_sgn.ini \
       --embedIntoFile bkg_Kine.root \
       > logsgn 2>&1

# PART d)

o2-sim-digitizer-workflow --sims bkg,sgn --tpc-lanes 4 -b --run
