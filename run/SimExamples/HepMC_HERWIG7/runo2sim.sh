#!/usr/bin/env bash
#
# This is a simple simulation example on how to generate HEPMC3 data from
# HERWIG7 and run an ALICE simulation using the o2-sim executable
# In the script we assume that the .run file has the same name of the input file, so change it accordingly.
# This script works only with AliGenO2 version containing the HERWIG7 generator

# HERWIG7 and O2 must be loaded
set -x
if [ ! "${HERWIG_ROOT}" ]; then
    echo "This needs HERWIG7 loaded; alienv enter ..."
    exit 1
fi

[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 2

NEV=-1
more=""
input="LHC"
eCM=-1
JOBS=2

usage()
{
    cat <<EOF
Usage: $0 [OPTIONS]

Options:

  -m,--more    CONFIG      More configurations ($more)
  -n,--nevents EVENTS      Number of events ($nev)
  -i,--input   INPUT       Options file fed to HERWIG7 ($input)
  -j,--jobs    JOBS        Number of jobs ($JOBS)
  -e,--ecm     ENERGY      Center-of-Mass energy
  -h,--help                Print these instructions
  --                       Rest of command line sent to o2-sim

COMMAND must be quoted if it contains spaces or other special
characters

Below follows the help output of o2-sim

EOF
}

if [ "$#" -lt 2 ]; then
    echo "Running with default values"
fi

while test $# -gt 0 ; do
    case $1 in
        -m|--more)    more="$2" ; shift ;;
        -n|--nevents) NEV=$2 ; shift ;;
        -i|--input)   input=$2 ; shift ;;
        -j|--jobs)    JOBS=$2 ; shift ;;
        -e|--ecm)     eCM=$2 ; shift ;;
        -h|--help) usage; o2-sim --help full ; exit 0 ;;
        --)           shift ; break ;;
        *) echo "Unknown option '$1', did you forget '--'?" >/dev/stderr
           exit 3
           ;;
    esac
    shift
done

echo "Input file: $input"

if [ ! -f $input.in ]; then
    echo "Error: Input file $input.in not found"
    exit 4
else
    if grep -Fq "saverun" $input.in; then
        sed -i "/saverun/c\saverun $input EventGenerator" $input.in
    else
        echo "saverun $input EventGenerator" >> $input.in
    fi
fi

# Set number of events to write in HepMC in input file
if [ ! $NEV -eq -1 ]; then
    echo "Setting number of events to $NEV"
    if grep -Fq "PrintEvent" $input.in; then
        sed -i "/PrintEvent/c\set /Herwig/Analysis/HepMC:PrintEvent $NEV" $input.in
    else
        echo "set /Herwig/Analysis/HepMC:PrintEvent $NEV" >> $input.in
    fi
else
    echo "Number of events not set, checking input file..."
    if grep -Fq "PrintEvent" $input.in; then
        NEV=$(grep -F "PrintEvent" $input.in | awk '{print $3}')
        echo "Number of events set to $NEV"
    else
        echo "Error: Number of events not set in HERWIG7"
        exit 5
    fi
fi

# Set ECM

if [ ! $eCM -eq -1 ]; then
    echo "Setting eCM to $eCM"
    if grep -Fq "Energy" $input.in; then
        sed -i "/Energy/c\set EventGenerator:EventHandler:LuminosityFunction:Energy $eCM" $input.in
    else
        echo "set EventGenerator:EventHandler:LuminosityFunction:Energy $eCM" >> $input.in
    fi
else
    echo "Energy not set, checking input file..."
    if grep -Fq "Energy" $input.in; then
        eCM=$(grep -F "Energy" $input.in | awk '{print $3}')
        echo "Energy set to $eCM"
    else
        echo "Error: eCM not set in HERWIG7"
        exit 6
    fi
fi

# Generating events using HERWIG7
Herwig read --repo=${HERWIG_ROOT}/share/Herwig/HerwigDefaults.rpo $input.in
Herwig run -N $NEV $input.run

# Starting simulation with o2-sim
o2-sim -j $JOBS -n ${NEV} -g hepmc  \
       --configKeyValues "GeneratorFileOrCmd.fileNames=herwig.hepmc;${more}"
