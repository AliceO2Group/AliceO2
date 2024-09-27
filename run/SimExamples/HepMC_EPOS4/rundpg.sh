#!/usr/bin/env bash
#
# This script shows how to start the DPG simulation machinery starting EPOS4
# automatically while using the FIFOs generated internally by GeneratorHepMC

# This script works only with O2sim version starting from the 20/09/2024

set -x
if [ ! "${EPOS4_ROOT}" ]; then
    echo "This needs EPOS4 loaded; alienv enter ..."
    exit 1
fi

# make sure O2DPG + O2 is loaded
[ ! "${O2DPG_ROOT}" ] && echo "Error: This needs O2DPG loaded" && exit 1
[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1


cmd="$PWD/epos.sh"
NEV=-1
more=""
optns="example"
TF=1
eCM=-1
JOBS=2

usage()
{
    cat <<EOF
Usage: $0 [OPTIONS]

Options:

  -m,--more    CONFIG      More configurations ($more)
  -n,--nevents EVENTS      Number of events ($nev)
  -i,--input   INPUT       Options file fed to EPOS4 ($optns)
  -j,--jobs    JOBS        Number of jobs ($JOBS)
  -h,--help                Print these instructions
  -e,--ecm     ENERGY      Center-of-Mass energy
  -t,--tf      TF          Timeframes ($TF)
  --                       Rest of command line sent to o2-sim

COMMAND must be quoted if it contains spaces or other special
characters

Below follows the help output of o2dpg_sim_workflow

EOF
}

if [ "$#" -lt 2 ]; then
    echo "Running with default values"
fi

while test $# -gt 0 ; do
    case $1 in
        -m|--more)    more="$2" ; shift ;;
        -n|--nevents) NEV=$2 ; shift ;;
        -i|--input)   optns=$2 ; shift ;;
        -j|--jobs)    JOBS=$2 ; shift ;;
        -e|--ecm)     eCM=$2 ; shift ;;
        -h|--help) usage; ${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py --help  ; exit 0 ;;
        -t|--tf)     TF=$2 ; shift ;;
        --)           shift ; break ;;
        *) echo "Unknown option '$1', did you forget '--'?" >/dev/stderr
           exit 3
           ;;
    esac
    shift
done

echo "Options file: $optns"

if [ ! -f $optns.optns ]; then
    echo "Error: Options file $optns.optns not found"
    exit 4
fi

# Set number of events in optns file
if [ ! $NEV -eq -1 ]; then
    echo "Setting number of events to $NEV"
    if grep -Fq "nfull" $optns.optns; then
        sed -i "/nfull/c\set nfull $NEV" $optns.optns
    else
        echo "set nfull $NEV" >> $optns.optns
    fi
else
    echo "Number of events not set, checking optns file..."
    if grep -Fq "nfull" $optns.optns; then
        NEV=$(grep -F "nfull" $optns.optns | awk '{print $3}')
        echo "Number of events set to $NEV"
    else
        echo "Error: Number of events not set in EPOS4"
        exit 5
    fi
fi

# Set ECM

if [ ! $eCM -eq -1 ]; then
    echo "Setting eCM to $eCM"
    if grep -Fq "ecms" $optns.optns; then
        sed -i "/ecms/c\set ecms $eCM" $optns.optns
    else
        echo "set ecms $eCM" >> $optns.optns
    fi
else
    echo "Energy not set, checking optns file..."
    if grep -Fq "ecms" $optns.optns; then
        eCM=$(grep -F "ecms" $optns.optns | awk '{print $3}')
        echo "Energy set to $eCM"
    else
        echo "Error: eCM not set in EPOS4"
        exit 6
    fi
fi

# Copy options file in each timeframe folder
for i in $(seq 1 $TF); do
    if [ ! -d tf$i ]; then
        mkdir tf$i
    fi
    cp $optns.optns tf$i/$optns.optns
done

# create workflow

${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py -eCM $eCM -ns $NEV -gen hepmc -tf $TF -j $JOBS \
        -interactionRate 500000 -confKey "GeneratorFileOrCmd.cmd=$cmd -i $optns;GeneratorFileOrCmd.bMaxSwitch=none;HepMC.version=2;${more}"

# Run workflow
${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt aod --stdout-on-failure
