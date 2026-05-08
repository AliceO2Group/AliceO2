#!/usr/bin/env bash
#
# This is a simple example script to bypass the Hyperloop limitations in using
# external generators only, by switching the generator to the hybrid mode

# This script works only with updated O2sim version containing the switchExtToHybrid option

[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 2
[ ! "${O2DPG_MC_CONFIG_ROOT}" ] && echo "Error: This needs O2DPG loaded" && exit 2

NEV=5
# Two example ini configurations are provided pointing to different hybrid JSON files
# One creates a cocktail based on Pythia8, while the other generates sequentially EPOS4 and boxgen events
ini="${O2_ROOT}/examples/ExternalToHybrid/GeneratorHyperloopHybridCocktail.ini"

usage()
{
    cat <<EOF
Usage: $0 [OPTIONS]

Options:

  -i,--ini     INI         Configuration ini file ($ini)
  -n,--nevents EVENTS      Number of events ($nev)
  -h,--help                Print these instructions
  --                       Rest of command line sent to o2-sim

COMMAND must be quoted if it contains spaces or other special
characters

Below follows the help output of o2-sim-dpl-eventgen

EOF
}

if [ "$#" -lt 2 ]; then
    echo "Running with default values"
fi

while test $# -gt 0 ; do
    case $1 in
        -i|--ini)     ini="$2" ; shift ;;
        -n|--nevents) NEV=$2 ; shift ;;
        -h|--help) usage; o2-sim-dpl-eventgen --help full ; exit 0 ;;
        --)           shift ; break ;;
        *) echo "Unknown option '$1', did you forget '--'?" >/dev/stderr
           exit 3
           ;;
    esac
    shift
done

# Starting the dpl-eventgen simulation
o2-sim-dpl-eventgen -b --generator external --nEvents $NEV --configFile $ini