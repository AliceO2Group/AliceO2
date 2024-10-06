#!/bin/sh
# Script based on CRMC example
# EPOS4 option files must contain ihepmc set to 2 to print HepMC
# data on stdout. -hepmc flag is not needed anymore, but -hepstd is fundamental
# in order not to print useless information on stdout (a z-*optns*.mtr file will be created)

optns="example"
seed=$RANDOM

while test $# -gt 0 ; do
    case $1 in
        -i|--input)   optns=$2 ; shift ;;
        -s|--seed)    seed=$2 ; shift ;;
        -h|--help) usage; exit 0 ;;
    esac
    shift
done

if [ ! -f $optns.optns ]; then
    echo "Error: Options file $optns.optns not found"
    exit 1
fi

if [ $seed -eq 0 ]; then
    echo "Seed can't be 0, random number will be used"
    seed=$RANDOM
fi

# Or filters the stdout with only HepMC2 useful data
$EPOS4_ROOT/epos4/scripts/epos -hepstd -s $seed $optns | sed -n 's/^\(HepMC::\|[EAUWVP] \)/\1/p'
