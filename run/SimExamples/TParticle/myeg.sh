#!/bin/sh


nev=1
out=particles.root
seed=0
opt=

usage()
{
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -n NUMBER   Number of events
  -o FILENAME Output file name
  -s SEED     Randon number seed
  -b FERMI    Maximum impact parameter
  -+          AcLIC the script
EOF
}

while test $# -gt 0 ; do
    case $1 in
        -h) usage ; exit 0;;
        -n) nev=$2 ; shift ;;
        -o) out=$2 ; shift ;;
        -s) seed=$2 ; shift ;;
        -b) imp=$2 ; shift ;;
        -+) opt=+ ;;
        *) ;;
    esac
    shift
done

root -q -l MyEG.macro${opt} -- $nev \"$out\" $seed

