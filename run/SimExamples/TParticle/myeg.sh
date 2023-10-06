#!/bin/sh


nev=1
out=particles.root
seed=0
opt=

while test $# -gt 0 ; do
    case $1 in
	-n) nev=$2 ; shift ;;
	-o) out=$2 ; shift ;;
	-s) seed=$2 ; shift ;;
	-b) imp=$2 ; shift ;;
	-+) opt=+ ;;
	*) ;;
    esac
    shift
done

root -q -l MyEG.cc${opt} -- $nev \"$out\" $seed

