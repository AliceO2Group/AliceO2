#!/bin/bash

MYDIR="$(dirname $(realpath $0))"
source $MYDIR/setenv.sh

o2-epn-shm-manager --shmid $SHM_MANAGER_SHMID --segments 0,$SHMSIZE 1,$SHMSIZE 2,$((10<<20)) --regions 100,$DDSHMSIZE 101,$DDHDRSIZE
