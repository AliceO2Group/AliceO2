#!/bin/bash

[[ -z $SHMSIZE ]] && export SHMSIZE=$(( 112 << 30 )) # Please keep these defaults in sync with those in start_tmux.sh
[[ -z $DDSHMSIZE ]] && export DDSHMSIZE=$(( 112 << 10 ))

[[ -z $GEN_TOPO_MYDIR ]] && GEN_TOPO_MYDIR="$(dirname $(realpath $0))"
source $GEN_TOPO_MYDIR/setenv.sh || { echo "setenv.sh failed" 1>&2 && exit 1; }

o2-epn-shm-manager --shmid $SHM_MANAGER_SHMID --segments 0,$SHMSIZE,0 1,$SHMSIZE,1 2,$((10<<20)),-1 --regions 100,$(($DDSHMSIZE << 20)),-1 101,$(($DDHDRSIZE << 20)),-1 --nozero
