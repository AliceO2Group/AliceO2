#! /bin/bash
TRANSPORT=shmem
#TRANSPORT=zeromq

StfBuilder --id stf_builder-0 --transport $TRANSPORT --detector MCH --dpl-channel-name=dpl-chan --channel-config "name=dpl-chan,type=push,method=bind,address=ipc://@tf-builder-pipe-0,transport=$TRANSPORT,rateLogging=1" --data-source-dir="$1" --data-source-enable --control=static #--max-built-stfs 10
