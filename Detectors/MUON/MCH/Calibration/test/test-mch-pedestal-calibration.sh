#! /bin/bash

export INFOLOGGER_MODE=stdout

#INPUT=Data/StfBuilder-CH5-pedestals-ul_v1-20210311-with_gnd-with_HV
INPUT="$1"

ARGS_ALL="-b --session default"
PROXY_INSPEC="A:MCH/RAWDATA;dd:FLP/DISTSUBTIMEFRAME/0;eos:***/INFORMATION"
DECOD_INSPEC="TF:MCH/RAWDATA;dd:FLP/DISTSUBTIMEFRAME/0;eos:***/INFORMATION"
CALIB_INSPEC="digits:MCH/PDIGITS;dd:FLP/DISTSUBTIMEFRAME/0;eos:***/INFORMATION"

xterm -bg black -fg white -geometry 100x50+0+100 -e ./run-tfbuilder.sh "$INPUT" &

sleep 10

o2-dpl-raw-proxy ${ARGS_ALL} --dataspec "${PROXY_INSPEC}" \
  --readout-proxy '--channel-config "name=readout-proxy,type=pull,method=connect,address=ipc://@tf-builder-pipe-0,transport=shmem,rateLogging=1"' \
 | o2-mch-pedestal-decoding-workflow ${ARGS_ALL} --input-spec "${DECOD_INSPEC}" --logging-interval 1 \
 | o2-calibration-mch-pedestal-calib-workflow ${ARGS_ALL} --input-spec "${CALIB_INSPEC}" \
     --pedestal-threshold 200 --noise-threshold 2.0 --logging-interval 1 \
 | o2-dpl-run ${ARGS_ALL} --run

# outputs to QC and CCDB
#| o2-qc -b --config json:/${SCRIPTDIR}/qc-pedestals.json 
#| o2-calibration-ccdb-populator-workflow -b --session default
