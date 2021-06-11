#!/usr/bin/env bash

set -u

display_usage() {
    echo -e "\nSpecify tf length and directory name containing raw files"
    echo -e "\nUsage:\ngen_rdo_cfg.sh <tf_len> <dir> \n"
}

if [[  ($# -le 1) || ( "$1" == "--help") ||  ("$1" == "-h") ]]; then
    display_usage
    exit 1
fi

#TF LENGTH
TF_LEN=$1

function file_input {
    name="${1//[-.\/]/_}"
    size=$(stat --printf="%s" $1)
    # calculate number of pages to hold the files
    page_cnt=$(( (size * 3 / 2)  / 524288 ))
    page_cnt=$(( page_cnt < 70 ? 70 : page_cnt))

    echo "[equipment-player-$name]"
    echo "## file size = ${size}"
    echo "## page cnt = ${page_cnt}"

    echo "equipmentType = player"
    echo "filePath = $1"
    echo "enabled = 1"
    echo "autoChunk = 1"
    echo "TFperiod = $2"
    echo "memoryPoolNumberOfPages = ${page_cnt}"
    echo "memoryPoolPageSize = 512k"
    echo ""
}


cat > rdo_TF.cfg <<- EOM
###################################
# general settings
###################################
[readout]

# per-equipment data rate limit, in Hertz (-1 for unlimited)
rate=-1
# time (in seconds) after which program exits automatically (-1 for unlimited)
exitTimeout=-1

### !!! Agreggating (buffering) for TF ordering
### 5 seconds buffering/sorting of individual sources

disableAggregatorSlicing=0
aggregatorSliceTimeout=12
aggregatorStfTimeout=10

[consumer-stats]
consumerType = stats
enabled = 1
monitoringEnabled = 0
monitoringUpdatePeriod = 1
processMonitoringInterval = 1
consoleUpdate = 1

[consumer-StfBuilder]
consumerType = FairMQChannel
enabled = 1
sessionName = default
fmq-transport = shmem
fmq-name = readout-out
fmq-type = push
fmq-address = ipc:///tmp/readout-to-datadist-0
unmanagedMemorySize = 96G
memoryBankName = bank-o2
memoryPoolNumberOfPages = 16000
memoryPoolPageSize = 1024k
disableSending = 0

###################################
# Begin file players
###################################
EOM


# iterate over directories
for dir in "${@:2}"
do
    for rawfile in $dir/*.raw
    do
        file_input "$rawfile" "$TF_LEN" >> rdo_TF.cfg
    done
done
