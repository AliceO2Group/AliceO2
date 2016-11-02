#!/bin/bash

WORKDIR=/home/bogdan/Software/AliceO2_mft/build_o2

TRANSPORT="zeromq"
TASKNAME="FindHits"
BRANCHNAME="MFTHits"

OUTPUTCLASS="--class-name TClonesArray(AliceO2::MFT::Hit)"
OUTPUTBRANCH="--branch-name MFTHits"

MQCONFIGFILE=${WORKDIR}/reco.json

INPUTFILE="${WORKDIR}/AliceO2_TGeant3.mc_1ev_100mu.root"
INPUTBRANCH="MFTPoints"
    
# output file for sink
OUTPUTFILE="${WORKDIR}/hits.root"
OUTPUTCLASS="--class-name TClonesArray(AliceO2::MFT::Hit)"
OUTPUTBRANCH="--branch-name MFTHits"

SERVER="${WORKDIR}/bin/mft-reco-sampler"
SERVER+=" --transport $TRANSPORT"
SERVER+=" --id sampler"
SERVER+=" --mq-config $MQCONFIGFILE"
SERVER+=" --file-name $INPUTFILE"
SERVER+=" --branch-name $INPUTBRANCH"

xterm -geometry 80x25+0+0 -hold -e $SERVER &

PROCESSOR="${WORKDIR}/bin/mft-reco-processor"
PROCESSOR+=" --transport $TRANSPORT"
PROCESSOR+=" --id processor"
PROCESSOR+=" --mq-config $MQCONFIGFILE"
PROCESSOR+=" --task-name $TASKNAME"
PROCESSOR+=" --keep-data $BRANCHNAME"

xterm +aw -geometry 80x25+500+0 -hold -e $PROCESSOR &

FILESINK="${WORKDIR}/bin/mft-reco-sink"
FILESINK+=" --transport $TRANSPORT"
FILESINK+=" --id sink"
FILESINK+=" --mq-config $MQCONFIGFILE"
FILESINK+=" --file-name hits.root $OUTPUTCLASS $OUTPUTBRANCH"

xterm +aw -geometry 80x25+500+350 -hold -e $FILESINK &
