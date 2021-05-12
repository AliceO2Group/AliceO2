#!/bin/bash
echo Running raw file to timeframe converter

if [ `which StfBuilder 2> /dev/null | wc -l` == "0" ]; then
    echo ERROR: StfBuilder is not in the path
    exit 1
fi
if [ `which o2-readout-exe 2> /dev/null | wc -l` == "0" ]; then
    echo ERROR: o2-readout-exe is not in the path
    exit 1
fi

pushd raw

rm -Rf timeframe rdo_TF.cfg *.log
if [ `ls | grep -v "^[A-Z0-9]\{3\}\$" | wc -l` != "0" ]; then
    echo Unexpected data in raw folder
    exit 1
fi

echo Generating readout config
$O2_ROOT/prodtests/full-system-test/gen_rdo_cfg.sh 128 *

echo Starting StfBuilder
StfBuilder --id=stfb --detector-rdh=6 --detector-subspec=feeid --stand-alone  --channel-config "name=readout,type=pull,method=connect,address=ipc:///tmp/readout-to-datadist-0,transport=shmem,rateLogging=1" --data-sink-dir=`pwd` --data-sink-sidecar --data-sink-enable --control=static &> stfbuilder.log &
STF_PID=$!
echo StfBuilder PID: $STF_PID, waiting 15 seconds
sleep 15

echo Starting Readout
export O2_INFOLOGGER_OPTIONS="floodProtection=0"
o2-readout-exe file:rdo_TF.cfg &> readout.log &
RD_PID=$!
echo Readout PID: $RD_PID

echo Waiting for data to arrive
while [ `ls run*_20*/run*_tf00000001.tf 2> /dev/null | wc -l` == "0" ]; do
    sleep 1
done
echo Data is arriving, waiting 20 seconds to be sure
sleep 20

echo Killing Readout
kill $RD_PID

sleep 10
echo Killing StfBuilder
kill $STF_PID

sleep 10
if [ -d /proc/$RD_PID ]; then
    kill -9 $RD_PID
fi
if [ -d /proc/$STF_PID ]; then
    kill -9 $STF_PID
fi

mv 20* timeframe
rm -f *.log rdo_TF.cfg

echo Done

popd
