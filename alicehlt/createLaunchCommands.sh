#!/bin/bash

# very simple helper script to create the commands for starting several processes

minSlice=0
maxSlice=0
minPart=0
maxPart=5
runNo=167808
msgSize=1000

firstSocketNo=45000
let socketNo=firstSocketNo
trackerInputSockets=

for ((slice=minSlice; slice<=maxSlice; slice++)); do
  for ((part=minPart; part<=maxPart; part++)); do
    spec=`printf 0x%02x%02x%02x%02x $slice $slice $part $part`
    command="aliceHLTWrapper ClusterPublisher_$spec 1 --output type=push,size=$msgSize,method=connect,address=tcp://localhost:$socketNo --library libAliHLTUtil.so --component FilePublisher --run 167808 --parameter '-datafilelist emulated-tpc-clusters_$spec.txt'"
    echo "xterm -e $command &"
    trackerInputSockets=`echo "$trackerInputSockets $socketNo"`
    let socketNo++
  done
done

command="aliceHLTWrapper Tracker 1 "
for socket in $trackerInputSockets; do
    command=`echo "$command --input type=pull,size=$msgSize,method=bind,address=tcp://*:$socket"`
done
command=`echo "$command --library libAliHLTTPC.so --component TPCCATracker --run $runNo --parameter '-GlobalTracking'"`
echo "xterm -e $command &"
