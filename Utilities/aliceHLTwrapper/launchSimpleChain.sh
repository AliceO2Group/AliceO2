#!/bin/bash

# very simple helper script to create the commands for starting several processes
# in multiple screen sessions

minSlice=0
maxSlice=0
minPart=0
maxPart=5
runNo=167808
msgSize=1000

firstSocketNo=45000
let socketNo=firstSocketNo
trackerInputSockets=

exit_with_message () {
    echo $1
    exit
}

start_device () {
    echo "starting $1 for specification $spec in screen \"$sessiontitle\""
    # start the screen session
    screen -S "$sessiontitle" -t 0 -A -d -m
    # propagate the environment to the screen window
    env | sed -e '/^[A-Z]/!d' -e 's|=|="|' -e 's|\(.\)$|\1"|' | while read line ; do screen -S "$sessiontitle" -p 0 -X stuff $"export ${line}\n"; done
    # change to the directory
    screen -S "$sessiontitle" -p 0 -X stuff $"cd ${PWD}\n"
    # start the device
    # Note: there seems to be a limit how many characters are sent
    # to the screen with the -X option. Probably one has to break down the
    # socket configuration string
    echo $command $sockets $parameters
    screen -S "$sessiontitle" -p 0 -X stuff $"${command} "
    screen -S "$sessiontitle" -p 0 -X stuff $"${sockets} "
    screen -S "$sessiontitle" -p 0 -X stuff $"${parameters}\n"
}


WrapperDeviceApplication=AliceHLTWrapperDevice

# check whether the required executables are available
which screen > /dev/null 2>&1 || exit_with_message "This script requires the 'screen' command to be installed"
which $WrapperDeviceApplication > /dev/null 2>&1 || exit_with_message "Can not find the $WrapperDeviceApplication executable"

for ((slice=minSlice; slice<=maxSlice; slice++)); do
  for ((part=minPart; part<=maxPart; part++)); do
    spec=`printf 0x%02x%02x%02x%02x $slice $slice $part $part`
    command="$WrapperDeviceApplication --id=ClusterPublisher_$spec --channel-config name=data-out,type=push,method=bind,address=tcp://*:$socketNo --library libAliHLTUtil.so --component FilePublisher --run 167808 --parameter '-datafilelist emulated-tpc-clusters_$spec.txt'"
    sessiontitle="ClusterPublisher_$spec"
    start_device ClusterPublisher
    trackerInputSockets=`echo "$trackerInputSockets $socketNo"`
    let socketNo++
  done
done

command="$WrapperDeviceApplication --id=Tracker"
parameters="--library libAliHLTTPC.so --component TPCCATracker --run $runNo --parameter '-GlobalTracking'"
sockets=""
for socket in $trackerInputSockets; do
    sockets=`echo "$sockets --channel-config name=input,type=pull,method=connect,address=tcp://localhost:$socket"`
done
spec=`printf 0x%02x%02x%02x%02x $maxSlice $minSlice $maxPart $minPart`
sessiontitle="Tracker"
start_device Tracker

echo
screen -ls
