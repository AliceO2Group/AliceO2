#! /bin/bash

#****************************************************************************
#* This file is free software: you can redistribute it and/or modify        *
#* it under the terms of the GNU General Public License as published by     *
#* the Free Software Foundation, either version 3 of the License, or        *
#* (at your option) any later version.                                      *
#*                                                                          *
#* Primary Authors: Matthias Richter <Matthias.Richter@scieq.net>           *
#*                                                                          *
#* The authors make no claims about the suitability of this software for    *
#* any purpose. It is provided "as is" without express or implied warranty. *
#****************************************************************************

#  @file   launch-flp-epn.sh
#  @author Matthias Richter
#  @since  2015-01-14
#  @brief  Launch script for a FLP to EPN data distribution topology


###################################################################
# global settings
number_of_flps=2
flp_command_socket=48490
epn_ack_socket=48491
baseport_on_flpgroup=48420
baseport_on_epngroup=48480
number_of_epns=2
number_of_epns_per_node=2
rundir=`pwd`

###################################################################
# argument scan
# TODO: implement real argument scan and more configurable options
while [ "x$1" != "x" ]; do
    if [ "x$1" == "x--print-commands" ]; then
        printcmdtoscreen='echo'
    elif [ "x$1" == "x--non-blocking-output" ] ; then
        # use method 'pub' for non-blocking EPN output
        outputmethod=pub
        shift
    else
	echo unknown option $1
	exit
    fi
    shift
done

###################################################################
######## end of configuration area                     ############
######## no changes beyond this point                  ############
###################################################################

###################################################################
# fill the list of nodes
#
# for every FLP group there needs to be an entry in the node list
# nodes can be specified multiple times
flpnodelist=
flpnodelist=(${flpnodelist[@]} localhost)
flpnodelist=(${flpnodelist[@]} localhost)

epnnodelist=
epnnodelist=(${epnnodelist[@]} 127.0.0.1)

nflpnodes=${#flpnodelist[@]}
nepnnodes=${#epnnodelist[@]}

###################################################################
# init the variables for the session commands
flpsessionnode=
flpinputsocket=
flpsessiontitle=
flpsessioncmd=
nflpsessions=0

epnsessionnode=
epninputsocket=
epnoutputsocket=
epnsessiontitle=
epnsessioncmd=
nepnsessions=0

# the same node name can be specified for multiple FLP groups
# this map holds a count of groups per node while adding the groups
declare -A flpgroupsPerNode

###################################################################

###################################################################
# create an flp node group
create_flpgroup() {
    node=$1
    basesocket=$2

    deviceid="FLP_$node${3:+_${3}}"
    command="flpSender"
    command+=" --id $deviceid"
    command+=" --control static"
    command+=" --flp-index 0"
    command+=" --num-epns $nepnsessions"
    command+=" --send-offset $((1+3*$nflpsessions))"
    command+=" "
    command+=" --in-chan-name data-in"
    flpinputsocket[nflpsessions]=$((basesocket + 0))
    command+=" --channel-config name=data-in,type=pull,size=5000,method=bind,address=tcp://*:${flpinputsocket[$nflpsessions]},rateLogging=1" # data input
    command+=" --out-chan-name data-out"
    for ((j=0; j<$nepnsessions; j++));
    do
        command+=" --channel-config name=data-out,type=push,size=5000,method=connect,address=tcp://${epnsessionnode[$j]}:${epninputsocket[$j]},rateLogging=1"
    done

    flpsessionnode[nflpsessions]=$node
    flpsessiontitle[nflpsessions]=$deviceid
    flpsessioncmd[nflpsessions]=$command
    let nflpsessions++
}

###################################################################
# create an epn node group
create_epngroup() {
    node=$1
    basesocket=$2

    for ((nepn=0; nepn<$number_of_epns_per_node; nepn++)); do
        deviceid=EPN_`printf %02d $nepnsessions`
        command="epnReceiver"
        command+=" --id $deviceid"
        command+=" --control static"
        command+=" --num-flps $number_of_flps"
        epninputsocket[nepnsessions]=$((2*nepn + basesocket))
        command+=" --in-chan-name data-in"
        command+=" --out-chan-name data-out"
        command+=" --channel-config name=data-in,type=pull,size=5000,method=bind,address=tcp://$node:${epninputsocket[$nepnsessions]},rateLogging=1" # data input
        command+=" --channel-config name=ack,type=pub,method=connect,address=tcp://localhost:$epn_ack_socket,rateLogging=1" # data input
        epnoutputsocket[nepnsessions]=$((2*nepn + basesocket +1))
        command+=" --channel-config name=data-out,type=${outputmethod:=push},size=5000,method=bind,address=tcp://*:${epnoutputsocket[$nepnsessions]},rateLogging=1" # data output
        
        epnsessionnode[nepnsessions]=$node
        epnsessiontitle[nepnsessions]="$deviceid"
        epnsessioncmd[nepnsessions]=$command
        let nepnsessions++
    done
}

########### main script ###########################################
#
# build the commands on the nodes
# epn nodegroups
error=0
inode=0
while [ "$nepnsessions" -lt "$number_of_epns" ]; do
    if [ "$inode" -ge "$nepnnodes" ]; then
        echo "error: too few nodes to create all epn devices"
        error=1
        break
    fi
    create_epngroup ${epnnodelist[$inode]} $baseport_on_epngroup

    let inode++
done

inode=0
while [ "$nflpsessions" -lt "$number_of_flps" ]; do
    if [ "$inode" -ge "$nflpnodes" ]; then
        echo "error: too few nodes to create all flp devices"
        error=1
        break
    fi
    key=${flpnodelist[$inode]}
    # keep record of the count of FLP groups on a node in a dedicated map and
    # calculate the input port based on that. There is only one input socket
    # per FLP group, multiple devices can connect to the same socket
    create_flpgroup ${flpnodelist[$inode]} $(( baseport_on_flpgroup + ${flpgroupsPerNode[$key]:=0} )) ${flpgroupsPerNode[$key]:=0}
    flpgroupsPerNode[$key]=$(( flpgroupsPerNode[$key] + 1 ))

    let inode++
done

if [ "$error" -gt 0 ]; then
    exit
fi

sessionmap=
for ((isession=$nflpsessions++-1; isession>=0; isession--)); do
    sessionmap[isession]=1
done
havesessions=1

applications=
while [ "$havesessions" -gt 0 ]; do
havesessions=0
lastnode=
for ((isession=$nflpsessions++-1; isession>=0; isession--)); do
    if [ "${sessionmap[isession]}" -eq 1 ]; then
    echo "FLP_DEVICE_IN=${flpsessionnode[$isession]}:${flpinputsocket[$isession]}"
    if [ "x$printcmdtoscreen" == "x" ]; then
        echo "scheduling ${flpsessiontitle[$isession]} on ${flpsessionnode[$isession]}: ${flpsessioncmd[$isession]}"
        echo
    fi
    applications+=" "`echo ${flpsessioncmd[$isession]} | sed -e 's| .*$||'`
    fi

    if [ "${sessionmap[isession]}" -gt 0 ]; then
        #echo $isession: ${sessionmap[isession]} $lastnode
        if [ "x$lastnode" == "x${flpsessionnode[$isession]}" ] && [ "${sessionmap[$isession]}" -lt 10 ]; then
            let sessionmap[isession]++
            let havesessions++
        else
            if [ "x$lastnode" == "x${flpsessionnode[$isession]}" ]; then
                # sleep between starts, some of the screens are not started if the frequency is too high
                usleep 500000
            fi

            logcmd=" 2>&1 | tee ${flpsessiontitle[$isession]}.log"
            $printcmdtoscreen screen -d -m -S "${flpsessiontitle[$isession]} on ${flpsessionnode[$isession]}" ssh ${flpsessionnode[$isession]} "cd $rundir && source setup.sh && ${flpsessioncmd[$isession]} $logcmd" &
            sessionmap[isession]=0
            lastnode=${flpsessionnode[$isession]}
        fi
    fi
done
done

for ((isession=$nepnsessions++-1; isession>=0; isession--)); do
    sessionmap[isession]=1
done
havesessions=1

while [ "$havesessions" -gt 0 ]; do
havesessions=0
lastnode=
for ((isession=$nepnsessions++-1; isession>=0; isession--)); do
    if [ "${sessionmap[isession]}" -eq 1 ]; then
    echo "EPN_DEVICE_OUT=${epnsessionnode[$isession]}:${epnoutputsocket[$isession]}"
    if [ "x$printcmdtoscreen" == "x" ]; then
        echo "scheduling ${epnsessiontitle[$isession]} on ${epnsessionnode[$isession]}: ${epnsessioncmd[$isession]}"
        echo
    fi
    applications+=" "`echo ${epnsessioncmd[$isession]} | sed -e 's| .*$||'`
    fi

    if [ "${sessionmap[isession]}" -gt 0 ]; then
        #echo $isession: ${sessionmap[isession]} $lastnode
        if [ "x$lastnode" == "x${epnsessionnode[$isession]}" ] && [ "${sessionmap[$isession]}" -lt 10 ]; then
            let sessionmap[isession]++
            let havesessions++
        else
            if [ "x$lastnode" == "x${epnsessionnode[$isession]}" ]; then
                # sleep between starts, some of the screens are not started if the frequency is too high
                usleep 500000
            fi
            logcmd=" 2>&1 | tee ${epnsessiontitle[$isession]}.log"
            $printcmdtoscreen screen -d -m -S "${epnsessiontitle[$isession]} on ${epnsessionnode[$isession]}" ssh ${epnsessionnode[$isession]} "cd $rundir && source setup.sh && ${epnsessioncmd[$isession]} $logcmd" &
            sessionmap[isession]=0
            lastnode=${epnsessionnode[$isession]}
        fi
    fi
done
done

if [ "x$printcmdtoscreen" == "x" ]; then
usednodes=`for n in ${flpsessionnode[@]} ${epnsessionnode[@]}; do echo $n; done | sort | uniq`
echo
echo "started FLP-EPN topology in $((nflpsessions + nepnsessions)) session(s) on `echo $usednodes | wc -w` node(s):"
usednodes=`echo $usednodes | sed ':a;N;$!ba;s/\n/ /g'`
echo $usednodes

applications=`for app in $applications; do echo $app; done | sort | uniq`
echo
echo "a simple method to stop the devices:"
for app in $applications; do
    echo "for node in $usednodes; do ssh \$node killall $app; done"
done
fi
