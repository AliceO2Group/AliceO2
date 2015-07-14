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
number_of_flps=36
flp_command_socket=48490
flp_heartbeat_socket=48491
baseport_on_flpgroup=48420
baseport_on_epngroup=48480
number_of_epns=28
number_of_epns_per_node=1
rundir=`pwd`

###################################################################
# argument scan
# TODO: implement real argument scan and more configurable options
while [ "x$1" != "x" ]; do
    if [ "x$1" == "x--print-commands" ]; then
	printcmdtoscreen='echo'
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
flpnodelist=
flpnodelist=(${flpnodelist[@]} cn48$ibkey)
flpnodelist=(${flpnodelist[@]} cn49$ibkey)
flpnodelist=(${flpnodelist[@]} cn50$ibkey)
flpnodelist=(${flpnodelist[@]} cn51$ibkey)
flpnodelist=(${flpnodelist[@]} cn52$ibkey)
flpnodelist=(${flpnodelist[@]} cn53$ibkey)
flpnodelist=(${flpnodelist[@]} cn54$ibkey)
flpnodelist=(${flpnodelist[@]} cn55$ibkey)
flpnodelist=(${flpnodelist[@]} cn56$ibkey)
flpnodelist=(${flpnodelist[@]} cn57$ibkey)
flpnodelist=(${flpnodelist[@]} cn58$ibkey)

#flpnodelist=(${flpnodelist[@]} cn00$ibkey)
flpnodelist=(${flpnodelist[@]} cn01$ibkey)
flpnodelist=(${flpnodelist[@]} cn02$ibkey)
flpnodelist=(${flpnodelist[@]} cn03$ibkey)
flpnodelist=(${flpnodelist[@]} cn04$ibkey)
flpnodelist=(${flpnodelist[@]} cn05$ibkey)
flpnodelist=(${flpnodelist[@]} cn06$ibkey)
flpnodelist=(${flpnodelist[@]} cn07$ibkey)
flpnodelist=(${flpnodelist[@]} cn08$ibkey)
flpnodelist=(${flpnodelist[@]} cn09$ibkey)
flpnodelist=(${flpnodelist[@]} cn10$ibkey)
flpnodelist=(${flpnodelist[@]} cn11$ibkey)
flpnodelist=(${flpnodelist[@]} cn12$ibkey)
flpnodelist=(${flpnodelist[@]} cn13$ibkey)
flpnodelist=(${flpnodelist[@]} cn14$ibkey)
flpnodelist=(${flpnodelist[@]} cn15$ibkey)
flpnodelist=(${flpnodelist[@]} cn16$ibkey)
flpnodelist=(${flpnodelist[@]} cn17$ibkey)
flpnodelist=(${flpnodelist[@]} cn18$ibkey)
flpnodelist=(${flpnodelist[@]} cn19$ibkey)
flpnodelist=(${flpnodelist[@]} cn26$ibkey)
flpnodelist=(${flpnodelist[@]} cn27$ibkey)
flpnodelist=(${flpnodelist[@]} cn28$ibkey)
#flpnodelist=(${flpnodelist[@]} cn29$ibkey)
flpnodelist=(${flpnodelist[@]} cn30$ibkey)
flpnodelist=(${flpnodelist[@]} cn31$ibkey)
flpnodelist=(${flpnodelist[@]} cn32$ibkey)
flpnodelist=(${flpnodelist[@]} cn33$ibkey)
flpnodelist=(${flpnodelist[@]} cn34$ibkey)
flpnodelist=(${flpnodelist[@]} cn35$ibkey)

# nodes with GPU
epnnodelist=
#epnnodelist=(${epnnodelist[@]} 10.162.130.20) # cn00 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.21) # cn01 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.22) # cn02 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.23) # cn03 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.24) # cn04 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.25) # cn05 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.26) # cn06 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.27) # cn07 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.28) # cn08 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.29) # cn09 infiniband

epnnodelist=(${epnnodelist[@]} 10.162.130.30) # cn10 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.31) # cn11 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.32) # cn12 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.33) # cn13 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.34) # cn14 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.35) # cn15 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.36) # cn16 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.37) # cn17 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.38) # cn18 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.39) # cn19 infiniband

epnnodelist=(${epnnodelist[@]} 10.162.130.46) # cn26 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.47) # cn27 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.48) # cn28 infiniband
#epnnodelist=(${epnnodelist[@]} 10.162.130.49) # cn29 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.50) # cn30 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.51) # cn31 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.52) # cn32 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.53) # cn33 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.54) # cn34 infiniband
epnnodelist=(${epnnodelist[@]} 10.162.130.55) # cn35 infiniband


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
epnsessiondataout=
nepnsessions=0

###################################################################

###################################################################
# create an flp node group
create_flpgroup() {
    node=$1
    basesocket=$2

    # for now only one flp per node
        deviceid="FLP_$node"
	command="flpSender"
	command+=" --id $deviceid"
	command+=" --num-inputs 3"
	command+=" --num-outputs $nepnsessions"
	command+=" --heartbeat-timeout 20000"
	command+=" --send-offset $((1+3*$nflpsessions))"
	command+=" --input-socket-type sub --input-buff-size 500 --input-method bind --input-address tcp://*:$flp_command_socket   --input-rate-logging 0" # command input
	command+=" --input-socket-type sub --input-buff-size 500 --input-method bind --input-address tcp://*:$flp_heartbeat_socket --input-rate-logging 0" # heartbeat input
	command+=" "
	flpinputsocket[nflpsessions]=$((basesocket + 0))
	command+=" --input-socket-type pull --input-buff-size 5000 --input-method bind --input-address tcp://*:${flpinputsocket[$nflpsessions]} --input-rate-logging 1" # data input
	for ((j=0; j<$nepnsessions; j++));
	do
            command+=" --output-socket-type push --output-buff-size 5000 --output-method connect --output-address tcp://${epnsessionnode[$j]}:${epninputsocket[$j]} --output-rate-logging 1"
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
	command+=" --num-outputs $((number_of_flps + 1))"
	command+=" --heartbeat-interval 5000"
	command+=" --buffer-timeout 60000"
	command+=" --num-flps $number_of_flps"
	epninputsocket[nepnsessions]=$((2*nepn + basesocket))
	command+=" --input-socket-type pull --input-buff-size 5000 --input-method bind --input-address tcp://$node:${epninputsocket[$nepnsessions]} --input-rate-logging 1" # data input
	epnoutputsocket[nepnsessions]=$((2*nepn + basesocket +1))
        
	epnsessionnode[nepnsessions]=$node
        epnsessiontitle[nepnsessions]="$deviceid"
        epnsessioncmd[nepnsessions]=$command
	# have to postpone adding the data output socket because of fixed order in the device outputs
        epnsessiondataout[nepnsessions]=" --output-socket-type push --output-buff-size 5000 --output-method bind --output-address tcp://*:${epnoutputsocket[$nepnsessions]} --output-rate-logging 1" # data output
        let nepnsessions++

# TBD after the FLP creation
#	for flpnode in ${epn1_input[@]}; # heartbeats
#	do
#	    command+=" --output-socket-type pub --output-buff-size 500 --output-method connect --output-address tcp://$flpnode:$flp_heartbeat_socket --output-rate-logging 0"
#	done
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
    create_flpgroup ${flpnodelist[$inode]} $baseport_on_flpgroup

    let inode++
done

if [ "$error" -gt 0 ]; then
    exit
fi

# now set the heartbeat channels
for ((iepnsession=0; iepnsession<$nepnsessions; iepnsession++)); do
    for ((iflpsession=0; iflpsession<$nflpsessions; iflpsession++)); do
	epnsessioncmd[$iepnsession]="${epnsessioncmd[$iepnsession]} --output-socket-type pub --output-buff-size 500 --output-method connect --output-address tcp://${flpsessionnode[$iflpsession]}:$flp_heartbeat_socket --output-rate-logging 0"
    done
    # now add the data output, note that there is a fixed order in the device outputs
    epnsessioncmd[$iepnsession]="${epnsessioncmd[$iepnsession]} ${epnsessiondataout[$iepnsession]}"
done

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
