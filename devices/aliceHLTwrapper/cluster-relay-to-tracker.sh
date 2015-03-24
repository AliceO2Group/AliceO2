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

#  @file   cluster-relay-to-tracker.sh
#  @author Matthias Richter
#  @since  2014-11-26 
#  @brief  Launch script for ALICE HLT TPC processing topology


###################################################################
# global settings
runno=167808

# note: don't want to overdo the logic now, the lastslice should be firstslice plus multiples of slices_per_node minus 1
# lastslice=(firstslice + n*slices_per_node) - 1
#
# TODO: this possibly needs some more checking 
firstslice=0
lastslice=35
slices_per_node=1
#dryrun="-n"
pollingtimeout=100
CFoptOpenFilesAtStart= #" -open_files_at_start"
CFoptPublishIndividualPartitions=no #yes
rundir=`pwd`

baseport_on_flpgroup=48400
baseport_on_epn1group=48450
baseport_on_epn2group=48470

# uncomment the following line to print the commands instead of
# actually launching them
#printcmdtoscreen='echo'

# uncomment the following line to bypass the FLP relays and
# send data directly to th tracker
#bypass_relays=yes

# uncomment the following line to bypass the tracking
#bypass_tracking=yes

###################################################################
# argument scan
# TODO: implement real argument scan and more configurable options
while [ "x$1" != "x" ]; do
    if [ "x$1" == "x--print-commands" ]; then
	printcmdtoscreen='echo'
    fi
    if [ "x$1" == "x--polling-timeout" ] && [ "x$2" != "x" ] ; then
	pollingtimeout=$2
	shift
    fi
    shift
done

###################################################################
######## end of configuration area                     ############
######## no changes beyond this point                  ############
###################################################################

###################################################################
# fill the list of nodes either from standard input or subsidiary
# script in the current directory
# 
nodelist=
epn2node=localhost

flpinputnode=
flpinputsocket=
nflpinputs=0
epnoutputnode=
epnoutputsocket=
nepnoutputs=0
postponed_messages=
npostponed_messages=0
while read line; do
    flpdevice=`echo $line | sed -e '/^FLP_DEVICE_IN=/!d' -e 's|^FLP_DEVICE_IN=||'`
    epndevice=`echo $line | sed -e '/^EPN_DEVICE_OUT=/!d' -e 's|^EPN_DEVICE_OUT=||'`
    if [ "x$flpdevice" != "x" ]; then
	echo "FLPINPUT: $flpdevice"
	flpinputnode[nflpinputs]=`echo ${flpdevice} | sed -e 's/:.*//'`
	flpinputsocket[nflpinputs]=`echo ${flpdevice} | sed -e 's/.*://'`
	let nflpinputs++
    elif [ "x$epndevice" != "x" ]; then
	echo "EPNOUTPUT $epndevice"
	epnoutputnode[nepnoutputs]=`echo ${epndevice} | sed -e 's/:.*//'`
	epnoutputsocket[nepnoutputs]=`echo ${epndevice} | sed -e 's/.*://'`
	let nepnoutputs++
    elif [ "x${line:0:11}" == "xscheduling " ]; then
	echo $line
    elif [ "x$line" != "x" ]; then
	postponed_messages[npostponed_messages]=" $line"
	let npostponed_messages++
    else
	echo
    fi
done

for node in `for n in ${flpinputnode[@]}; do echo $n; done | sort | uniq`; do
    flpnodelist=(${flpnodelist[@]} $node)
    nodelist=(${nodelist[@]} $node)
done
for node in `for n in ${epnoutputnode[@]}; do echo $n; done | sort | uniq`; do
    epnnodelist=(${epnnodelist[@]} $node)
    nodelist=(${nodelist[@]} $node)
done

# read from subsidiary script in the current directory if nodelist not
# yet filled
nodelistfile=nodelist.sh
if [ "x$nodelist" == "x" ] && [ -e $nodelistfile ]; then
    . $nodelistfile
fi

if [ "x$nodelist" == "x" ]; then
cat <<EOF
error: can not find node definition

Please add a script file 'nodelist.sh' in the current directory
defining the nodes to be used in this topology.

############## example ##############################
nodelist=
nodelist=(\${nodelist[@]} localhost)
#nodelist=(\${nodelist[@]} someothernode)

EOF
exit -1
fi
nnodes=${#nodelist[@]}
nflpnodes=${#flpnodelist[@]}
nepnnodes=${#epnnodelist[@]}

echo "using $nflpnodes FLP and $nepnnodes EPN node(s) for running processing topology"
echo "FLP ${flpnodelist[@]} - EPN ${epnnodelist[@]}"

# init the variables for the session commands
sessionnode=
sessiontitle=
sessioncmd=
nsessions=0


###################################################################
# check if the output of the parent is binding or not
# and set the attributes according to that
# --output --> --input
# bind <--> connect
# push <--> pull
translate_io_attributes() {
    __inputattributes=$1
    __translated=`echo $1 | sed -e 'h; /--output/{s|--output|--input|g; p}; g; /--input/{s|--input|--output|g; p}; d'`
    __node=${2:-localhost}
    if [ x`echo ${__inputattributes} | sed -e 's|.*method=\(.*\)|\1|' -e 's|,.*$||'` == "xbind" ]; then
	__translated=`echo ${__translated} | sed -e "s|://\*:|://${__node}:|g" -e 's|method=bind|method=connect|g'`
    else
	__translated=`echo ${__translated} | sed -e 's|://.*:|://*:|g' -e 's|method=connect|method=bind|g'`
    fi
    __translated=`echo ${__translated} | sed -e 'h; /type=push/{s/type=push/type=pull/g; p}; g; /type=pull/{s/type=pull/type=push/g; p}; d'`
    echo $__translated
}

translate_io_alternateformat() {
    __inputattributes=$1
    io=`echo $__inputattributes | sed -e 's| .*$||`
    type=`echo $__inputattributes | sed -e 's| .*$||`
    io=`echo $__inputattributes | sed -e 's| .*$||`
}

###################################################################
# create an flp node group
epn1_input=
n_epn1_inputs=0
create_flpgroup() {
    node=$1
    basesocket=$2
    firstslice_on_node=$3
    nofslices=$4
    socketcount=0
    cf_output=
    if [ "x$CFoptPublishIndividualPartitions" == "xyes" ]; then
      nofIndividualPartitions=6
    else
      nofIndividualPartitions=1
    fi
    for ((c=0; c<nofslices; c++)); do
	for ((partition=0; partition<nofIndividualPartitions; partition++)); do
	slice=$((firstslice_on_node + c))
	socket=$((basesocket + socketcount))
        let socketcount++
	if [ "$nofIndividualPartitions" -le 1 ]; then
	spec=`printf %02d $slice`
	specprefix="slice_"
        else
        spec=0x`printf %02x%02x%02x%02x $slice $slice $partition $partition`
        specprefix=
        fi
        deviceid="ClusterPublisher_$spec"
	output="--output type=push,size=5000,method=bind,address=tcp://*:$socket"
	publisher_conf_file=data/emulated-tpc-clusters_$specprefix$spec.txt
	local_conf_file=/tmp/tpc-cluster-publisher.conf
	scp $publisher_conf_file $node:$local_conf_file
	command="aliceHLTWrapper $deviceid 1 --poll-period $pollingtimeout $output --library libAliHLTUtil.so --component FilePublisher --run $runno --parameter '-datafilelist $local_conf_file $CFoptOpenFilesAtStart'"

	sessionnode[nsessions]=$node
	sessiontitle[nsessions]=$deviceid
	sessioncmd[nsessions]=$command
	let nsessions++

	cf_output="$cf_output ${output}"
        done
    done

    if [ "x$bypass_relays" != "xyes" ]; then
        # add a relay combining CF output into one (multi)message
        deviceid="Relay_$node"
        input=`translate_io_attributes "$cf_output"`
        # output configuration is either taken from the flp/epn network
        # or according to the base socket
        output=
        if [ "$nflpinputs" -eq 0 ]; then
        output="--output type=push,size=5000,method=bind,address=tcp://*:$((basesocket + c))"
        else
            for ((iflpinput=0; iflpinput<nflpinputs; iflpinput++)); do
                if [ "x${flpinputnode[$iflpinput]}" == "x$node" ]; then
                    output="--output type=push,size=5000,method=connect,address=tcp://${flpinputnode[$iflpinput]}:${flpinputsocket[$iflpinput]}"
                fi
            done
        fi

        let socketcount++
        command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output --library libAliHLTUtil.so --component BlockFilter --run $runno --parameter ''"

        sessionnode[nsessions]=$node
        sessiontitle[nsessions]="$deviceid"
        sessioncmd[nsessions]=$command
        let nsessions++

        epn1_input[n_epn1_inputs]=${output/\/\/\*:///$node:}
        let n_epn1_inputs++
    else
        # add each CF output directly to EPN input
	# TODO: this is a bug, but it does not harm
	# the string is broken up at blanks, so the option-parameter
	# relation is lost and there are too many elements in the array,
	# the array is expanded to a string, so does not matter at the moment
        for output in $cf_output; do
            epn1_input[n_epn1_inputs]=${output/\/\/\*:///$node:}
            let n_epn1_inputs++
        done
    fi
}

###################################################################
# create an epn1 node group
epn2_input=
ntrackers=0
globalmergerid=0
create_epn1group() {
    node=$1
    basesocket=$2
    # epn2 input commonly subscribes to output of top devices of all epn1 node groups
    epn2_input=$3
    socketcount=0

    # check if there is a FLP to EPN topology available
    epnoutputonnode=
    nepnoutputonnode=0
    for ((iepnoutput=0; iepnoutput<nepnoutputs; iepnoutput++)); do
	if [ "${epnoutputnode[$iepnoutput]}" == "$node" ]; then
	    epnoutputonnode[nepnoutputonnode]="tcp://${epnoutputnode[$iepnoutput]}:${epnoutputsocket[$iepnoutput]}"
	    let nepnoutputonnode++
	fi
    done

    ntrackersonnode=$nepnoutputonnode
    [ "$nepnoutputonnode" -eq 0 ] && ntrackersonnode=1 # at least one tracker
    if [ "x$bypass_tracking" != "xyes" ]; then
        for ((trackerid=0; trackerid<$ntrackersonnode; trackerid++)); do

        deviceid=`printf %03d $ntrackers`
        deviceid="Tracker_$deviceid"
        let ntrackers++
        if [ "$nepnoutputonnode" -eq 0 ]; then
            output=`echo "${epn1_input[@]}"`
            input=`translate_io_attributes "$output"`
        else
            input="--input type=pull,size=1000,method=connect,address=${epnoutputonnode[$trackerid]}"
        fi

        output="--output type=push,size=1000,method=connect,address=tcp://localhost:$((basesocket + socketcount))"
        command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output --library libAliHLTTPC.so --component TPCCATracker --run $runno --parameter '-GlobalTracking -allowGPU -GPUHelperThreads 4 -loglevel=0x7c'"

        sessionnode[nsessions]=$node
        sessiontitle[nsessions]="$deviceid"
        sessioncmd[nsessions]=$command
        let nsessions++
        done
        socketcount=$((trackerid + 1))

        deviceid=GlobalMerger_`printf %02d $globalmergerid`
        input=`translate_io_attributes "$output"`
        output=`translate_io_attributes "$epn2_input"`
        let socketcount++
        command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output --library libAliHLTTPC.so --component TPCCAGlobalMerger --run $runno --parameter '-loglevel=0x7c'"

        sessionnode[nsessions]=$node
        sessiontitle[nsessions]="$deviceid"
        sessioncmd[nsessions]=$command
        let nsessions++
        let globalmergerid++
    fi

    # deviceid=FileWriter
    # input=`translate_io_attributes "$output"`
    # output=
    # let socketcount++
    # command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output --library libAliHLTUtil.so --component FileWriter --run $runno --parameter '-directory tracker-output -idfmt=%04d -specfmt=_%08x -blocknofmt= -loglevel=0x7c -write-all-blocks -publisher-conf tracker-output/datablocks.txt'"

    # sessionnode[nsessions]=$node
    # sessiontitle[nsessions]="$deviceid"
    # sessioncmd[nsessions]=$command
    # let nsessions++
}

###################################################################
# create an epn2 node group
# collects different processing branches from epn1
create_epn2group() {
    node=$1
    basesocket=$2
    socketcount=0

    deviceid=FileWriter
    epn2_input="--input type=pull,size=1000,method=bind,address=tcp://$node:$((basesocket + socketcount))"
    input=$epn2_input
    output=
    let socketcount++
    command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output --library libAliHLTUtil.so --component FileWriter --run $runno --parameter '-directory tracker-output -subdir -idfmt=%04d -specfmt=_%08x -blocknofmt= -loglevel=0x7c -write-all-blocks -publisher-conf tracker-output/datablocks.txt'"

    sessionnode[nsessions]=$node
    sessiontitle[nsessions]="$deviceid"
    sessioncmd[nsessions]=$command
    let nsessions++
}

########### main script ###########################################
#
# now build the commands on the nodes
# flp nodegroups
sliceno=$firstslice
inode=0
while [ "$sliceno" -le "$lastslice" ]; do
    if [ "$inode" -ge "$nflpnodes" ]; then
        echo "error: too few nodes to create all flp node groups"
        sliceno=$((lastslice + 1))
        exit -1
    fi
    create_flpgroup ${flpnodelist[inode]} $baseport_on_flpgroup $sliceno $slices_per_node
    sliceno=$((sliceno + slices_per_node))

    let inode++
done

# epn2 nodegroup
# has one input which all epn1 outputs will connect to
create_epn2group ${epn2node} $baseport_on_epn2group 

# epn1 nodegroup
inode=0
while [ "$inode" -lt "$nepnnodes" ]; do
    create_epn1group ${epnnodelist[inode]} $baseport_on_epn1group "$epn2_input"
    let inode++
done

# start the screen sessions and devices
sessionmap=
for ((isession=$nsessions++-1; isession>=0; isession--)); do
    sessionmap[isession]=1
done
havesessions=1

applications=
while [ "$havesessions" -gt 0 ]; do
havesessions=0
lastnode=
for ((isession=$nsessions++-1; isession>=0; isession--)); do
    if [ "${sessionmap[isession]}" -eq 1 ]; then
    if [ "x$printcmdtoscreen" == "x" ]; then
        echo "scheduling ${sessiontitle[$isession]} on ${sessionnode[$isession]}: ${sessioncmd[$isession]}"
    fi
    #logcmd=" 2>&1 | tee ${sessiontitle[$isession]}.log"
    applications+=" "`echo ${sessioncmd[$isession]} | sed -e 's| .*$||'`
    fi

    if [ "${sessionmap[isession]}" -gt 0 ]; then
        #echo $isession: ${sessionmap[isession]} $lastnode
        if [ "x$lastnode" == "x${sessionnode[$isession]}" ] && [ "${sessionmap[$isession]}" -lt 10 ]; then
            let sessionmap[isession]++
            havesessions=1
        else
            if [ "x$lastnode" == "x${sessionnode[$isession]}" ]; then
                # sleep between starts, some of the screens are not started if the frequency is too high
                usleep 500000
            fi
	    logcmd="2>&1 | tee ${sessiontitle[$isession]}.log";
            $printcmdtoscreen screen -d -m -S "${sessiontitle[$isession]} on ${sessionnode[$isession]}" ssh ${sessionnode[$isession]} "cd $rundir && source setup.sh && ${sessioncmd[$isession]} $logcmd" &
            sessionmap[isession]=0
            lastnode=${sessionnode[$isession]}
        fi
    fi
done
done

echo
for ((imsg=0; imsg<npostponed_messages; imsg++)); do
    [ "x${postponed_messages[$imsg]:0:12}" == "xfor node in" ] && continue
    echo ${postponed_messages[$imsg]}
done
if [ "x$printcmdtoscreen" == "x" ]; then
echo
usednodes=`for n in ${sessionnode[@]}; do echo $n; done | sort | uniq`
echo
echo "started processing topology in ${#sessionnode[@]} session(s) on `echo $usednodes | wc -w` node(s):"
usednodes=`echo $usednodes | sed ':a;N;$!ba;s/\n/ /g'`
echo $usednodes

applications=`for app in $applications; do echo $app; done | sort | uniq`
echo
echo "a simple method to stop the devices:"
for app in $applications; do
    echo "for node in $usednodes; do ssh \$node killall $app; done"
done
fi
