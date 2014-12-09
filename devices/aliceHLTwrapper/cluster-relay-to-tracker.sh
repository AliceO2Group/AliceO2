#! /bin/bash

###################################################################
# global settings
runno=167808

# note: don't want to overdo the logic now, the lastslice should be firstslice plus multiples of slices_per_node minus 1
# lastslice=(firstslice + n*slices_per_node) - 1
firstslice=0
lastslice=3
slices_per_node=4
ibkey="-ib"
#dryrun="-n"
pollingtimeout=100

###################################################################
# fill the list of nodes
# be aware that the node strings must not contain blanks
nodelist=
nodelist=(${nodelist[@]} cn48)
nodelist=(${nodelist[@]} cn49)
nodelist=(${nodelist[@]} cn50)
nodelist=(${nodelist[@]} cn51)
nodelist=(${nodelist[@]} cn53)
nodelist=(${nodelist[@]} cn54)
nodelist=(${nodelist[@]} cn55)
nodelist=(${nodelist[@]} cn56)
nodelist=(${nodelist[@]} cn58)
nodelist=(${nodelist[@]} cn59)

# nodes with GPU
#nodelist=(${nodelist[@]} cn26)
#nodelist=(${nodelist[@]} cn27)
#nodelist=(${nodelist[@]} cn28)
#nodelist=(${nodelist[@]} cn29)
#nodelist=(${nodelist[@]} cn30)
#nodelist=(${nodelist[@]} cn31)
#nodelist=(${nodelist[@]} cn32)
#nodelist=(${nodelist[@]} cn33)
#nodelist=(${nodelist[@]} cn34)
#nodelist=(${nodelist[@]} cn35)

nnodes=${#nodelist[@]}

baseport_on_flpgroup=48000
baseport_on_epn1group=48100

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
    __translated=`echo $1 | sed -e 's|--output|--input|g'`
    __node=${2:-localhost}
    if [ x`echo ${__inputattributes} | sed -e 's|.*method=\(.*\)|\1|' -e 's|,.*$||'` == "xbind" ]; then
	__translated=`echo ${__translated} | sed -e "s|://\*:|://${__node}:|g" -e 's|method=bind|method=connect|g'`
    else
	__translated=`echo ${__translated} | sed -e 's|://.*:|://*:|g' -e 's|method=connect|method=bind|g'`
    fi
    __translated=`echo ${__translated} | sed -e 's/type=push/type=pull/g'`
    echo $__translated
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
    cf_output=
    for ((c=0; c<nofslices; c++)); do
	slice=$((firstslice_on_node + c))
	socket=$((basesocket + c))
	spec=`printf %02d $slice`
	output="--output type=push,size=5000,method=bind,address=tcp://*:$socket"
	command="aliceHLTWrapper ClusterPublisher_$spec 1 --poll-period $pollingtimeout $output --library libAliHLTUtil.so --component FilePublisher --run $runno --parameter '-datafilelist data/emulated-tpc-clusters_slice_$spec.txt'"
	deviceid="ClusterPublisher_$spec"

	sessionnode[nsessions]=$node
	sessiontitle[nsessions]=$deviceid
	sessioncmd[nsessions]=$command
	let nsessions++

	cf_output="$cf_output ${output}"
    done

    deviceid="Relay_$node"
    input=`translate_io_attributes "$cf_output"`
    output="--output type=push,size=5000,method=bind,address=tcp://*:$((basesocket + c))"
    let socketcount++
    command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output -m 1 --library libAliHLTUtil.so --component BlockFilter --run $runno --parameter ''"

    sessionnode[nsessions]=$node
    sessiontitle[nsessions]="$deviceid"
    sessioncmd[nsessions]=$command
    let nsessions++

    epn1_input[n_epn1_inputs]=${output/\/\/\*:///$node$ibkey:}
    let n_epn1_inputs++
}

###################################################################
# create an epn1 node group
create_epn1group() {
    node=$1
    basesocket=$2
    socketcount=0

    deviceid=Tracker
    output=`echo "${epn1_input[@]}"`
    input=`translate_io_attributes "$output"`
    output="--output type=push,size=1000,method=bind,address=tcp://*:$((basesocket + socketcount))"
    let socketcount++
    command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output --library libAliHLTTPC.so --component TPCCATracker --run $runno --parameter '-GlobalTracking -loglevel=0x79'"

    sessionnode[nsessions]=$node
    sessiontitle[nsessions]="$deviceid"
    sessioncmd[nsessions]=$command
    let nsessions++

    deviceid=GlobalMerger
    input=`translate_io_attributes "$output"`
    output="--output type=push,size=1000,method=bind,address=tcp://*:$((basesocket + socketcount))"
    let socketcount++
    command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output --library libAliHLTTPC.so --component TPCCAGlobalMerger --run $runno --parameter '-loglevel=0x7c'"

    sessionnode[nsessions]=$node
    sessiontitle[nsessions]="$deviceid"
    sessioncmd[nsessions]=$command
    let nsessions++

    deviceid=FileWriter
    input=`translate_io_attributes "$output"`
    output=
    let socketcount++
    command="aliceHLTWrapper $deviceid 1 ${dryrun} --poll-period $pollingtimeout $input $output --library libAliHLTUtil.so --component FileWriter --run $runno --parameter '-directory tracker-output -idfmt=%04d -specfmt=_%08x -blocknofmt= -loglevel=0x7c -write-all-blocks -publisher-conf tracker-output/datablocks.txt'"

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
    create_flpgroup ${nodelist[inode]} $baseport_on_flpgroup $sliceno $slices_per_node
    sliceno=$((sliceno + slices_per_node))

    let inode++
    if [ "$inode" -ge "$nnodes" ]; then
	echo "error: too few nodes to create all flp node groups"
	sliceno=$lastslice
    fi
done

# epn1 nodegroup
if [ "$inode" -lt "$nnodes" ]; then
    # epn1 group on the last node
    inode=$((nnodes - 1))
    create_epn1group ${nodelist[inode]} $baseport_on_epn1group 
else
    echo "error: too few nodes to create the epn1 node group"
fi

# start the screen sessions and devices
for ((isession=$nsessions++-1; isession>=0; isession--)); do
    echo "starting ${sessiontitle[$isession]} on ${sessionnode[$isession]}: ${sessioncmd[$isession]}"
    #$logcmd=" 2>&1 | tee ${sessiontitle[$isession]}.log"
    screen -d -m -S "${sessiontitle[$isession]} on ${sessionnode[$isession]}" ssh ${sessionnode[$isession]} "(cd workdir/alfa-rundir && source setup.sh && ${sessioncmd[$isession]}) $logcmd" &
done

