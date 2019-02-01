let counter=0
workflowexec=$1
while [ 1 ] ; do
    mem=`for pid in $(pgrep $workflowexec); do cat /proc/$pid/smaps | awk -v pid=$pid '/Pss/{mem+=$2} END {print mem/1024.}'; done | tr '\n' ' '`
    ids=`for pid in $(pgrep $workflowexec); do cat /proc/$pid/cmdline | awk '/--id/{print } !/--id/{print "MASTER "}' | sed 's/.*--id//g' | sed 's/--control.*//'; done | tr '\n' ' '`  
    pidline=`for pid in $(pgrep $workflowexec); do echo $pid; done | tr '\n' ' '`
    echo "# ${counter} ${pidline}"
    echo "# ${counter} ${ids}"
    echo "${counter} ${mem}"
    sleep 0.02
    let counter=counter+1
done

