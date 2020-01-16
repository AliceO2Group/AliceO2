command=$*
echo $command
# launch the command in the background
$command &
# get the PID
PID=$!

memlogfile="mem_evolution_${PID}.log"
echo "#memory evolution of process ${PID}; columns indicate active children" > ${memlogfile}
echo "#command line: ${command}" >> ${memlogfile}

child_pid_list=

# finds out all the (recursive) child process starting from a parent
# output includes the parent
# output is saves
childprocs() {
  local parent=$1
  if [ "$parent" ] ; then
    child_pid_list="$child_pid_list $parent"
    for childpid in $(pgrep -P ${parent}); do
      childprocs $childpid
    done;
  fi
}

# while this PID exists we sample
while [ 1 ] ; do
  child_pid_list=
  childprocs ${PID}

  # sum up memory from all child processes
  mem=`for pid in $child_pid_list; do cat /proc/$pid/smaps | awk -v pid=$pid '/Pss/{mem+=$2} END {print mem/1024.}'; done | tr '\n' ' '`
  echo "${mem}" >> ${memlogfile}
 
  # check if the job is still there
  ps -p $PID > /dev/null
  [ $? == 1 ] && echo "Job finished; Exiting " && break

  sleep 0.005
done


# print summary
MAXMEM=`awk '/^[0-9]/{ for(i=1; i<=NF;i++) j+=$i; print j; j=0 }' ${memlogfile} | awk 'BEGIN {m = 0} //{if($1>m){m=$1}} END {print m}'`
AVGMEM=`awk '/^[0-9]/{ for(i=1; i<=NF;i++) j+=$i; print j; j=0 }' ${memlogfile} | awk 'BEGIN {a = 0;c = 0} //{c=c+1;a=a+$1} END {print a/c}'`

echo "PROCESS MAX MEM = ${MAXMEM}"
echo "PROCESS AVG MEM = ${AVGMEM}"
