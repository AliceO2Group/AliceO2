command=$*
echo $command
# launch the command in the background
$command &
# get the PID
PID=$!

memlogfile="mem_evolution_${PID}.txt"
cpulogfile="cpu_evolution_${PID}.txt"
timelogfile="time_evolution_${PID}.txt"
idlogfile="pid_evolution_${PID}.txt"

echo "#command line: ${command}" > ${memlogfile}
echo "#memory evolution of process ${PID}; columns indicate active children" >> ${memlogfile}

echo "#command line: ${command}" > ${cpulogfile}
echo "#cpu evolution of process ${PID}; columns indicate active children" >> ${cpulogfile}

echo "#command line: ${command}" > ${idlogfile}
echo "#child process id evolution of leading o2-sim process with number ${PID}; columns indicate active children" >> ${idlogfile}

echo "# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #" >> ${memlogfile}
echo "# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #" >> ${cpulogfile}
echo "# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #" >> ${idlogfile}


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
  
  # get time stamp in mili-seconds
  echo $(($(date +%s%N)/1000000)) >> ${timelogfile}
  
  child_pid_list=
  childprocs ${PID}
  
  # sum up memory from all child processes
  mem=`for pid in $child_pid_list; do cat /proc/$pid/smaps | awk -v pid=$pid '/Pss/{mem+=$2} END {print mem/1024.}'; done | tr '\n' ' '`
  echo "${mem}" >> ${memlogfile}
  
  # sum up cpu from all child processes
  total_time=`for pid in $child_pid_list; do cat /proc/$pid/stat | awk -v pid=$pid '{total_time=$14+$15+$16+$17} END {print total_time/100.}'; done | tr '\n' ' '`
  echo "${total_time}" >> ${cpulogfile}
  
  # record the PID
  id=`for pid in $child_pid_list; do echo $pid; done | tr '\n' ' '`
  echo "${id}" >> ${idlogfile}
  
  # check if the job is still there
  ps -p $PID > /dev/null
  [ $? == 1 ] && echo "Job finished; Exiting " && break
  
  sleep 0.005
done
