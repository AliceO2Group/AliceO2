# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
#
# author: Sandro Wenzel

# This file contains a couple of utility functions for reuse
# in shell job scripts (such as on the GRID).
# In order to use these functions in scripts, this file needs to be
# simply sourced into the target script.


o2_cleanup_shm_files() {
  # check if we have lsof (otherwise we do nothing)
  which lsof &> /dev/null
  if [ "$?" = "0" ]; then
    # find shared memory files **CURRENTLY IN USE** by FairMQ
    USEDFILES=`lsof -u $(whoami) | grep -e \"/dev/shm/.*fmq\" | sed 's/.*\/dev/\/dev/g' | sort | uniq | tr '\n' ' '`

    echo "${USEDFILES}"
    if [ ! "${USEDFILES}" ]; then
      # in this case we can remove everything
      COMMAND="find /dev/shm/ -user $(whoami) -name \"*fmq_*\" -delete 2> /dev/null"
    else
      # build exclusion list
      for f in ${USEDFILES}; do
        LOGICALOP=""
        [ "${EXCLPATTERN}" ] && LOGICALOP="-o"
        EXCLPATTERN="${EXCLPATTERN} ${LOGICALOP} -wholename ${f}"
      done
      COMMAND="find /dev/shm/ -user $(whoami) -type f -not \( ${EXCLPATTERN} \) -delete 2> /dev/null"
    fi
    eval "${COMMAND}"
  else
    echo "Can't do shared mem cleanup: lsof not found"
  fi
}

# Function to find out all the (recursive) child processes starting from a parent PID.
# The output includes includes the parent
# output is saved in child_pid_list
childprocs() {
  local parent=$1
  if [ "$parent" ] ; then
    child_pid_list="$child_pid_list $parent"
    for childpid in $(pgrep -P ${parent}); do
      childprocs $childpid "nottoplevel"
    done;
  fi
  # return via a string list (only if toplevel)
  if [ ! "$2" ]; then
    echo "${child_pid_list}"
  fi
}

taskwrapper_cleanup_handler() {
  PID=$1
  SIGNAL=$2
  echo "CLEANUP HANDLER FOR PROCESS ${PID} AND SIGNAL ${SIGNAL}"
  PROCS=$(childprocs ${PID})
  # make sure to bring down everything, including all kids
  for p in ${PROCS}; do
    echo "killing ${p}"
    kill -s ${SIGNAL} ${p} 2> /dev/null
  done
  o2_cleanup_shm_files
  # I prefer to exit the current job completely
  exit 1
}

# Function wrapping some process and asyncronously supervises and controls it.
# Main features provided at the moment are:
# - optional recording of walltime and memory consumption (time evolution)
# - optional recording of CPU utilization
# - Some job control and error detection (in particular for DPL workflows). 
#   If exceptions are found, all participating processes will be sent a termination signal.
#   The rational behind this function is to be able to determine failing 
#   conditions early and prevent longtime hanging executables 
#   (until DPL offers signal handling and automatic shutdown)
# - possibility to provide user hooks for "start" and "failure"
# - possibility to skip (jump over) job alltogether
taskwrapper() {
  local logfile=$1
  shift 1
  local command="$*"

  # launch the actual command in the background
  echo "Launching task: ${command} &> $logfile &"
  # the command might be a complex block: For the timing measurement below
  # it is better to execute this as a script
  SCRIPTNAME="${logfile}_tmp.sh"
  echo "${command}" > ${SCRIPTNAME}
  chmod +x ${SCRIPTNAME}

  # this gives some possibility to customize the wrapper
  # and do some special task at the start. The hook takes 2 arguments: 
  # The original command and the logfile
  if [ "${JOBUTILS_JOB_STARTHOOK}" ]; then
    hook="${JOBUTILS_JOB_STARTHOOK} '$command' $logfile"
    eval "${hook}"
  fi

  # We offer the possibility to jump this stage/task when a "done" file is present.
  # (this is mainly interesting for debugging in order to avoid going through all pipeline stages again)
  # The feature should be used with care! To make this nice, a proper dependency chain and a checksum mechanism
  # needs to be put into place.
  if [ "${JOBUTILS_SKIPDONE}" ]; then
    if [ -f "${logfile}_done" ]; then
       echo "Skipping task since file ${logfile}_done found";
       [ ! "${JOBUTILS_KEEPJOBSCRIPT}" ] && rm ${SCRIPTNAME} 2> /dev/null
       return 0
    fi
  fi
  [ -f "${logfile}_done" ] && rm "${logfile}"_done

  # the time command is non-standard on MacOS
  if [ "$(uname)" == "Darwin" ]; then
    GTIME=$(which gtime)
    TIMECOMMAND=${GTIME:+"${GTIME} --output=${logfile}_time"}
  else
    TIMECOMMAND="/usr/bin/time --output=${logfile}_time"
  fi

  # with or without memory monitoring ?
  finalcommand="TIME=\"#walltime %e\" ${TIMECOMMAND} ./${SCRIPTNAME}"
  if [[ "$(uname)" != "Darwin" && "${JOBUTILS_MONITORMEM}" ]]; then
    finalcommand="TIME=\"#walltime %e\" ${O2_ROOT}/share/scripts/monitor-mem.sh ${TIMECOMMAND} './${SCRIPTNAME}'"
  fi
  echo "Running: ${finalcommand}" > ${logfile}
  eval ${finalcommand} >> ${logfile} 2>&1 &

  # THE NEXT PART IS THE SUPERVISION PART
  # get the PID
  PID=$!
  # register signal handlers
  trap "taskwrapper_cleanup_handler ${PID} SIGINT" SIGINT
  trap "taskwrapper_cleanup_handler ${PID} SIGTERM" SIGTERM

  cpucounter=1

  while [ 1 ]; do
    # We don't like to see critical problems in the log file.

    # We need to grep on multitude of things:
    # - all sorts of exceptions (may need to fine-tune)  
    # - segmentation violation
    # - there was a crash
    # - bus error (often occuring with shared mem)
    pattern="-e \"xception\"                        \
             -e \"segmentation violation\"          \
             -e \"error while setting up workflow\" \
             -e \"bus error\"                       \
             -e \"Assertion.*failed\"               \
             -e \"There was a crash.\""
      
    grepcommand="grep -H ${pattern} $logfile >> encountered_exceptions_list 2>/dev/null"
    eval ${grepcommand}
    
    grepcommand="grep -h --count ${pattern} $logfile 2>/dev/null"
    # using eval here since otherwise the pattern is translated to a
    # a weirdly quoted stringlist
    RC=$(eval ${grepcommand})
    
    # if we see an exception we will bring down the DPL workflow
    # after having given it some chance to shut-down itself
    # basically --> send kill to all children
    if [ "$RC" != "" -a "$RC" != "0" ]; then
      echo "Detected critical problem in logfile $logfile"

      # this gives some possibility to customize the wrapper
      # and do some special task at the start. The hook takes 2 arguments: 
      # The original command and the logfile
      if [ "${JOBUTILS_JOB_FAILUREHOOK}" ]; then
        hook="${JOBUTILS_JOB_FAILUREHOOK} '$command' $logfile"
        eval "${hook}"
      fi

      sleep 2

      # query processes still alive and terminate them
      for p in $(childprocs ${PID}); do
        echo "killing child $p"
        kill -9 $p 2> /dev/null
      done
      sleep 2

      # remove leftover shm files
      o2_cleanup_shm_files

      RC_ACUM=$((RC_ACUM+1))
      [ ! "${JOBUTILS_KEEPJOBSCRIPT}" ] && rm ${SCRIPTNAME} 2> /dev/null
      return 1
    fi

    # check if command returned which may bring us out of the loop
    ps -p $PID > /dev/null
    [ $? == 1 ] && break

    if [ "${JOBUTILS_MONITORCPU}" ]; then
      # get some CPU usage statistics per process --> actual usage can be calculated thereafter
      for p in $(childprocs ${PID}); do
        total=`awk 'BEGIN{s=0}/cpu /{for (i=1;i<=NF;i++) s+=$i;} END {print s}' /proc/stat`
        utime=`awk '//{print $14}' /proc/${p}/stat 2> /dev/null`
        stime=`awk '//{print $15}' /proc/${p}/stat 2> /dev/null`
        name=`awk '//{print $2}' /proc/${p}/stat 2> /dev/null`
        echo "${cpucounter} ${p} ${total} ${utime} ${stime} ${name}" >> ${logfile}_cpuusage
      done
      let cpucounter=cpucounter+1
    fi

    # sleep for some time (can be customized for power user)
    sleep ${JOBUTILS_WRAPPER_SLEEP:-5}
  done

  # wait for PID and fetch return code
  # ?? should directly exit here?
  wait $PID
  # return code
  RC=$?
  RC_ACUM=$((RC_ACUM+RC))
  if [ "${RC}" -eq "0" ]; then
    # if return code 0 we mark this task as done
    echo "Command \"${command}\" successfully finished." > "${logfile}"_done
    echo "The presence of this file can be used to skip this command in future runs" >> "${logfile}"_done
    echo "of the pipeline by setting the JOBUTILS_SKIPDONE environment variable." >> "${logfile}"_done
  else
    echo "command ${command} had nonzero exit code ${RC}"
  fi
  [ ! "${JOBUTILS_KEEPJOBSCRIPT}" ] && rm ${SCRIPTNAME} 2> /dev/null

  # deregister signal handlers
  trap '' SIGINT
  trap '' SIGTERM

  o2_cleanup_shm_files #--> better to register a general trap at EXIT
  return ${RC}
}

getNumberOfPhysicalCPUCores() {
  if [ "$(uname)" == "Darwin" ]; then
    CORESPERSOCKET=`system_profiler SPHardwareDataType | grep "Total Number of Cores:" | awk '{print $5}'`
    SOCKETS=`system_profiler SPHardwareDataType | grep "Number of Processors:" | awk '{print $4}'`
  else
    # Do something under GNU/Linux platform
    CORESPERSOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $4}'`
    SOCKETS=`lscpu | grep "Socket(s)" | awk '{print $2}'`
  fi
  N=`bc <<< "${CORESPERSOCKET}*${SOCKETS}"`
  echo "${N}"
}

