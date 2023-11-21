# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
#
# author: Sandro Wenzel

# This file contains a couple of utility functions for reuse
# in shell job scripts (such as on the GRID).
# In order to use these functions in scripts, this file needs to be
# simply sourced into the target script. The script needs bash versions > 4

# TODOs:
# -harmonize use of bc/awk for calculations
# -harmonize coding style for variables

o2_cleanup_shm_files() {
  if [ "${JOBUTILS_INTERNAL_DPL_SESSION}" ]; then
    # echo "cleaning up session ${JOBUTILS_INTERNAL_DPL_SESSION}"
    fairmq-shmmonitor -s ${JOBUTILS_INTERNAL_DPL_SESSION} -c &> /dev/null
  fi
}

# Function to find out all the (recursive) child processes starting from a parent PID.
# The output includes the parent
childprocs() {
  local parent=$1
  if [ ! "$2" ]; then
    child_pid_list=""
  fi
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

taskwrapper_cleanup() {
  MOTHERPID=$1
  SIGNAL=${2:-SIGTERM}
  for p in $(childprocs ${MOTHERPID}); do
    echo "killing child $p"
    kill -s ${SIGNAL} $p 2> /dev/null
  done
  sleep 2
  # remove leftover shm files
  o2_cleanup_shm_files
  unset JOBUTILS_INTERNAL_DPL_SESSION
}

taskwrapper_cleanup_handler() {
  PID=$1
  SIGNAL=$2
  echo "CLEANUP HANDLER FOR PROCESS ${PID} AND SIGNAL ${SIGNAL}"
  taskwrapper_cleanup ${PID} ${SIGNAL}
  # I prefer to exit the current job completely
  return 1 2>/dev/null || exit 1
}

# Function monitoring (DPL) log output for signs of failure
monitorlog() {
    [[ ! "${JOBUTILS_PERFORM_MONITORING}" ]] && exit 0
    # We need to grep on multitude of things:
    # - all sorts of exceptions (may need to fine-tune)
    # - segmentation violation
    # - there was a crash
    # - bus error (often occuring with shared mem)
    pattern="-e \"\<[Ee]xception\"                         \
             -e \"segmentation violation\"                 \
             -e \"error while setting up workflow\"        \
             -e \"bus error\"                              \
             -e \"Assertion.*failed\"                      \
             -e \"Fatal in\"                               \
             -e \"libc++abi.*terminating\"                 \
             -e \"There was a crash.\"                     \
             -e \"arrow.*Check failed\"                    \
             -e \"terminate called after\"                 \
             -e \"terminate called without an active\"     \
             -e \"\]\[FATAL\]\"                            \
             -e \"TASK-EXIT-CODE\"                         \
             -e \"\*\*\* Error in\"" # <--- LIBC fatal error messages

    # arguments to function:
    logfile=$1

    # runs resource-friendly until trigger match found
    # only invokes tail + grep once
    tmpfile=$(mktemp /tmp/taskwrapper-tail.XXXXXX)
    command="grep ${pattern} -m 1 &> /dev/null"
    ( tail -f ${logfile} & echo $! >${tmpfile} ) | eval "${command}"
    kill -s SIGKILL $(<${tmpfile})
    rm ${tmpfile}

    tail -n 10 ${logfile} | grep "TASK-EXIT-CODE" &> /dev/null
    RC=$?  # this will be one whenever TASK-EXIT-CODE could not be found
    # echo "check for task exit code yielded ${RC}"

    if [ "$RC" = "1" ]; then
      echo "Detected critical problem in logfile $logfile"
      if [ "${JOBUTILS_PRINT_ON_ERROR}" ]; then
       grepcommand="grep -a -H -A 2 -B 2 ${pattern} $logfile ${JOBUTILS_JOB_SUPERVISEDFILES}"
       eval ${grepcommand}
      fi

      # this gives some possibility to customize the wrapper
      # and do some special task at the start. The hook takes 2 arguments:
      # The original command and the logfile
      if [ "${JOBUTILS_JOB_FAILUREHOOK}" ]; then
        hook="${JOBUTILS_JOB_FAILUREHOOK} '$command' $logfile"
        eval "${hook}"
      fi

      exit 1
    fi
    exit 0
} # end of function monitorlog

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
# - possibility to define timeout
# - possibility to control/limit the CPU load
taskwrapper() {
  unset JOBUTILS_INTERNAL_DPL_SESSION
  # nested helper to parse DPL session ID
  _parse_DPL_session ()
  {
    childpids=$(childprocs ${1})
    for p in ${childpids}; do
      command=$(ps -o command ${p} | grep -v "COMMAND" | grep "session")
      if [ "$?" = "0" ]; then
        # echo "parsing from ${command}"
        session=`echo ${command} | sed 's/.*--session//g' | awk '//{print $1}'`
        if [ "${session}" ]; then
          # echo "found ${session}"
          break
        fi
      fi
    done
    echo "${session:-""}"
  }

  local logfile=$1
  shift 1
  local command="$*"

  STARTTIME=$SECONDS

  # launch the actual command in the background
  echo "Launching task: ${command} &> $logfile &"
  # the command might be a complex block: For the timing measurement below
  # it is better to execute this as a script
  SCRIPTNAME="${logfile}_tmp.sh"
  echo "#!/usr/bin/env bash" > ${SCRIPTNAME}
  echo "export LIBC_FATAL_STDERR_=1" >> ${SCRIPTNAME}        # <--- needed ... otherwise the LIBC fatal messages appear on a different tty
  echo "${command};" >> ${SCRIPTNAME}
  echo 'RC=$?; echo "TASK-EXIT-CODE: ${RC}"; exit ${RC}' >> ${SCRIPTNAME}
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
  finalcommand="TIME=\"#walltime %e\\n#systime %S\\n#usertime %U\\n#maxmem %M\\n#CPU %P\" ${TIMECOMMAND} ./${SCRIPTNAME}"
  if [[ "$(uname)" != "Darwin" && "${JOBUTILS_MONITORMEM}" ]]; then
    finalcommand="TIME=\"#walltime %e\\n#systime %S\\n#usertime %U\\n#maxmem %M\\n#CPU %P\" ${O2_ROOT}/share/scripts/monitor-mem.sh ${TIMECOMMAND} './${SCRIPTNAME}'"
  fi
  echo "Running: ${finalcommand}" > ${logfile}

  # launch task to monitoring log (in background)
  monitorlog ${logfile} &
  MONITORLOGPID=$!

  eval ${finalcommand} >> ${logfile} 2>&1 & #cannot disown here since we want to retrieve exit status later on

  # THE NEXT PART IS THE SUPERVISION PART
  # get the PID
  PID=$!
  # register signal handlers
  trap "taskwrapper_cleanup_handler ${PID} SIGINT" SIGINT
  trap "taskwrapper_cleanup_handler ${PID} SIGTERM" SIGTERM

  cpucounter=1
  inactivitycounter=0   # used to detect periods of inactivity
  NLOGICALCPUS=$(getNumberOfLogicalCPUCores)

  control_iteration=1
  while [ "${CONTROLLOOP}" ]; do
    # check if command returned which may bring us out of the loop
    ps -p $PID > /dev/null
    [ $? == 1 ] && break

    if [ "${JOBUTILS_MONITORMEM}" ]; then
      if [ "${JOBUTILS_INTERNAL_DPL_SESSION}" ]; then
        MAX_FMQ_SHM=${MAX_FMQ_SHM:-0}
        text=$(fairmq-shmmonitor -v -s ${JOBUTILS_INTERNAL_DPL_SESSION})
        line=$(echo ${text} | tr '[' '\n[' | grep "^0" | tail -n1)
        CURRENT_FMQ_SHM=$(echo ${line} | sed 's/.*used://g')
        # echo "current shm ${CURRENT_FMQ_SHM}"
        MAX_FMQ_SHM=$(awk -v "t=${CURRENT_FMQ_SHM}" -v "s=${MAX_FMQ_SHM}" 'BEGIN { if(t>=s) { print t; } else { print s; } }')
      fi
    fi

    if [ "${JOBUTILS_MONITORCPU}" ]; then
      # NOTE: The following section is "a bit" compute intensive and currently not optimized
      # A careful evaluation of awk vs bc or other tools might be needed -- or a move to a more
      # system oriented language/tool

      # get some CPU usage statistics per process --> actual usage can be calculated thereafter
      total=`awk 'BEGIN{s=0}/cpu /{for (i=1;i<=NF;i++) s+=$i;} END {print s}' /proc/stat`
      previous_total=${current_total}
      current_total=${total}
      # quickly fetch the data
      childpids=$(childprocs ${PID})

      for p in $childpids; do
        while read -r name utime stime; do
          echo "${cpucounter} ${p} ${total} ${utime} ${stime} ${name}" >> ${logfile}_cpuusage
          previous[$p]=${current[$p]}
          current[$p]=${utime}
          name[$p]=${name}
        done <<<$(awk '//{print $2" "$14" "$15}' /proc/${p}/stat 2>/dev/null)
      done
    fi

    # a good moment to check for jobs timeout (or other resources)
    if [ "$JOBUTILS_JOB_TIMEOUT" ]; then
      $(awk -v S="${SECONDS}" -v T="${JOBUTILS_JOB_TIMEOUT}" -v START="${STARTTIME}" 'BEGIN {if((S-START)>T){exit 1;} exit 0;}')
      if [ "$?" = "1" ]; then
        echo "task timeout reached .. killing all processes";
        taskwrapper_cleanup $PID SIGKILL
        # call a more specialized hook for this??
        if [ "${JOBUTILS_JOB_FAILUREHOOK}" ]; then
          hook="${JOBUTILS_JOB_FAILUREHOOK} '$command' $logfile"
          eval "${hook}"
        fi
        [ "${JOBUTILS_PRINT_ON_ERROR}" ] && echo ----- Last log: ----- && pwd && cat ${logfile} && echo ----- End of log -----
        [[ ! "${JOBUTILS_NOEXIT_ON_ERROR}" ]] && [[ ! $- == *i* ]] && exit 1
        return 1
      fi
    fi

    # Try to find out DPL session ID
    # if [ -z "${JOBUTILS_INTERNAL_DPL_SESSION}" ]; then
        JOBUTILS_INTERNAL_DPL_SESSION=$(_parse_DPL_session ${PID})
    #   echo "got session ${JOBUTILS_INTERNAL_DPL_SESSION}"
    # fi

    # sleep for some time (can be customized for power user)
    sleep ${JOBUTILS_WRAPPER_SLEEP:-1}

    # power feature: we allow to call a user hook at each i-th control
    # iteration
    if [ "${JOBUTILS_JOB_PERIODICCONTROLHOOK}" ]; then
      if [ "${control_iteration}" = "${JOBUTILS_JOB_CONTROLITERS:-10}" ]; then
        hook="${JOBUTILS_JOB_PERIODICCONTROLHOOK} '$command' $logfile"
        eval "${hook}"
        control_iteration=0
      fi
    fi

    let control_iteration=control_iteration+1
  done

  wait ${MONITORLOGPID}
  MRC=$?
  if [ "${MRC}" = "1" ]; then
     echo "Abnormal problem detected; Bringing down workflow after 2 seconds"
     sleep 2
     [ ! "${JOBUTILS_DEBUGMODE}" ] && taskwrapper_cleanup ${PID} SIGKILL
  fi

  # wait for man task PID and fetch return code
  # ?? should directly exit here?
  wait $PID || QUERY_RC_FROM_LOG="ON"

  # query return code from log (seems to be safer as sometimes the wait issues "PID" not a child of this shell)
  RC=$(awk '/TASK-EXIT-CODE:/{print $2}' ${logfile})
  if [ ! "${RC}" ]; then
    RC=1
  fi
  if [ "${RC}" -eq "0" ]; then
    if [ ! "${JOBUTILS_JOB_SKIPCREATEDONE}" ]; then
      # if return code 0 we mark this task as done
      echo "Command \"${command}\" successfully finished." > "${logfile}"_done
      echo "The presence of this file can be used to skip this command in future runs" >> "${logfile}"_done
      echo "of the pipeline by setting the JOBUTILS_SKIPDONE environment variable." >> "${logfile}"_done
    fi
  else
    echo "command ${command} had nonzero exit code ${RC}"
    [ "${JOBUTILS_PRINT_ON_ERROR}" ] && echo ----- Last log: ----- && pwd && cat ${logfile} && echo ----- End of log -----
  fi
  [ ! "${JOBUTILS_KEEPJOBSCRIPT}" ] && rm ${SCRIPTNAME} 2> /dev/null

  # deregister signal handlers
  trap '' SIGINT
  trap '' SIGTERM

  o2_cleanup_shm_files #--> better to register a general trap at EXIT

  # this gives some possibility to customize the wrapper
  # and do some special task at the ordinary exit. The hook takes 3 arguments:
  # - The original command
  # - the logfile
  # - the return code from the execution
  if [ "${JOBUTILS_JOB_ENDHOOK}" ]; then
    hook="${JOBUTILS_JOB_ENDHOOK} '$command' $logfile ${RC}"
    eval "${hook}"
  fi

  if [ ! "${RC}" -eq "0" ]; then
    if [ ! "${JOBUTILS_NOEXIT_ON_ERROR}" ]; then
      # in case of incorrect termination, we usually like to stop the whole outer script (== we are in non-interactive mode)
      [[ ! $- == *i* ]] && exit ${RC}
    fi
  fi
  if [ "${JOBUTILS_MONITORMEM}" ]; then
     # convert bytes in MB
     MAX_FMQ_SHM=${MAX_FMQ_SHM:-0}
     MAX_FMQ_SHM=$(awk -v "s=${MAX_FMQ_SHM}" 'BEGIN { print s/(1024.*1024) }')
     echo "PROCESS MAX FMQ_SHM = ${MAX_FMQ_SHM}" >> ${logfile}
  fi
  unset JOBUTILS_INTERNAL_DPL_SESSION
  return ${RC}
}

getNumberOfPhysicalCPUCores() {
  if [ "$(uname)" == "Darwin" ]; then
    CORESPERSOCKET=`system_profiler SPHardwareDataType | grep "Total Number of Cores:" | awk '{print $5}'`
    if [ "$(uname -m)" == "arm64" ]; then
  SOCKETS=1
    else
  SOCKETS=`system_profiler SPHardwareDataType | grep "Number of Processors:" | awk '{print $4}'`
    fi
  else
    # Do something under GNU/Linux platform
    CORESPERSOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $4}'`
    SOCKETS=`lscpu | grep "Socket(s)" | awk '{print $2}'`
  fi
  N=`bc <<< "${CORESPERSOCKET}*${SOCKETS}"`
  echo "${N}"
}

getNumberOfLogicalCPUCores() {
  if [ "$(uname)" == "Darwin" ]; then
    echo $(sysctl -n hw.logicalcpu)
  else
    # Do something under GNU/Linux platform
    echo $(grep "processor" /proc/cpuinfo | wc -l)
  fi
}
