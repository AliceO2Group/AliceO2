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
# simply sourced into the target script. The script needs bash versions > 4

# TODOs:
# -harmonize use of bc/awk for calculations
# -harmonize coding style for variables

o2_cleanup_shm_files() {
  # check if we have lsof (otherwise we do nothing)
  which lsof &> /dev/null
  if [ "$?" = "0" ]; then
    # find shared memory files **CURRENTLY IN USE** by FairMQ
    USEDFILES=`lsof -u $(whoami) 2> /dev/null | grep -e \"/dev/shm/.*fmq\" | sed 's/.*\/dev/\/dev/g' | sort | uniq | tr '\n' ' '`

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
}

taskwrapper_cleanup_handler() {
  PID=$1
  SIGNAL=$2
  echo "CLEANUP HANDLER FOR PROCESS ${PID} AND SIGNAL ${SIGNAL}"
  taskwrapper_cleanup ${PID} ${SIGNAL}
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
# - possibility to define timeout
# - possibility to control/limit the CPU load
taskwrapper() {
  local logfile=$1
  shift 1
  local command="$*"

  STARTTIME=$SECONDS

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
  eval ${finalcommand} >> ${logfile} 2>&1 & #can't disown here since we want to retrieve exit status later on

  # THE NEXT PART IS THE SUPERVISION PART
  # get the PID
  PID=$!
  # register signal handlers
  trap "taskwrapper_cleanup_handler ${PID} SIGINT" SIGINT
  trap "taskwrapper_cleanup_handler ${PID} SIGTERM" SIGTERM

  cpucounter=1
  inactivitycounter=0   # used to detect periods of inactivity
  NLOGICALCPUS=$(getNumberOfLogicalCPUCores)

  reduction_factor=1
  control_iteration=1
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
      
    grepcommand="grep -H ${pattern} $logfile ${JOBUTILS_JOB_SUPERVISEDFILES} >> encountered_exceptions_list 2>/dev/null"
    eval ${grepcommand}
    
    grepcommand="grep -h --count ${pattern} $logfile ${JOBUTILS_JOB_SUPERVISEDFILES} 2>/dev/null"
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

      taskwrapper_cleanup ${PID} SIGKILL

      RC_ACUM=$((RC_ACUM+1))
      [ ! "${JOBUTILS_KEEPJOBSCRIPT}" ] && rm ${SCRIPTNAME} 2> /dev/null
      [[ ! "${JOBUTILS_NOEXIT_ON_ERROR}" ]] && [[ ! $- == *i* ]] && exit 1
      return 1
    fi

    # check if command returned which may bring us out of the loop
    ps -p $PID > /dev/null
    [ $? == 1 ] && break

    if [ "${JOBUTILS_MONITORCPU}" ] || [ "${JOBUTILS_LIMITLOAD}" ]; then
      # NOTE: The following section is "a bit" compute intensive and currently not optimized
      # A careful evaluation of awk vs bc or other tools might be needed -- or a move to a more
      # system oriented language/tool

      for p in $limitPIDs; do
        wait ${p}
      done

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
      # do some calculations based on the data
      totalCPU=0             # actual CPU load measured
      totalCPU_unlimited=0   # extrapolated unlimited CPU load
      line=""
      for p in $childpids; do
        C=${current[$p]}
        P=${previous[$p]}
        CT=${total}
        PT=${previous_total}
        # echo "${p} : current ${C} previous ${P} ${CT} ${PT}"
        thisCPU[$p]=$(awk -v "c=${C}" -v "p=${P}" -v "ct=${CT}" -v "pt=${PT}" -v "ncpu=${NLOGICALCPUS}" 'BEGIN { print 100.*ncpu*(c-p)/(ct-pt); }')
        line="${line} $p:${thisCPU[$p]}"
        totalCPU=$(awk -v "t=${totalCPU}" -v "this=${thisCPU[$p]}" 'BEGIN { print (t + this); }')
        previousfactor=1
        [ ${waslimited[$p]} ] && previousfactor=${reduction_factor}
        totalCPU_unlimited=$(awk -v "t=${totalCPU_unlimited}" -v "this=${thisCPU[$p]}" -v f="${previousfactor}" 'BEGIN { print (t + this/f); }')
        # echo "CPU last time window ${p} : ${thisCPU[$p]}"
      done

      echo "${line}"
      echo "${cpucounter} totalCPU = ${totalCPU} -- without limitation ${totalCPU_unlimited}"
      # We can check if the total load is above a resource limit
      # And take corrective actions if we extend by 10%
      limitPIDs=""
      unset waslimited
      if [ ${JOBUTILS_LIMITLOAD} ]; then
        if (( $(echo "${totalCPU_unlimited} > 1.1*${JOBUTILS_LIMITLOAD}" | bc -l) )); then
          # we reduce each pid proportionally for the time until the next check and record the reduction factor in place
          oldreduction=${reduction_factor}
          reduction_factor=$(awk -v limit="${JOBUTILS_LIMITLOAD}" -v cur="${totalCPU_unlimited}" 'BEGIN{ print limit/cur;}')
          echo "APPLYING REDUCTION = ${reduction_factor}"

          for p in $childpids; do
            cpulim=$(awk -v a="${thisCPU[${p}]}" -v newr="${reduction_factor}" -v oldr="${oldreduction}" 'BEGIN { r=(a/oldr)*newr; print r; if(r > 0.05) {exit 0;} exit 1; }')
            if [ $? = "0" ]; then
              # we only apply to jobs above a certain threshold
              echo "Setting CPU lim for job ${p} / ${name[$p]} to ${cpulim}";

              timeout ${JOBUTILS_WRAPPER_SLEEP} ${O2_ROOT}/share/scripts/cpulimit -l ${cpulim} -p ${p} > /dev/null 2> /dev/null & disown
              proc=$!
              limitPIDs="${limitPIDs} ${proc}"
              waslimited[$p]=1
            fi
          done
        else
          # echo "RESETING REDUCTION = 1"
          reduction_factor=1.
        fi
      fi

      let cpucounter=cpucounter+1
      # our condition for inactive
      if (( $(echo "${totalCPU} < 5" | bc -l) )); then
        let inactivitycounter=inactivitycounter+JOBUTILS_WRAPPER_SLEEP
      else
        inactivitycounter=0
      fi
      if [ "${JOBUTILS_JOB_KILLINACTIVE}" ]; then
        $(awk -v I="${inactivitycounter}" -v T="${JOBUTILS_JOB_KILLINACTIVE}" 'BEGIN {if(I>T){exit 1;} exit 0;}')
        if [ "$?" = "1" ]; then
          echo "task inactivity limit reached .. killing all processes";
          taskwrapper_cleanup $PID SIGKILL
          # call a more specialized hook for this??
          if [ "${JOBUTILS_JOB_FAILUREHOOK}" ]; then
            hook="${JOBUTILS_JOB_FAILUREHOOK} '$command' $logfile"
            eval "${hook}"
          fi
          return 1
        fi
      fi
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
        return 1
      fi
    fi

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

  # wait for PID and fetch return code
  # ?? should directly exit here?
  wait $PID
  # return code
  RC=$?
  RC_ACUM=$((RC_ACUM+RC))
  if [ "${RC}" -eq "0" ]; then
    if [ ! "${JOBUTILS_JOB_SKIPCREATEDONE}" ]; then
      # if return code 0 we mark this task as done
      echo "Command \"${command}\" successfully finished." > "${logfile}"_done
      echo "The presence of this file can be used to skip this command in future runs" >> "${logfile}"_done
      echo "of the pipeline by setting the JOBUTILS_SKIPDONE environment variable." >> "${logfile}"_done
    fi
  else
    echo "command ${command} had nonzero exit code ${RC}"
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

getNumberOfLogicalCPUCores() {
  if [ "$(uname)" == "Darwin" ]; then
    echo $(sysctl -n hw.logicalcpu)
  else
    # Do something under GNU/Linux platform
    echo $(grep "processor" /proc/cpuinfo | wc -l)
  fi
}
