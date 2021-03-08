#!/bin/bash
#
# A workflow performing a full system test:
# - simulation of digits
# - creation of raw data
# - reconstruction of raw data
#
# Note that this might require a production server to run.
#
# This script can use additional binary objects which can be optionally provided:
# - matbud.root + ITSdictionary.bin
#
# authors: D. Rohr / S. Wenzel

# include jobutils, which notably brings
# --> the taskwrapper as a simple control and monitoring tool
#     (look inside the jobutils.sh file for documentation)
# --> utilities to query CPU count
. ${O2_ROOT}/share/scripts/jobutils.sh

export NEvents=${NEvents:-10} #550 for full TF (the number of PbPb events)
export NEventsQED=${NEventsQED:-1000} #35000 for full TF
export NCPUS=$(getNumberOfPhysicalCPUCores)
echo "Found ${NCPUS} physical CPU cores"
export NJOBS=${NJOBS:-"${NCPUS}"}
export SHMSIZE=${SHMSIZE:-8000000000} # Size of shared memory for messages (use 128 GB for 550 event full TF)
export TPCTRACKERSCRATCHMEMORY=${SHMSIZE:-4000000000} # Size of memory allocated by TPC tracker. (Use 24 GB for 550 event full TF)
export ENABLE_GPU_TEST=${ENABLE_GPU_TEST:-0} # Run the full system test also on the GPU
export NTIMEFRAMES=${NTIMEFRAMES:-1} # Number of time frames to process
export TFDELAY=${TFDELAY:-100} # Delay in seconds between publishing time frames
export NOMCLABELS="--disable-mc"
export O2SIMSEED=${O2SIMSEED:--1}

# allow skipping
JOBUTILS_SKIPDONE=ON
# enable memory monitoring (independent on whether DPL or not)
JOBUTILS_MONITORMEM=ON
# CPU monitoring JOBUTILS_MONITORCPU=ON

# prepare some metrics file for the monitoring system
METRICFILE=metrics.dat
CONFIG="full_system_test_N${NEvents}"
HOST=`hostname`

# include header information such as tested alidist tag and O2 tag
TAG="conf=${CONFIG},host=${HOST}${ALIDISTCOMMIT:+,alidist=$ALIDISTCOMMIT}${O2COMMIT:+,o2=$O2COMMIT}"
echo "versions,${TAG} alidist=\"${ALIDISTCOMMIT}\",O2=\"${O2COMMIT}\" " > ${METRICFILE}

ulimit -n 4096 # Make sure we can open sufficiently many files

export GLOBALDPLOPT="-b" #  --monitoring-backend no-op:// is currently removed due to https://alice.its.cern.ch/jira/browse/O2-1887

mkdir -p raw

# create the full workflow as json
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
${DIR}/full-system-test/create_full_system_pipeline.py

# create a visualization of the pipeline-workflow (in workflow.gv.pdf) and print the list of tasks
${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json --visualize-workflow --list-tasks

# run the workflow (not constraining in parallelism) until RAW creation is finished
${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json --target-labels RAW

# We run the workflow in both CPU-only and With-GPU mode
STAGES="NOGPU"
if [ $ENABLE_GPU_TEST != "0" ]; then
  STAGES+=" WITHGPU"
fi
STAGES+=" ASYNC"
for STAGE in $STAGES; do
  logfile=reco_${STAGE}.log
  export GLOBALDPLOPT

  # run each of the reco but in strictly serial mode
  ${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt reco_${STAGE} -jmax 1

  # --- record interesting metrics to monitor ----
  # boolean flag indicating if workflow completed successfully at all
  RC=$?
  SUCCESS=0
  [ -f "${logfile}_done" ] && [ "$RC" = 0 ] && SUCCESS=1
  echo "success_${STAGE},${TAG} value=${SUCCESS}" >> ${METRICFILE}

  if [ "${SUCCESS}" = "1" ]; then
    # runtime
    walltime=`grep "#walltime" ${logfile}_time | awk '//{print $2}'`
    echo "walltime_${STAGE},${TAG} value=${walltime}" >> ${METRICFILE}

    # GPU reconstruction (also in CPU version) processing time
    gpurecotime=`grep "tpc-tracker" reco_NOGPU.log | grep -e "Total Wall Time:" | awk '//{printf "%f", $6/1000000}'`
    echo "gpurecotime_${STAGE},${TAG} value=${gpurecotime}" >> ${METRICFILE}

    # memory
    maxmem=`awk '/PROCESS MAX MEM/{print $5}' ${logfile}`  # in MB
    avgmem=`awk '/PROCESS AVG MEM/{print $5}' ${logfile}`  # in MB
    echo "maxmem_${STAGE},${TAG} value=${maxmem}" >> ${METRICFILE}
    echo "avgmem_${STAGE},${TAG} value=${avgmem}" >> ${METRICFILE}

    # some physics quantities
    tpctracks=`grep "tpc-tracker" ${logfile} | grep -e "found.*track" | awk '//{print $4}'`
    echo "tpctracks_${STAGE},${TAG} value=${tpctracks}" >> ${METRICFILE}
    tpcclusters=`grep -e "Event has.*TPC Clusters" ${logfile} | awk '//{print $5}'`
    echo "tpcclusters_${STAGE},${TAG} value=${tpcclusters}" >> ${METRICFILE}
  fi
done
