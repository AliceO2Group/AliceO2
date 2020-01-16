#!/bin/bash

# Script performing a standard collection of 
# simulation / reco / etc. tasks in order to collect benchmark 
# metrics that can be fed into a time-series database for long term monitoring

# In principle we'd like to take O2 from the last nightly

# number of events / take from first argument or default
NEVENTS=${1:-"2"}
# generator / take from second argument or default
GEN=${2:-"pythia8"}

# STARTSEED
SEED=1234
# REPETITIONS FOR STATISTICS
REPS=5

# TIMESTAMP TO USE FOR INJECTION INTO DATABASE
TIMEST=`date +%s`

METRICFILE=metrics.dat
for ENGINE in TGeant3 TGeant4; do

### an outer loop over all different configurations
### this can include type of engine, central-barrel vs all, etc.
SIMCONFIG="${GEN}_N${NEVENTS}_${ENGINE}"
HOST=`hostname`

# we count some simple indicators for problems
WARNCOUNT=0
ERRORCOUNT=0
EXCEPTIONCOUNT=0

SECONDS=0

### ------ we run the transport simulation
LOGFILE=log_${SIMCONFIG}
HITFILE=o2sim_${SIMCONFIG}
taskset -c 1 o2-sim-serial -n ${NEVENTS} -e ${ENGINE} --skipModules ZDC -g ${GEN} --seed $SEED -o ${HITFILE} > $LOGFILE 2>&1

TRANSPORTTIME=$SECONDS

### ------- extract metrics for simulation

# initialization real time
INITTIME=`grep "Init: Real time" $LOGFILE | awk '//{print $5}'`
# initialization memory
INITMEM=`grep "Init: Memory" $LOGFILE | awk '//{print $5}'`

# get the actual transport runtime 
RUNTIME=`grep " Real time" $LOGFILE | grep -v "Init:" | awk '//{print $4}'`
# initialization memory
TOTALMEM=`grep " Memory used" $LOGFILE | grep -v "Init:" | awk '//{print $4}'`

WARN=`grep "\[WARN\]" $LOGFILE | wc | awk '//{print $1}'`
ERR=`grep "\[ERROR\]" $LOGFILE | wc | awk '//{print $1}'`
EXC=`grep "Exce" $LOGFILE | wc | awk '//{print $1}'`
WARNCOUNT=`bc <<< "${WARNCOUNT} + ${WARN}"`
ERRORCOUNT=`bc <<< "${ERRORCOUNT} + ${ERR}"`
EXCEPTIONCOUNT=`bc <<< "${EXCEPTIONCOUNT} + ${EXC}"`

echo "time_sim,conf=$SIMCONFIG,host=${HOST} init=$INITTIME,run=${RUNTIME} ${TIMEST}" >> metrics.dat
echo "mem_sim,conf=$SIMCONFIG,host=${HOST} init=$INITMEM,run=${TOTALMEM} ${TIMEST}" >> metrics.dat

### ------ we run the digitization steps

## we record simple walltime and maximal memory used for each detector
SECONDS=0

digi_time_metrics="walltime_digitizer,conf=$SIMCONFIG,host=${HOST} ";
digi_mem_metrics="maxmem_digitizer,conf=$SIMCONFIG,host=${HOST} ";

digi_total_time=0.
digi_total_mem=0.
# for the moment soft link files to what digitizer expect
unlink o2sim.root
unlink o2sim_grp.root
ln -s ${HITFILE}.root o2sim.root
ln -s ${HITFILE}_grp.root o2sim_grp.root
for d in TRD ITS EMC TPC MFT MCH MID; do
  DIGILOGFILE=logdigi_${SIMCONFIG}_${d}

  TIME="#walltime %e" ${O2_ROOT}/share/scripts/monitor-mem.sh time o2-sim-digitizer-workflow -b --onlyDet ${d} --tpc-lanes 1 --configKeyValues "TRDSimParams.digithreads=1" > ${DIGILOGFILE} 2>&1

  # parse metrics
  maxmem=`awk '/PROCESS MAX MEM/{print $5}' ${DIGILOGFILE}`  # in MB
  walltime=`grep "#walltime" ${DIGILOGFILE} | awk '//{print $2}'`

  # parse digitizer time which is more accurately representing the algorithmic component

  # add value for this detector
  digi_time_metrics="${digi_time_metrics}$d=${walltime},"
  digi_mem_metrics="${digi_mem_metrics}$d=${maxmem},"

  # accumulate
  digi_total_time=`bc <<< "${digi_total_time} + ${walltime}"`
  digi_total_mem=`bc <<< "${digi_total_mem} + ${maxmem}"`

  # analyse messages
  WARN=`grep "\[WARN\]" ${DIGILOGFILE} | wc | awk '//{print $1}'`
  ERR=`grep "\[ERROR\]" ${DIGILOGFILE} | wc | awk '//{print $1}'`
  EXC=`grep "Exce" ${DIGILOGFILE} | wc | awk '//{print $1}'`
  WARNCOUNT=`bc <<< "${WARNCOUNT} + ${WARN}"`
  ERRORCOUNT=`bc <<< "${ERRORCOUNT} + ${ERR}"`
  EXCEPTIONCOUNT=`bc <<< "${EXCEPTIONCOUNT} + ${EXC}"`

done
DIGITTIME=$SECONDS  # with this we can calculate fractional contribution
# add value for this detector
digi_time_metrics="${digi_time_metrics}total=${digi_total_time} ${TIMEST}"
digi_mem_metrics="${digi_mem_metrics}total=${digi_total_mem} ${TIMEST}"

echo ${digi_time_metrics} >> metrics.dat
echo ${digi_mem_metrics} >> metrics.dat
echo "warncount,conf=$SIMCONFIG,host=${HOST} value=${WARNCOUNT} ${TIMEST}" >> metrics.dat
echo "errorcount,conf=$SIMCONFIG,host=${HOST} value=${ERRORCOUNT} ${TIMEST}" >> metrics.dat
echo "exceptioncount,conf=$SIMCONFIG,host=${HOST} value=${EXCEPTIONCOUNT} ${TIMEST}" >> metrics.dat


done # end loop over configurations engines
