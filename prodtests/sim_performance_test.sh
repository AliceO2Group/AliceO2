#!/bin/bash

# Script performing a standard collection of
# simulation / reco / etc. tasks in order to collect benchmark
# metrics that can be fed into a time-series database for long term monitoring

# In principle we'd like to take O2 from the last nightly

# number of events / take from first argument or default
NEVENTS=${1:-"2"}
# generator / take from second argument or default
GEN=${2:-"pythia8pp"}

# STARTSEED
SEED=1234
# REPETITIONS FOR STATISTICS
REPS=5

# TIMESTAMP TO USE FOR INJECTION INTO DATABASE
TIMEST=`date +%s`
# convert to nanoseconds needed by InfluxDB
TIMEST="${TIMEST}000000000"

METRICFILE=metrics.dat
for ENGINE in TGeant3 TGeant4; do

### an outer loop over all different configurations
### this can include type of engine, central-barrel vs all, etc.
SIMCONFIG="${GEN}_N${NEVENTS}_${ENGINE}"
HOST=`hostname`

# include header information such as tested alidist tag and O2 tag
TAG="conf=${SIMCONFIG},host=${HOST}${ALIDISTCOMMIT:+,alidist=$ALIDISTCOMMIT}${O2COMMIT:+,o2=$O2COMMIT}"

echo "versions,${TAG} alidist=\"${ALIDISTCOMMIT}\",O2=\"${O2COMMIT}\" " >> metrics.dat

# we count some simple indicators for problems
WARNCOUNT=0
ERRORCOUNT=0
EXCEPTIONCOUNT=0

SECONDS=0

### ------ we run the transport simulation
LOGFILE=log_${SIMCONFIG}
SIMPREFIX=o2sim_${SIMCONFIG}
HITSTATFILE=hitstats_${SIMCONFIG}
taskset -c 1 o2-sim-serial -n ${NEVENTS} -e ${ENGINE} --skipModules ZDC -g ${GEN} --seed $SEED -o ${SIMPREFIX} > $LOGFILE 2>&1

### ------ extract number of hits
root -q -b -l ${O2_ROOT}/share/macro/analyzeHits.C\(\"${SIMPREFIX}\"\) > $HITSTATFILE

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

echo "time_sim,${TAG} init=$INITTIME,run=${RUNTIME} " >> metrics.dat
echo "mem_sim,${TAG} init=$INITMEM,run=${TOTALMEM} " >> metrics.dat

### ------ we run the digitization steps

## we record simple walltime and maximal memory used for each detector
SECONDS=0

digi_time_metrics="walltime_digitizer,${TAG} ";
digi_mem_metrics="maxmem_digitizer,${TAG} ";
hit_metrics="hitnumber,${TAG} ";

digi_total_time=0.
digi_total_mem=0.
hits_total=0

for d in TRD ITS EMC TPC MFT MID MCH FDD FV0 FT0 PHS TOF HMP CPV; do
  DIGILOGFILE=logdigi_${SIMCONFIG}_${d}
  DIGITIMEFILE=timedigi_${SIMCONFIG}_${d}

  TIME="#walltime %e" ${O2_ROOT}/share/scripts/monitor-mem.sh /usr/bin/time --output=${DIGITIMEFILE} o2-sim-digitizer-workflow --sims ${SIMPREFIX} -b --onlyDet ${d} --tpc-lanes 1 --configKeyValues "TRDSimParams.digithreads=1" > ${DIGILOGFILE} 2>&1

  # parse metrics
  maxmem=`awk '/PROCESS MAX MEM/{print $5}' ${DIGILOGFILE}`  # in MB
  walltime=`grep "#walltime" ${DIGITIMEFILE} | awk '//{print $2}'`

  # parse digitizer time which is more accurately representing the algorithmic component

  # add value for this detector
  digi_time_metrics="${digi_time_metrics}$d=${walltime},"
  digi_mem_metrics="${digi_mem_metrics}$d=${maxmem},"

  # accumulate
  digi_total_time=`bc <<< "${digi_total_time} + ${walltime}"`
  digi_total_mem=`bc <<< "${digi_total_mem} + ${maxmem}"`

  # analyse messages in log files
  WARN=`grep "\[WARN\]" ${DIGILOGFILE} | wc | awk '//{print $1}'`
  ERR=`grep "\[ERROR\]" ${DIGILOGFILE} | wc | awk '//{print $1}'`
  EXC=`grep "Exce" ${DIGILOGFILE} | wc | awk '//{print $1}'`
  WARNCOUNT=`bc <<< "${WARNCOUNT} + ${WARN}"`
  ERRORCOUNT=`bc <<< "${ERRORCOUNT} + ${ERR}"`
  EXCEPTIONCOUNT=`bc <<< "${EXCEPTIONCOUNT} + ${EXC}"`

  # analyse number of hits
  HITNUMBER=`grep "${d}" $HITSTATFILE | awk '//{print $2}'`
  hit_metrics="${hit_metrics}$d=${HITNUMBER},"
  hits_total=`bc <<< "${hits_total} + ${HITNUMBER}"`
done
DIGITTIME=$SECONDS  # with this we can calculate fractional contribution

# finish by adding a total field
digi_time_metrics="${digi_time_metrics}total=${digi_total_time} "
digi_mem_metrics="${digi_mem_metrics}total=${digi_total_mem} "
hit_metrics="${hit_metrics}total=${hits_total} "

echo ${digi_time_metrics} >> metrics.dat
echo ${digi_mem_metrics} >> metrics.dat
echo ${hit_metrics} >> metrics.dat
echo "warncount,${TAG} value=${WARNCOUNT} " >> metrics.dat
echo "errorcount,${TAG} value=${ERRORCOUNT} " >> metrics.dat
echo "exceptioncount,${TAG} value=${EXCEPTIONCOUNT} " >> metrics.dat


done # end loop over configurations engines

# remove empty DPL files
find ./ -size 0 -exec rm {} ';'
