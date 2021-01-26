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


NEvents=${NEvents:-10} #550 for full TF (the number of PbPb events)
NEventsQED=${NEventsQED:-1000} #35000 for full TF
NCPUS=$(getNumberOfPhysicalCPUCores)
echo "Found ${NCPUS} physical CPU cores"
NJOBS=${NJOBS:-"${NCPUS}"}
SHMSIZE=${SHMSIZE:-8000000000} # Size of shared memory for messages (use 128 GB for 550 event full TF)
TPCTRACKERSCRATCHMEMORY=${SHMSIZE:-4000000000} # Size of memory allocated by TPC tracker. (Use 24 GB for 550 event full TF)
ENABLE_GPU_TEST=${ENABLE_GPU_TEST:-0} # Run the full system test also on the GPU
NTIMEFRAMES=${NTIMEFRAMES:-1} # Number of time frames to process
TFDELAY=${TFDELAY:-100} # Delay in seconds between publishing time frames
NOMCLABELS="--disable-mc"

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
mkdir -p qed
cd qed
taskwrapper qedsim.log o2-sim -j $NJOBS -n$NEventsQED -m PIPE ITS MFT FT0 FV0 -g extgen --configKeyValues '"GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/QEDLoader.C;QEDGenParam.yMin=-5;QEDGenParam.yMax=6;QEDGenParam.ptMin=0.001;QEDGenParam.ptMax=1.;Diamond.width[2]=6."'
cd ..

GLOBALDPLOPT="-b" #  --monitoring-backend no-op:// is currently removed due to https://alice.its.cern.ch/jira/browse/O2-1887
taskwrapper sim.log o2-sim -n $NEvents --skipModules ZDC --configKeyValues "Diamond.width[2]=6." -g pythia8hi -j $NJOBS
taskwrapper digi.log o2-sim-digitizer-workflow -n $NEvents --simPrefixQED qed/o2sim --qed-x-section-ratio 3735  ${NOMCLABELS} --firstOrbit 0 --firstBC 0 --skipDet TRD --tpc-lanes $((NJOBS < 36 ? NJOBS : 36)) --shm-segment-size $SHMSIZE ${GLOBALDPLOPT}
taskwrapper digiTRD.log o2-sim-digitizer-workflow -n $NEvents --simPrefixQED qed/o2sim --qed-x-section-ratio 3735  ${NOMCLABELS} --firstOrbit 0 --firstBC 0 --onlyDet TRD --shm-segment-size $SHMSIZE ${GLOBALDPLOPT} --incontext collisioncontext.root --configKeyValues "TRDSimParams.digithreads=${NJOBS}"
mkdir -p raw
taskwrapper itsraw.log o2-its-digi2raw --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/ITS
taskwrapper mftraw.log o2-mft-digi2raw --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/MFT
taskwrapper ft0raw.log o2-ft0-digi2raw --file-per-link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/FT0
taskwrapper fv0raw.log o2-fv0-digi2raw --file-per-link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/FV0
taskwrapper fddraw.log o2-fdd-digit2raw --file-per-link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/FDD
taskwrapper tpcraw.log o2-tpc-digits-to-rawzs  --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -i tpcdigits.root -o raw/TPC
taskwrapper tofraw.log o2-tof-reco-workflow ${GLOBALDPLOPT} --tof-raw-file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' --output-type raw --tof-raw-outdir raw/TOF
taskwrapper midraw.log o2-mid-digits-to-raw-workflow ${GLOBALDPLOPT} --mid-raw-outdir raw/MID --mid-raw-perlink  --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"'
taskwrapper emcraw.log o2-emcal-rawcreator --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/EMC
taskwrapper phsraw.log o2-phos-digi2raw  --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/PHS
taskwrapper cpvraw.log o2-cpv-digi2raw  --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/CPV
cat raw/*/*.cfg > rawAll.cfg

# We run the workflow in both CPU-only and With-GPU mode
STAGES="NOGPU"
if [ $ENABLE_GPU_TEST != "0" ]; then
  STAGES+=" WITHGPU"
fi
STAGES+=" ASYNC"
for STAGE in $STAGES; do
  logfile=reco_${STAGE}.log

  ARGS_ALL="--session default"
  DICTCREATION=""
  if [[ "$STAGE" = "WITHGPU" ]]; then
    export CREATECTFDICT=0
    export GPUTYPE=CUDA
    export GPUMEMSIZE=6000000000
    export HOSTMEMSIZE=1000000000
    export SYNCMODE=1
    export CTFINPUT=0
    export SAVECTF=0
  elif [[ "$STAGE" = "ASYNC" ]]; then
    export CREATECTFDICT=0
    export GPUTYPE=CPU
    export SYNCMODE=0
    export HOSTMEMSIZE=$TPCTRACKERSCRATCHMEMORY
    export CTFINPUT=1
    export SAVECTF=0
  else
    export CREATECTFDICT=1
    export GPUTYPE=CPU
    export SYNCMODE=0
    export HOSTMEMSIZE=$TPCTRACKERSCRATCHMEMORY
    export CTFINPUT=0
    export SAVECTF=1
    JOBUTILS_JOB_SKIPCREATEDONE=1 taskwrapper ${logfile} "rm -f ctf_dictionary.root"
    unset JOBUTILS_JOB_SKIPCREATEDONE
  fi
  export SHMSIZE
  export NTIMEFRAMES
  export TFDELAY
  export GLOBALDPLOPT

  taskwrapper ${logfile} "$O2_ROOT/prodtests/full-system-test/dpl-workflow.sh"

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
