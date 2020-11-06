#!/bin/bash
#
# A workflow performing a full system test:
# - simulation of digits
# - creation of raw data
# - reconstruction of raw data
#
# Note that this might require a production server to run.
#
# This script needs some binary objects (for the moment):
# - matbud.root + ITSdictionary.bin + ctf_dictionary.root
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
SHMSIZE=128000000000 # 128000000000  # Size of shared memory for messages
TPCTRACKERSCRATCHMEMORY=22000000000
NTIMEFRAMES=${NTIMEFRAMES:-1} # Number of time frames to process
TFDELAY=100 # Delay in seconds between publishing time frames
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
mkdir qed
cd qed
taskwrapper qedsim.log o2-sim -j $NJOBS -n$NEventsQED -m PIPE ITS MFT FT0 FV0 -g extgen --configKeyValues '"GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/QEDLoader.C;QEDGenParam.yMin=-5;QEDGenParam.yMax=6;QEDGenParam.ptMin=0.001;QEDGenParam.ptMax=1.;Diamond.width[2]=6."'
cd ..

GLOBALDPLOPT="-b --monitoring-backend no-op://"
taskwrapper sim.log o2-sim -n $NEvents --skipModules ZDC --configKeyValues "Diamond.width[2]=6." -g pythia8hi -j $NJOBS
taskwrapper digi.log o2-sim-digitizer-workflow -n $NEvents --simPrefixQED qed/o2sim --qed-x-section-ratio 3735  ${NOMCLABELS} --firstOrbit 0 --firstBC 0 --skipDet TRD --tpc-lanes $((NJOBS < 36 ? NJOBS : 36)) --shm-segment-size $SHMSIZE ${GLOBALDPLOPT}
taskwrapper digiTRD.log o2-sim-digitizer-workflow -n $NEvents --simPrefixQED qed/o2sim --qed-x-section-ratio 3735  ${NOMCLABELS} --firstOrbit 0 --firstBC 0 --onlyDet TRD --shm-segment-size $SHMSIZE ${GLOBALDPLOPT} --incontext collisioncontext.root --configKeyValues "TRDSimParams.digithreads=${NJOBS}"
mkdir raw
taskwrapper itsraw.log o2-its-digi2raw --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/ITS
taskwrapper mftraw.log o2-mft-digi2raw --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/MFT
taskwrapper ft0raw.log o2-ft0-digi2raw --file-per-link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -o raw/FT0
taskwrapper tpcraw.log o2-tpc-digits-to-rawzs  --file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' -i tpcdigits.root -o raw/TPC
taskwrapper tofraw.log o2-tof-reco-workflow ${GLOBALDPLOPT} --tof-raw-file-for link --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"' --output-type raw --tof-raw-outdir raw/TOF
taskwrapper midraw.log o2-mid-digits-to-raw-workflow ${GLOBALDPLOPT} --mid-raw-outdir raw/MID --mid-raw-perlink  --configKeyValues '"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0"'
cat raw/*/*.cfg > rawAll.cfg

# We run the workflow in both CPU-only and With-GPU mode
for STAGE in "NOGPU" "WITHGPU"; do

  if [[ "$STAGE" = "WITHGPU" ]]; then
    TPC_GPU_OPT="GPU_proc.deviceNum=0;GPU_global.deviceType=CUDA;GPU_proc.forceMemoryPoolSize=6000000000;GPU_proc.forceHostMemoryPoolSize=3000000000"
  else
    TPC_GPU_OPT="GPU_proc.forceHostMemoryPoolSize=${TPCTRACKERSCRATCHMEMORY}"
  fi

  ARGS_ALL="--session default"
  logfile=reco_${STAGE}.log
  taskwrapper ${logfile} "o2-raw-file-reader-workflow $ARGS_ALL --configKeyValues \"HBFUtils.nHBFPerTF=128\" --delay $TFDELAY --loop $NTIMEFRAMES --max-tf 0 --input-conf rawAll.cfg |  
o2-itsmft-stf-decoder-workflow $ARGS_ALL  |  
o2-itsmft-stf-decoder-workflow $ARGS_ALL --runmft true  |  
o2-its-reco-workflow $ARGS_ALL --trackerCA ${NOMCLABELS} --clusters-from-upstream --disable-root-output --entropy-encoding --configKeyValues \"fastMultConfig.cutMultClusLow=30;fastMultConfig.cutMultClusHigh=2000;fastMultConfig.cutMultVtxHigh=500\" |  
o2-itsmft-entropy-encoder-workflow $ARGS_ALL --runmft true |  
o2-tpc-reco-workflow $ARGS_ALL --input-type=zsraw ${NOMCLABELS} --output-type tracks,clusters,encoded-clusters,disable-writer --configKeyValues \"HBFUtils.nHBFPerTF=128;${TPC_GPU_OPT}\" |  
o2-ft0-flp-dpl-workflow $ARGS_ALL --disable-root-output |  
o2-ft0-reco-workflow $ARGS_ALL --disable-root-input --disable-root-output ${NOMCLABELS} |  
o2-ft0-entropy-encoder-workflow $ARGS_ALL  |  
o2-tpcits-match-workflow $ARGS_ALL -b --disable-root-input --disable-root-output ${NOMCLABELS}  |  
o2-mid-raw-to-digits-workflow $ARGS_ALL | \  
o2-mid-reco-workflow $ARGS_ALL --disable-root-output |  
o2-mid-entropy-encoder-workflow $ARGS_ALL |  
o2-tof-compressor $ARGS_ALL |  
o2-tof-reco-workflow $ARGS_ALL --configKeyValues \"HBFUtils.nHBFPerTF=128\" --input-type raw --output-type ctf,clusters,matching-info --disable-root-output  ${NOMCLABELS}  |  
o2-tpc-scdcalib-interpolation-workflow $ARGS_ALL --disable-root-output --disable-root-input --shm-segment-size $SHMSIZE ${GLOBALDPLOPT}"


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

