#!/bin/bash

# add distortion maps
# https://alice.its.cern.ch/jira/browse/O2-3346?focusedCommentId=300982&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-300982
#
# export O2DPG_ENABLE_TPC_DISTORTIONS=ON
# SCFile=$PWD/distortions_5kG_lowIR.root # file needs to be downloaded
# export O2DPG_TPC_DIGIT_EXTRA=" --distortionType 2 --readSpaceCharge ${SCFile} "

#
# procedure setting up and executing an anchored MC
#

# make sure O2DPG + O2 is loaded
[ ! "${O2DPG_ROOT}" ] && echo "Error: This needs O2DPG loaded" && exit 1
[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1

# ------ CREATE AN MC CONFIG STARTING FROM RECO SCRIPT --------
# - this part should not be done on the GRID, where we should rather
#   point to an existing config (O2DPG repo or local disc or whatever)
export ALIEN_JDL_LPMANCHORYEAR=${ALIEN_JDL_LPMANCHORYEAR:-2022}
RUNNUMBER=${ALIEN_JDL_LPMRUNNUMBER:-517616}
#INTERACTIONRATE=${INTERACTIONRATE:-2000}

# get the async script (we need to modify it)
# the script location can be configured with a JDL option

ALIEN_JDL_LPMPRODUCTIONTAG_KEEP=$ALIEN_JDL_LPMPRODUCTIONTAG
echo "Substituting ALIEN_JDL_LPMPRODUCTIONTAG=$ALIEN_JDL_LPMPRODUCTIONTAG with ALIEN_JDL_LPMANCHORPRODUCTION=$ALIEN_JDL_LPMANCHORPRODUCTION for simulating reco pass..."
ALIEN_JDL_LPMPRODUCTIONTAG=$ALIEN_JDL_LPMANCHORPRODUCTION

# ZDC causes issues for sim
#export ALIEN_JDL_WORKFLOWDETECTORS=ITS,TPC,TOF,FV0,FT0,FDD,MID,MFT,MCH,TRD,EMC,PHS,CPV,HMP,ZDC,CTP
export ALIEN_JDL_WORKFLOWDETECTORS=ITS,TPC,TOF,FV0,FT0,FDD,MID,MFT,MCH,TRD,EMC,PHS,CPV,HMP,CTP

### async_pass.sh
DPGRECO=$O2DPG_ROOT/DATA/production/configurations/asyncReco/async_pass.sh
SETENV=$O2DPG_ROOT/DATA/production/configurations/asyncReco/setenv_extra.sh

if [[ -f async_pass.sh ]]; then
    chmod +x async_pass.sh
    DPGRECO=./async_pass.sh
else
    cp -v $DPGRECO .
fi

if [[ ! -f setenv_extra.sh ]]; then
    setenv=${DPGRECO%/*}/setenv_extra.sh
    [[ -f ${setenv} ]] && cp ${setenv} . || cp ${SETENV} .
fi

echo "[INFO alien_async_pass.sh] Setting up DPGRECO to ${DPGRECO}"

#settings that are MC-specific
sed -i 's/GPU_global.dEdxUseFullGainMap=1;GPU_global.dEdxDisableResidualGainMap=1/GPU_global.dEdxSplineTopologyCorrFile=splines_for_dedx_V1_MC_iter0_PP.root;GPU_global.dEdxDisableTopologyPol=1;GPU_global.dEdxDisableGainMap=1;GPU_global.dEdxDisableResidualGainMap=1;GPU_global.dEdxDisableResidualGain=1/' setenv_extra.sh
### ???

chmod +x async_pass.sh

# take out line running the workflow (if we don't have data input)
[ ${CTF_TEST_FILE} ] || sed -i '/WORKFLOWMODE=run/d' async_pass.sh

# create workflow ---> creates the file that can be parsed
export IGNORE_EXISTING_SHMFILES=1
touch list.list

./async_pass.sh ${CTF_TEST_FILE:-""} 2&> async_pass_log.log
RECO_RC=$?

echo "RECO finished with ${RECO_RC}"
if [ "${NO_MC}" ]; then
  return ${RECO_RC} 2>/dev/null || exit ${RECO_RC} # optionally quit here and don't do MC (useful for testing)
fi

ALIEN_JDL_LPMPRODUCTIONTAG=$ALIEN_JDL_LPMPRODUCTIONTAG_KEEP
echo "Setting back ALIEN_JDL_LPMPRODUCTIONTAG to $ALIEN_JDL_LPMPRODUCTIONTAG"

# now create the local MC config file --> config-config.json
${O2DPG_ROOT}/UTILS/parse-async-WorkflowConfig.py

# check if config reasonably created
if [[ `grep "o2-ctf-reader-workflow-options" config-json.json 2> /dev/null | wc -l` == "0" ]]; then
  echo "Problem in anchor config creation. Stopping."
  exit 1
fi

# -- CREATE THE MC JOB DESCRIPTION ANCHORED TO RUN --

NWORKERS=${NWORKERS:-8}
MODULES="--skipModules ZDC"
SIMENGINE=${SIMENGINE:-TGeant4}
SIMENGINE=${ALIEN_JDL_SIMENGINE:-${SIMENGINE}}
NTIMEFRAMES=${NTIMEFRAMES:-50}
NSIGEVENTS=${NSIGEVENTS:-22}

SPLITID=${SPLITID:-0}
PRODSPLIT=${PRODSPLIT:-100}
CYCLE=${CYCLE:-0}
# let SEED=$SPLITID+$CYCLE*$PRODSPLIT
SEED=${ALIEN_PROC_ID}

# create workflow
# THIS NEEDS TO COME FROM OUTSIDE
# echo "$" | awk -F' -- ' '{print $1, $3}'

baseargs="-tf ${NTIMEFRAMES} --split-id ${SPLITID} --prod-split ${PRODSPLIT} --cycle ${CYCLE} --run-number ${RUNNUMBER}"

# THIS NEEDS TO COME FROM OUTSIDE
remainingargs="-gen pythia8 -proc cdiff -seed ${SEED} -ns ${NSIGEVENTS} --include-local-qc"

remainingargs="${remainingargs} -e ${SIMENGINE} -j ${NWORKERS}"
remainingargs="${remainingargs} -productionTag ${ALIEN_JDL_LPMPRODUCTIONTAG:-alibi_anchorTest_tmp}"
remainingargs="${remainingargs} --anchor-config config-json.json"

echo "baseargs: ${baseargs}"
echo "remainingargs: ${remainingargs}"

# query CCDB has changed, w/o "_"
${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow_anchored.py ${baseargs} -- ${remainingargs} &> timestampsampling_${RUNNUMBER}.log
if [ "$?" != "0" ] ; then
    echo "ERROR: Problem during anchor timestamp sampling " && exit 1
fi
TIMESTAMP=`grep "Determined timestamp to be" timestampsampling_${RUNNUMBER}.log | awk '//{print $6}'`
echo "TIMESTAMP IS ${TIMESTAMP}"

# -- PREFETCH CCDB OBJECTS TO DISC      --
# (make sure the right objects at the right timestamp are fetched
#  until https://alice.its.cern.ch/jira/browse/O2-2852 is fixed)
export ALICEO2_CCDB_LOCALCACHE=$PWD/.ccdb
[ ! -d .ccdb ] && mkdir .ccdb

CCDBOBJECTS="/CTP/Calib/OrbitReset /GLO/Config/GRPMagField/ /GLO/Config/GRPLHCIF /ITS/Calib/DeadMap /ITS/Calib/NoiseMap /ITS/Calib/ClusterDictionary /TPC/Calib/PadGainFull /TPC/Calib/TopologyGain /TPC/Calib/TimeGain /TPC/Calib/PadGainResidual /TPC/Config/FEEPad /TOF/Calib/Diagnostic /TOF/Calib/LHCphase /TOF/Calib/FEELIGHT /TOF/Calib/ChannelCalib /MFT/Calib/DeadMap /MFT/Calib/NoiseMap /MFT/Calib/ClusterDictionary /FT0/Calibration/ChannelTimeOffset /FV0/Calibration/ChannelTimeOffset /GLO/GRP/BunchFilling"

${O2_ROOT}/bin/o2-ccdb-downloadccdbfile --host http://alice-ccdb.cern.ch/ -p ${CCDBOBJECTS} -d .ccdb --timestamp ${TIMESTAMP}
if [ ! "$?" == "0" ]; then
  echo "Problem during CCDB prefetching of ${CCDBOBJECTS}. Exiting."
  exit 1
fi

# -- Create aligned geometry using ITS and MFT ideal alignments to avoid overlaps in geant
CCDBOBJECTS_IDEAL_MC="ITS/Calib/Align MFT/Calib/Align"
TIMESTAMP_IDEAL_MC=1
${O2_ROOT}/bin/o2-ccdb-downloadccdbfile --host http://alice-ccdb.cern.ch/ -p ${CCDBOBJECTS_IDEAL_MC} -d .ccdb --timestamp ${TIMESTAMP_IDEAL_MC}
if [ ! "$?" == "0" ]; then
  echo "Problem during CCDB prefetching of ${CCDBOBJECTS_IDEAL_MC}. Exiting."
  exit 1
fi

# ${O2_ROOT}/bin/o2-create-aligned-geometry-workflow --configKeyValues "HBFUtils.startTime=${TIMESTAMP}" --condition-remap=file://${ALICEO2_CCDB_LOCALCACHE}=ITS/Calib/Align,MFT/Calib/Align -b
echo "run with echo in pipe" | ${O2_ROOT}/bin/o2-create-aligned-geometry-workflow --configKeyValues "HBFUtils.startTime=${TIMESTAMP}" --condition-remap=file://${ALICEO2_CCDB_LOCALCACHE}=ITS/Calib/Align,MFT/Calib/Align -b
mkdir -p $ALICEO2_CCDB_LOCALCACHE/GLO/Config/GeometryAligned
ln -s -f $PWD/o2sim_geometry-aligned.root $ALICEO2_CCDB_LOCALCACHE/GLO/Config/GeometryAligned/snapshot.root

# -- RUN THE MC WORKLOAD TO PRODUCE AOD --

export FAIRMQ_IPC_PREFIX=./

echo "Ready to start main workflow"

${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt ${ALIEN_JDL_O2DPGWORKFLOWTARGET:-aod} --cpu-limit ${ALIEN_JDL_CPULIMIT:-8}
MCRC=$?  # <--- we'll report back this code
#exit 0

if [[ "${MCRC}" = "0" && "${remainingargs}" == *"--include-local-qc"* ]] ; then
  # do QC tasks
  echo "Doing QC"
  ${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json --target-labels QC --cpu-limit ${ALIEN_JDL_CPULIMIT:-8} -k
  MCRC=$?
fi

#
# full logs tar-ed for output, regardless the error code or validation - to catch also QC logs...
#
if [[ -n "$ALIEN_PROC_ID" ]]; then
  find ./ \( -name "*.log*" -o -name "*mergerlog*" -o -name "*serverlog*" -o -name "*workerlog*" -o -name "pythia8.cfg" \) | tar -czvf debug_log_archive.tgz -T -
  find ./ \( -name "*.log*" -o -name "*mergerlog*" -o -name "*serverlog*" -o -name "*workerlog*" -o -name "*.root" \) | tar -czvf debug_full_archive.tgz -T -
fi

unset FAIRMQ_IPC_PREFIX

return ${MCRC} 2>/dev/null || exit ${MCRC}

