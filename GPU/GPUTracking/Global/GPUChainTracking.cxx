// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTracking.cxx
/// \author David Rohr

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <fstream>
#include <chrono>

#include "GPUChainTracking.h"
#include "GPUChainTrackingGetters.inc"
#include "GPUReconstructionIO.h"
#include "GPUChainTrackingDefs.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTPCTrack.h"
#include "GPUTPCHitId.h"
#include "GPUTPCClusterData.h"
#include "TPCZSLinkMapping.h"
#include "GPUTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUTRDTrackletLabels.h"
#include "GPUTRDRecoParam.h"
#include "display/GPUDisplayInterface.h"
#include "GPUQA.h"
#include "GPULogging.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"
#include "GPUNewCalibValues.h"
#include "GPUTriggerOutputs.h"
#include "GPUDefParametersRuntime.h"

#include "GPUTPCClusterStatistics.h"
#include "GPUHostDataTypes.h"
#include "GPUTPCCFChainContext.h"
#include "GPUTrackingRefit.h"
#include "CalibdEdxContainer.h"

#include "TPCFastTransformPOD.h"

#include "utils/linux_helpers.h"
#include "utils/strtag.h"
using namespace o2::gpu;

#include "GPUO2DataTypes.h"

using namespace o2::tpc;
using namespace o2::trd;

GPUChainTracking::GPUChainTracking(GPUReconstruction* rec, uint32_t maxTPCHits, uint32_t maxTRDTracklets) : GPUChain(rec), mIOPtrs(processors()->ioPtrs), mInputsHost(new GPUTrackingInputProvider), mInputsShadow(new GPUTrackingInputProvider), mClusterNativeAccess(new ClusterNativeAccess), mTriggerBuffer(new GPUTriggerOutputs), mMaxTPCHits(maxTPCHits), mMaxTRDTracklets(maxTRDTracklets), mDebugFile(new std::ofstream)
{
  ClearIOPointers();
  mFlatObjectsShadow.mChainTracking = this;
  mFlatObjectsDevice.mChainTracking = this;
}

GPUChainTracking::~GPUChainTracking() = default;

void GPUChainTracking::RegisterPermanentMemoryAndProcessors()
{
  if (mRec->IsGPU()) {
    mFlatObjectsShadow.InitGPUProcessor(mRec, GPUProcessor::PROCESSOR_TYPE_SLAVE);
    mFlatObjectsDevice.InitGPUProcessor(mRec, GPUProcessor::PROCESSOR_TYPE_DEVICE, &mFlatObjectsShadow);
    mFlatObjectsShadow.mMemoryResFlat = mRec->RegisterMemoryAllocation(&mFlatObjectsShadow, &GPUTrackingFlatObjects::SetPointersFlatObjects, GPUMemoryResource::MEMORY_PERMANENT, "CalibObjects");
  }

  mRec->RegisterGPUProcessor(mInputsHost.get(), mRec->IsGPU());
  if (GetRecoSteps() & RecoStep::TPCSectorTracking) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      mRec->RegisterGPUProcessor(&processors()->tpcTrackers[i], GetRecoStepsGPU() & RecoStep::TPCSectorTracking);
    }
  }
  if (GetRecoSteps() & RecoStep::TPCMerging) {
    mRec->RegisterGPUProcessor(&processors()->tpcMerger, GetRecoStepsGPU() & RecoStep::TPCMerging);
  }
  if (GetRecoSteps() & RecoStep::TRDTracking) {
    mRec->RegisterGPUProcessor(&processors()->trdTrackerGPU, GetRecoStepsGPU() & RecoStep::TRDTracking);
  }
  if (GetRecoSteps() & RecoStep::TRDTracking) {
    mRec->RegisterGPUProcessor(&processors()->trdTrackerO2, GetRecoStepsGPU() & RecoStep::TRDTracking);
  }
  if (GetRecoSteps() & RecoStep::TPCCompression) {
    mRec->RegisterGPUProcessor(&processors()->tpcCompressor, GetRecoStepsGPU() & RecoStep::TPCCompression);
  }
  if (GetRecoSteps() & RecoStep::TPCDecompression) {
    mRec->RegisterGPUProcessor(&processors()->tpcDecompressor, GetRecoStepsGPU() & RecoStep::TPCDecompression);
  }
  if (GetRecoSteps() & RecoStep::TPCClusterFinding) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      mRec->RegisterGPUProcessor(&processors()->tpcClusterer[i], GetRecoStepsGPU() & RecoStep::TPCClusterFinding);
#ifdef GPUCA_HAS_ONNX
      mRec->RegisterGPUProcessor(&processors()->tpcNNClusterer[i], GetRecoStepsGPU() & RecoStep::TPCClusterFinding);
#endif
    }
  }
  if (GetRecoSteps() & RecoStep::Refit) {
    mRec->RegisterGPUProcessor(&processors()->trackingRefit, GetRecoStepsGPU() & RecoStep::Refit);
  }
#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
  mRec->RegisterGPUProcessor(&processors()->debugOutput, true);
#endif
  mRec->AddGPUEvents(mEvents);
}

void GPUChainTracking::RegisterGPUProcessors()
{
  if (mRec->IsGPU()) {
    mRec->RegisterGPUDeviceProcessor(mInputsShadow.get(), mInputsHost.get());
  }
  memcpy((void*)&processorsShadow()->trdTrackerGPU, (const void*)&processors()->trdTrackerGPU, sizeof(processors()->trdTrackerGPU));
  if (GetRecoStepsGPU() & RecoStep::TPCSectorTracking) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcTrackers[i], &processors()->tpcTrackers[i]);
    }
  }
  if (GetRecoStepsGPU() & RecoStep::TPCMerging) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcMerger, &processors()->tpcMerger);
  }
  if (GetRecoStepsGPU() & RecoStep::TRDTracking) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->trdTrackerGPU, &processors()->trdTrackerGPU);
  }

  memcpy((void*)&processorsShadow()->trdTrackerO2, (const void*)&processors()->trdTrackerO2, sizeof(processors()->trdTrackerO2));
  if (GetRecoStepsGPU() & RecoStep::TRDTracking) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->trdTrackerO2, &processors()->trdTrackerO2);
  }
  if (GetRecoStepsGPU() & RecoStep::TPCCompression) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcCompressor, &processors()->tpcCompressor);
  }
  if (GetRecoStepsGPU() & RecoStep::TPCDecompression) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcDecompressor, &processors()->tpcDecompressor);
  }
  if (GetRecoStepsGPU() & RecoStep::TPCClusterFinding) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcClusterer[i], &processors()->tpcClusterer[i]);
#ifdef GPUCA_HAS_ONNX
      mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcNNClusterer[i], &processors()->tpcNNClusterer[i]);
#endif
    }
  }
  if (GetRecoStepsGPU() & RecoStep::Refit) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->trackingRefit, &processors()->trackingRefit);
  }
#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
  mRec->RegisterGPUDeviceProcessor(&processorsShadow()->debugOutput, &processors()->debugOutput);
#endif
}

void GPUChainTracking::MemorySize(size_t& gpuMem, size_t& pageLockedHostMem)
{
  gpuMem = constants::GPU_DEFAULT_MEMORY_SIZE;
  pageLockedHostMem = constants::GPU_DEFAULT_HOST_MEMORY_SIZE;
}

bool GPUChainTracking::ValidateSteps()
{
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCdEdx) && !(GetRecoSteps() & gpudatatypes::RecoStep::TPCMerging)) {
    GPUError("Invalid Reconstruction Step Setting: dEdx requires TPC Merger to be active");
    return false;
  }
  if ((GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCdEdx) && !(GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCMerging)) {
    GPUError("Invalid GPU Reconstruction Step Setting: dEdx requires TPC Merger to be active");
    return false;
  }
  if (((GetRecoSteps() & gpudatatypes::RecoStep::TPCSectorTracking) || (GetRecoSteps() & gpudatatypes::RecoStep::TPCMerging)) && !(GetRecoSteps() & gpudatatypes::RecoStep::TPCConversion)) {
    GPUError("Invalid Reconstruction Step Setting: Tracking requires TPC Conversion to be active");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCClusterFinding) && !(GetRecoStepsInputs() & gpudatatypes::InOutType::TPCRaw)) {
    GPUError("Invalid input, TPC Clusterizer needs TPC raw input");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCMerging) && !(GetRecoSteps() & gpudatatypes::RecoStep::TPCConversion)) {
    GPUError("Invalid input / output / step, merger cannot read/store sectors tracks and needs TPC conversion");
    return false;
  }
  bool tpcClustersAvail = (GetRecoStepsInputs() & gpudatatypes::InOutType::TPCClusters) || (GetRecoSteps() & gpudatatypes::RecoStep::TPCClusterFinding) || (GetRecoSteps() & gpudatatypes::RecoStep::TPCDecompression);
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCMerging) && !tpcClustersAvail) {
    GPUError("Invalid Inputs for track merging, TPC Clusters required");
    return false;
  }
#ifdef GPUCA_RUN2
  if (GetRecoSteps() & gpudatatypes::RecoStep::TPCClusterFinding) {
    GPUError("Can not run TPC GPU Cluster Finding with Run 2 Data");
    return false;
  }
#endif
  if (((GetRecoSteps() & gpudatatypes::RecoStep::TPCConversion) || (GetRecoSteps() & gpudatatypes::RecoStep::TPCSectorTracking) || (GetRecoSteps() & gpudatatypes::RecoStep::TPCCompression) || (GetRecoSteps() & gpudatatypes::RecoStep::TPCdEdx)) && !tpcClustersAvail) {
    GPUError("Missing input for TPC Cluster conversion / sector tracking / compression / dEdx: TPC Clusters required");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCMerging) && !(GetRecoSteps() & gpudatatypes::RecoStep::TPCSectorTracking)) {
    GPUError("Input for TPC merger missing");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCCompression) && !((GetRecoStepsInputs() & gpudatatypes::InOutType::TPCMergedTracks) || (GetRecoSteps() & gpudatatypes::RecoStep::TPCMerging))) {
    GPUError("Input for TPC compressor missing");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TRDTracking) && (!((GetRecoStepsInputs() & gpudatatypes::InOutType::TPCMergedTracks) || (GetRecoSteps() & gpudatatypes::RecoStep::TPCMerging)) || !(GetRecoStepsInputs() & gpudatatypes::InOutType::TRDTracklets))) {
    GPUError("Input for TRD Tracker missing");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TRDTracking) && (processors()->calibObjects.trdRecoParam == nullptr)) {
    GPUError("TRD Reco Parameters are missing");
    return false;
  }
  if ((GetRecoStepsOutputs() & gpudatatypes::InOutType::TPCRaw) || (GetRecoStepsOutputs() & gpudatatypes::InOutType::TRDTracklets)) {
    GPUError("TPC Raw / TPC Clusters / TRD Tracklets cannot be output");
    return false;
  }
  if ((GetRecoStepsOutputs() & gpudatatypes::InOutType::TPCMergedTracks) && !(GetRecoSteps() & gpudatatypes::RecoStep::TPCMerging)) {
    GPUError("No TPC Merged Track Output available");
    return false;
  }
  if ((GetRecoStepsOutputs() & gpudatatypes::InOutType::TPCCompressedClusters) && !(GetRecoSteps() & gpudatatypes::RecoStep::TPCCompression)) {
    GPUError("No TPC Compression Output available");
    return false;
  }
  if ((GetRecoStepsOutputs() & gpudatatypes::InOutType::TRDTracks) && !(GetRecoSteps() & gpudatatypes::RecoStep::TRDTracking)) {
    GPUError("No TRD Tracker Output available");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCdEdx) && (processors()->calibObjects.dEdxCalibContainer == nullptr)) {
    GPUError("Cannot run dE/dx without dE/dx calibration container object");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCClusterFinding) && processors()->calibObjects.tpcPadGain == nullptr) {
    GPUError("Cannot run gain calibration without calibration object");
    return false;
  }
  if ((GetRecoSteps() & gpudatatypes::RecoStep::TPCClusterFinding) && processors()->calibObjects.tpcZSLinkMapping == nullptr && mIOPtrs.tpcZS != nullptr) {
    GPUError("Cannot run TPC ZS Decoder without mapping object. (tpczslinkmapping.dump missing?)");
    return false;
  }
  return true;
}

bool GPUChainTracking::ValidateSettings()
{
  int32_t gatherMode = mRec->GetProcessingSettings().tpcCompressionGatherMode == -1 ? mRec->getGPUParameters(mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression).par_COMP_GATHER_MODE : mRec->GetProcessingSettings().tpcCompressionGatherMode;
  if ((param().rec.tpc.nWays & 1) == 0) {
    GPUError("nWay setting musst be odd number!");
    return false;
  }
  if (param().rec.tpc.mergerInterpolateErrors && param().rec.tpc.nWays < 3) {
    GPUError("Cannot do error interpolation with NWays < 3!");
    return false;
  }
  if (param().continuousMaxTimeBin > (int32_t)GPUSettings::TPC_MAX_TF_TIME_BIN) {
    GPUError("configured max time bin exceeds 256 orbits");
    return false;
  }
  if ((GetRecoStepsGPU() & RecoStep::TPCClusterFinding) && std::max(GetProcessingSettings().nTPCClustererLanes + 1, GetProcessingSettings().nTPCClustererLanes * 2) + (GetProcessingSettings().doublePipeline ? 1 : 0) > (int32_t)mRec->NStreams()) {
    GPUError("NStreams of %d insufficient for %d nTPCClustererLanes", mRec->NStreams(), (int32_t)GetProcessingSettings().nTPCClustererLanes);
    return false;
  }
  if ((mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression) && GetProcessingSettings().noGPUMemoryRegistration && gatherMode != 3) {
    GPUError("noGPUMemoryRegistration only possible with gather mode 3 (set to %d / %d)", mRec->GetProcessingSettings().tpcCompressionGatherMode, gatherMode);
    return false;
  }
  if (mRec->IsGPU() && (GetProcessingSettings().clusterizerZSSanityCheck || GetProcessingSettings().mergerSanityCheck)) {
    GPUError("Clusterizer and merger Sanity checks only supported when not running on GPU");
    return false;
  }
  if (GetProcessingSettings().tpcWriteClustersAfterRejection && (mRec->IsGPU() || param().rec.tpc.compressionTypeMask || !(GetRecoSteps() & gpudatatypes::RecoStep::TPCCompression))) {
    GPUError("tpcWriteClustersAfterRejection requires compressionTypeMask = 0, no GPU usage, and compression enabled");
    return false;
  }
  if (GetProcessingSettings().doublePipeline) {
    if (GetProcessingSettings().tpcFreeAllocatedMemoryAfterProcessing) {
      GPUError("Cannot use double pipeline with tpcFreeAllocatedMemoryAfterProcessing");
      return false;
    }
    if (!GetRecoStepsOutputs().isOnlySet(gpudatatypes::InOutType::TPCMergedTracks, gpudatatypes::InOutType::TPCCompressedClusters, gpudatatypes::InOutType::TPCClusters)) {
      GPUError("Invalid outputs for double pipeline mode 0x%x", (uint32_t)GetRecoStepsOutputs());
      return false;
    }
    if (((GetRecoStepsOutputs().isSet(gpudatatypes::InOutType::TPCCompressedClusters) && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::compressedClusters)] == nullptr) ||
         (GetRecoStepsOutputs().isSet(gpudatatypes::InOutType::TPCClusters) && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)] == nullptr) ||
         (GetRecoStepsOutputs().isSet(gpudatatypes::InOutType::TPCMergedTracks) && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracks)] == nullptr) ||
         (GetProcessingSettings().outputSharedClusterMap && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::sharedClusterMap)] == nullptr))) {
      GPUError("Must use external output for double pipeline mode");
      return false;
    }
    if (gatherMode == 1) {
      GPUError("Double pipeline incompatible to compression mode 1");
      return false;
    }
    if (!(GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression) || !(GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCClusterFinding) || param().rec.fwdTPCDigitsAsClusters) {
      GPUError("Invalid reconstruction settings for double pipeline: Needs compression and cluster finding");
      return false;
    }
  }
  if ((GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression) && !(GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression) && (gatherMode == 1 || gatherMode == 3)) {
    GPUError("Invalid tpcCompressionGatherMode for compression on CPU");
    return false;
  }
  if (GetProcessingSettings().tpcApplyClusterFilterOnCPU > 0 && (GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCClusterFinding || GetProcessingSettings().runMC)) {
    GPUError("tpcApplyClusterFilterOnCPU cannot be used with GPU clusterization or with MC labels");
    return false;
  }
  if (GetRecoSteps() & RecoStep::TRDTracking) {
    if (GetProcessingSettings().trdTrackModelO2 && (GetProcessingSettings().createO2Output == 0 || (GetMatLUT() == nullptr && !GetProcessingSettings().willProvideO2PropagatorLate))) {
      GPUError("TRD tracking can only run on O2 TPC tracks if createO2Output is enabled (%d), and matBudLUT is available (0x%p)", (int32_t)GetProcessingSettings().createO2Output, (void*)GetMatLUT());
      return false;
    }
    if ((GetRecoStepsGPU() & RecoStep::TRDTracking) && !GetProcessingSettings().trdTrackModelO2 && GetProcessingSettings().createO2Output > 1) {
      GPUError("TRD tracking can only run on GPU TPC tracks if the createO2Output setting does not suppress them");
      return false;
    }
    if ((((GetRecoStepsGPU() & RecoStep::TRDTracking) && GetProcessingSettings().trdTrackModelO2) || ((GetRecoStepsGPU() & RecoStep::Refit) && !param().rec.trackingRefitGPUModel)) && (!GetProcessingSettings().o2PropagatorUseGPUField || (GetMatLUT() == nullptr && !GetProcessingSettings().willProvideO2PropagatorLate))) {
      GPUError("Cannot use TRD tracking or Refit on GPU without GPU polynomial field map (%d) or matlut table (%p)", (int32_t)GetProcessingSettings().o2PropagatorUseGPUField, (void*)GetMatLUT());
      return false;
    }
  }
  return true;
}

int32_t GPUChainTracking::Init()
{
  const auto& threadContext = GetThreadContext();
  if (GetProcessingSettings().debugLevel >= 1) {
    printf("Enabled Reconstruction Steps: 0x%x (on GPU: 0x%x)", (int32_t)GetRecoSteps().get(), (int32_t)GetRecoStepsGPU().get());
    for (uint32_t i = 0; i < sizeof(gpudatatypes::RECO_STEP_NAMES) / sizeof(gpudatatypes::RECO_STEP_NAMES[0]); i++) {
      if (GetRecoSteps().isSet(1u << i)) {
        printf(" - %s", gpudatatypes::RECO_STEP_NAMES[i]);
        if (GetRecoStepsGPU().isSet(1u << i)) {
          printf(" (G)");
        }
      }
    }
    printf("\n");
  }
  if (!ValidateSteps()) {
    return 1;
  }

  for (uint32_t i = 0; i < mSubOutputControls.size(); i++) {
    if (mSubOutputControls[i] == nullptr) {
      mSubOutputControls[i] = &mRec->OutputControl();
    }
  }

  if (!ValidateSettings()) {
    return 1;
  }

  if (GPUQA::QAAvailable() && (GetProcessingSettings().runQA || GetProcessingSettings().eventDisplay)) {
    auto& qa = mQAFromForeignChain ? mQAFromForeignChain->mQA : mQA;
    if (!qa) {
      qa.reset(new GPUQA(this));
    }
  }
  if (GetProcessingSettings().eventDisplay) {
    mEventDisplay.reset(GPUDisplayInterface::getDisplay(GetProcessingSettings().eventDisplay, this, GetQA()));
    if (mEventDisplay == nullptr) {
      throw std::runtime_error("Error loading event display");
    }
  }

  processors()->errorCodes.setMemory(mInputsHost->mErrorCodes);
  processors()->errorCodes.clear();

  if (mRec->IsGPU()) {
    UpdateGPUCalibObjects(-1);
    UpdateGPUCalibObjectsPtrs(-1); // First initialization, for users not using RunChain
    processorsShadow()->errorCodes.setMemory(mInputsShadow->mErrorCodes);
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->errorCodes - (char*)processors(), &processorsShadow()->errorCodes, sizeof(processorsShadow()->errorCodes), -1);
    TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mInputsHost->mResourceErrorCodes);
  }

  if (GetProcessingSettings().debugLevel >= 6) {
    std::string filename = std::string(mRec->IsGPU() ? "GPU" : "CPU") + (mRec->slaveId() != -1 ? (std::string("_slave") + std::to_string(mRec->slaveId())) : std::string(mRec->slavesExist() ? "_master" : "")) + GetProcessingSettings().debugLogSuffix + ".out";
    mDebugFile->open(filename.c_str());
    if (GetProcessingSettings().debugFileHexFloat >= 1 || (GetProcessingSettings().debugFileHexFloat == -1 && GetProcessingSettings().deterministicGPUReconstruction)) {
      *mDebugFile << std::hexfloat;
    }
  }

  return 0;
}

void GPUChainTracking::UpdateGPUCalibObjects(int32_t stream, const GPUCalibObjectsConst* ptrMask)
{
  // TODO: Is this correct?
  if (processors()->calibObjects.fastTransform && (ptrMask == nullptr || ptrMask->fastTransform)) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.fastTransform, (const void*)processors()->calibObjects.fastTransform, processors()->calibObjects.fastTransform->size());
  }

  if (processors()->calibObjects.dEdxCalibContainer && (ptrMask == nullptr || ptrMask->dEdxCalibContainer)) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.dEdxCalibContainer, (const void*)processors()->calibObjects.dEdxCalibContainer, sizeof(*processors()->calibObjects.dEdxCalibContainer));
    memcpy((void*)mFlatObjectsShadow.mdEdxSplinesBuffer, (const void*)processors()->calibObjects.dEdxCalibContainer->getFlatBufferPtr(), processors()->calibObjects.dEdxCalibContainer->getFlatBufferSize());
    mFlatObjectsShadow.mCalibObjects.dEdxCalibContainer->clearInternalBufferPtr();
    mFlatObjectsShadow.mCalibObjects.dEdxCalibContainer->setActualBufferAddress(mFlatObjectsShadow.mdEdxSplinesBuffer);
    mFlatObjectsShadow.mCalibObjects.dEdxCalibContainer->setFutureBufferAddress(mFlatObjectsDevice.mdEdxSplinesBuffer);
  }
  if (processors()->calibObjects.matLUT && (ptrMask == nullptr || ptrMask->matLUT)) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.matLUT, (const void*)processors()->calibObjects.matLUT, sizeof(*processors()->calibObjects.matLUT));
    memcpy((void*)mFlatObjectsShadow.mMatLUTBuffer, (const void*)processors()->calibObjects.matLUT->getFlatBufferPtr(), processors()->calibObjects.matLUT->getFlatBufferSize());
    mFlatObjectsShadow.mCalibObjects.matLUT->clearInternalBufferPtr();
    mFlatObjectsShadow.mCalibObjects.matLUT->setActualBufferAddress(mFlatObjectsShadow.mMatLUTBuffer);
    mFlatObjectsShadow.mCalibObjects.matLUT->setFutureBufferAddress(mFlatObjectsDevice.mMatLUTBuffer);
  }
  if (processors()->calibObjects.trdGeometry && (ptrMask == nullptr || ptrMask->trdGeometry)) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.trdGeometry, (const void*)processors()->calibObjects.trdGeometry, sizeof(*processors()->calibObjects.trdGeometry));
    mFlatObjectsShadow.mCalibObjects.trdGeometry->clearInternalBufferPtr();
  }
  if (processors()->calibObjects.trdRecoParam && (ptrMask == nullptr || ptrMask->trdRecoParam)) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.trdRecoParam, (const void*)processors()->calibObjects.trdRecoParam, sizeof(*processors()->calibObjects.trdRecoParam));
  }
  if (processors()->calibObjects.tpcPadGain && (ptrMask == nullptr || ptrMask->tpcPadGain)) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.tpcPadGain, (const void*)processors()->calibObjects.tpcPadGain, sizeof(*processors()->calibObjects.tpcPadGain));
  }
  if (processors()->calibObjects.tpcZSLinkMapping && (ptrMask == nullptr || ptrMask->tpcZSLinkMapping)) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.tpcZSLinkMapping, (const void*)processors()->calibObjects.tpcZSLinkMapping, sizeof(*processors()->calibObjects.tpcZSLinkMapping));
  }
  if (processors()->calibObjects.o2Propagator && (ptrMask == nullptr || ptrMask->o2Propagator)) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.o2Propagator, (const void*)processors()->calibObjects.o2Propagator, sizeof(*processors()->calibObjects.o2Propagator));
    mFlatObjectsShadow.mCalibObjects.o2Propagator->setGPUField(&processorsDevice()->param.polynomialField);
    mFlatObjectsShadow.mCalibObjects.o2Propagator->setMatLUT(mFlatObjectsShadow.mCalibObjects.matLUT);
  }
  TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mFlatObjectsShadow.mMemoryResFlat, stream);
  memcpy((void*)&processorsShadow()->calibObjects, (void*)&mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects));
}

void GPUChainTracking::UpdateGPUCalibObjectsPtrs(int32_t stream)
{
  WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->calibObjects - (char*)processors(), &mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects), stream);
}

int32_t GPUChainTracking::PrepareEvent()
{
  mRec->MemoryScalers()->nTRDTracklets = mIOPtrs.nTRDTracklets;
  if (mIOPtrs.clustersNative) {
    mRec->MemoryScalers()->nTPCHits = mIOPtrs.clustersNative->nClustersTotal;
  }
  if (mIOPtrs.tpcZS && param().rec.fwdTPCDigitsAsClusters) {
    throw std::runtime_error("Forwading zero-suppressed hits not supported");
  }
  ClearErrorCodes();
  return 0;
}

int32_t GPUChainTracking::ForceInitQA()
{
  auto& qa = mQAFromForeignChain ? mQAFromForeignChain->mQA : mQA;
  if (!qa) {
    qa.reset(new GPUQA(this));
  }
  if (!GetQA()->IsInitialized()) {
    return GetQA()->InitQA(GetProcessingSettings().runQA <= 0 ? -GetProcessingSettings().runQA : gpudatatypes::gpuqa::tasksAutomatic);
  }
  return 0;
}

int32_t GPUChainTracking::Finalize()
{
  if (GetProcessingSettings().runQA && GetQA()->IsInitialized() && !(mConfigQA && mConfigQA->shipToQC) && !mQAFromForeignChain) {
    GetQA()->UpdateChain(this);
    GetQA()->DrawQAHistograms();
  }
  if (GetProcessingSettings().debugLevel >= 6) {
    mDebugFile->close();
  }
  if (mCompressionStatistics) {
    mCompressionStatistics->Finish();
  }
  return 0;
}

void* GPUChainTracking::GPUTrackingFlatObjects::SetPointersFlatObjects(void* mem)
{
  char* fastTransformBase = (char*)mem;
  if (mChainTracking->processors()->calibObjects.fastTransform) {
    // TODO: Is this correct?!
    char* podBuf = nullptr;
    computePointerWithAlignment(mem, podBuf, mChainTracking->processors()->calibObjects.fastTransform->size()); // raw bytes
    mCalibObjects.fastTransform = reinterpret_cast<TPCFastTransformPOD*>(podBuf);
  }
  if ((char*)mem - fastTransformBase < mChainTracking->GetProcessingSettings().fastTransformObjectsMinMemorySize) {
    mem = fastTransformBase + mChainTracking->GetProcessingSettings().fastTransformObjectsMinMemorySize; // TODO: Fixme and do proper dynamic allocation
  }
  if (mChainTracking->processors()->calibObjects.tpcPadGain) {
    computePointerWithAlignment(mem, mCalibObjects.tpcPadGain, 1);
  }
  if (mChainTracking->processors()->calibObjects.tpcZSLinkMapping) {
    computePointerWithAlignment(mem, mCalibObjects.tpcZSLinkMapping, 1);
  }
  char* dummyPtr;
  if (mChainTracking->processors()->calibObjects.matLUT) {
    computePointerWithAlignment(mem, mCalibObjects.matLUT, 1);
    computePointerWithAlignment(mem, mMatLUTBuffer, mChainTracking->GetMatLUT()->getFlatBufferSize());
  } else if (mChainTracking->GetProcessingSettings().lateO2MatLutProvisioningSize) {
    computePointerWithAlignment(mem, dummyPtr, mChainTracking->GetProcessingSettings().lateO2MatLutProvisioningSize);
  }
  if (mChainTracking->processors()->calibObjects.dEdxCalibContainer) {
    computePointerWithAlignment(mem, mCalibObjects.dEdxCalibContainer, 1);
    computePointerWithAlignment(mem, mdEdxSplinesBuffer, mChainTracking->GetdEdxCalibContainer()->getFlatBufferSize());
  }
  if (mChainTracking->processors()->calibObjects.trdGeometry) {
    computePointerWithAlignment(mem, mCalibObjects.trdGeometry, 1);
  }
  if (mChainTracking->processors()->calibObjects.trdRecoParam) {
    computePointerWithAlignment(mem, mCalibObjects.trdRecoParam, 1);
  }
  computePointerWithAlignment(mem, mCalibObjects.o2Propagator, 1);
  if (!mChainTracking->processors()->calibObjects.o2Propagator) {
    mCalibObjects.o2Propagator = nullptr; // Always reserve memory for o2::Propagator, since it may be propagatred only during run() not during init().
  }
  if (!mChainTracking->mUpdateNewCalibObjects) {
    mem = (char*)mem + mChainTracking->GetProcessingSettings().calibObjectsExtraMemorySize; // TODO: Fixme and do proper dynamic allocation
  }
  return mem;
}

void GPUChainTracking::ClearIOPointers()
{
  std::memset((void*)&mIOPtrs, 0, sizeof(mIOPtrs));
  mIOMem.~InOutMemory();
  new (&mIOMem) InOutMemory;
  mClusterNativeAccessReduced.reset(nullptr);
  if (mClusterNativeAccess.get()) {
    memset((void*)mClusterNativeAccess.get(), 0, sizeof(*mClusterNativeAccess));
  }
}

void GPUChainTracking::AllocateIOMemory()
{
  for (uint32_t i = 0; i < NSECTORS; i++) {
    AllocateIOMemoryHelper(mIOPtrs.nClusterData[i], mIOPtrs.clusterData[i], mIOMem.clusterData[i]);
    AllocateIOMemoryHelper(mIOPtrs.nRawClusters[i], mIOPtrs.rawClusters[i], mIOMem.rawClusters[i]);
    AllocateIOMemoryHelper(mIOPtrs.nSectorTracks[i], mIOPtrs.sectorTracks[i], mIOMem.sectorTracks[i]);
    AllocateIOMemoryHelper(mIOPtrs.nSectorClusters[i], mIOPtrs.sectorClusters[i], mIOMem.sectorClusters[i]);
  }
  mIOMem.clusterNativeAccess.reset(new ClusterNativeAccess);
  std::memset(mIOMem.clusterNativeAccess.get(), 0, sizeof(ClusterNativeAccess)); // ClusterNativeAccess has no its own constructor
  AllocateIOMemoryHelper(mIOMem.clusterNativeAccess->nClustersTotal, mIOMem.clusterNativeAccess->clustersLinear, mIOMem.clustersNative);
  mIOPtrs.clustersNative = mIOMem.clusterNativeAccess->nClustersTotal ? mIOMem.clusterNativeAccess.get() : nullptr;
  AllocateIOMemoryHelper(mIOPtrs.nMCLabelsTPC, mIOPtrs.mcLabelsTPC, mIOMem.mcLabelsTPC);
  AllocateIOMemoryHelper(mIOPtrs.nMCInfosTPC, mIOPtrs.mcInfosTPC, mIOMem.mcInfosTPC);
  AllocateIOMemoryHelper(mIOPtrs.nMCInfosTPCCol, mIOPtrs.mcInfosTPCCol, mIOMem.mcInfosTPCCol);
  AllocateIOMemoryHelper(mIOPtrs.nMergedTracks, mIOPtrs.mergedTracks, mIOMem.mergedTracks);
  AllocateIOMemoryHelper(mIOPtrs.nMergedTrackHits, mIOPtrs.mergedTrackHits, mIOMem.mergedTrackHits);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracks, mIOPtrs.trdTracks, mIOMem.trdTracks);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdTracklets, mIOMem.trdTracklets);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdSpacePoints, mIOMem.trdSpacePoints);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTriggerRecords, mIOPtrs.trdTrigRecMask, mIOMem.trdTrigRecMask);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTriggerRecords, mIOPtrs.trdTriggerTimes, mIOMem.trdTriggerTimes);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTriggerRecords, mIOPtrs.trdTrackletIdxFirst, mIOMem.trdTrackletIdxFirst);
}

void GPUChainTracking::SetTPCFastTransform(aligned_unique_buffer_ptr<TPCFastTransformPOD>&& tpcFastTransform)
{
  mTPCFastTransformU = std::move(tpcFastTransform);
  processors()->calibObjects.fastTransform = (TPCFastTransformPOD*)mTPCFastTransformU.get();
}

void GPUChainTracking::SetMatLUT(std::unique_ptr<o2::base::MatLayerCylSet>&& lut)
{
  mMatLUTU = std::move(lut);
  processors()->calibObjects.matLUT = mMatLUTU.get();
}

void GPUChainTracking::SetTRDGeometry(std::unique_ptr<o2::trd::GeometryFlat>&& geo)
{
  mTRDGeometryU = std::move(geo);
  processors()->calibObjects.trdGeometry = mTRDGeometryU.get();
}

void GPUChainTracking::SetTRDRecoParam(std::unique_ptr<GPUTRDRecoParam>&& par)
{
  mTRDRecoParamU = std::move(par);
  processors()->calibObjects.trdRecoParam = mTRDRecoParamU.get();
}

int32_t GPUChainTracking::DoQueuedUpdates(int32_t stream, bool updateSlave)
{
  int32_t retVal = 0;
  std::unique_ptr<GPUSettingsGRP> grp;
  const GPUSettingsProcessing* p = nullptr;
  std::lock_guard lk(mMutexUpdateCalib);
  if (mUpdateNewCalibObjects) {
    if (mNewCalibValues->newSolenoidField || mNewCalibValues->newContinuousMaxTimeBin || mNewCalibValues->newTPCTimeBinCut) {
      grp = std::make_unique<GPUSettingsGRP>(mRec->GetGRPSettings());
      if (mNewCalibValues->newSolenoidField) {
        grp->solenoidBzNominalGPU = mNewCalibValues->solenoidField;
      }
      if (mNewCalibValues->newContinuousMaxTimeBin) {
        grp->grpContinuousMaxTimeBin = mNewCalibValues->continuousMaxTimeBin;
      }
      if (mNewCalibValues->newTPCTimeBinCut) {
        grp->tpcCutTimeBin = mNewCalibValues->tpcTimeBinCut;
      }
    }
  }
  if (GetProcessingSettings().tpcDownscaledEdx != 0) {
    p = &GetProcessingSettings();
  }
  if (grp || p) {
    mRec->UpdateSettings(grp.get(), p);
    retVal = 1;
  }
  if (mUpdateNewCalibObjects) {
    if (mNewCalibObjects->o2Propagator && ((mNewCalibObjects->o2Propagator->getGPUField() != nullptr) ^ GetProcessingSettings().o2PropagatorUseGPUField)) {
      GPUFatal("GPU magnetic field for propagator requested, but received an O2 propagator without GPU field");
    }
    void* const* pSrc = (void* const*)mNewCalibObjects.get();
    void** pDst = (void**)&processors()->calibObjects;
    for (uint32_t i = 0; i < sizeof(processors()->calibObjects) / sizeof(void*); i++) {
      if (pSrc[i]) {
        pDst[i] = pSrc[i];
      }
    }
    if (mNewCalibObjects->trdGeometry && (GetRecoSteps() & gpudatatypes::RecoStep::TRDTracking)) {
      if (GetProcessingSettings().trdTrackModelO2) {
        processors()->trdTrackerO2.UpdateGeometry();
        if (mRec->IsGPU()) {
          TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, processors()->trdTrackerO2.MemoryPermanent(), stream);
        }
      } else {
        processors()->trdTrackerGPU.UpdateGeometry();
        if (mRec->IsGPU()) {
          TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, processors()->trdTrackerGPU.MemoryPermanent(), stream);
        }
      }
    }
    if (mRec->IsGPU()) {
      std::array<uint8_t, sizeof(GPUTrackingFlatObjects)> oldFlatPtrs, oldFlatPtrsDevice;
      memcpy(oldFlatPtrs.data(), (void*)&mFlatObjectsShadow, oldFlatPtrs.size());
      memcpy(oldFlatPtrsDevice.data(), (void*)&mFlatObjectsDevice, oldFlatPtrsDevice.size());
      mRec->ResetRegisteredMemoryPointers(mFlatObjectsShadow.mMemoryResFlat);
      bool ptrsChanged = memcmp(oldFlatPtrs.data(), (void*)&mFlatObjectsShadow, oldFlatPtrs.size()) || memcmp(oldFlatPtrsDevice.data(), (void*)&mFlatObjectsDevice, oldFlatPtrsDevice.size());
      if (ptrsChanged) {
        GPUInfo("Updating all calib objects since pointers changed");
      }
      UpdateGPUCalibObjects(stream, ptrsChanged ? nullptr : mNewCalibObjects.get());
    }
  }

  if ((mUpdateNewCalibObjects || (mRec->slavesExist() && updateSlave)) && mRec->IsGPU()) {
    UpdateGPUCalibObjectsPtrs(stream); // Reinitialize
    retVal = 1;
  }
  mNewCalibObjects.reset(nullptr);
  mNewCalibValues.reset(nullptr);
  mUpdateNewCalibObjects = false;
  return retVal;
}

int32_t GPUChainTracking::RunChain()
{
  if ((((GetRecoSteps() & RecoStep::TRDTracking) && !GetProcessingSettings().trdTrackModelO2 && !GetProcessingSettings().willProvideO2PropagatorLate) || ((GetRecoSteps() & RecoStep::Refit) && !param().rec.trackingRefitGPUModel)) && processors()->calibObjects.o2Propagator == nullptr) {
    GPUFatal("Cannot run TRD tracking or refit with o2 track model without o2 propagator"); // This check must happen during run, since o2::Propagator cannot be available during init
  }
  if (GetProcessingSettings().autoAdjustHostThreads && !mRec->IsGPU()) {
    mRec->SetNActiveThreads(-1);
  }
  const auto threadContext = GetThreadContext();
  if (GetProcessingSettings().runCompressionStatistics && mCompressionStatistics == nullptr) {
    mCompressionStatistics.reset(new GPUTPCClusterStatistics);
  }
  const bool needQA = GPUQA::QAAvailable() && (GetProcessingSettings().runQA || (GetProcessingSettings().eventDisplay && (mIOPtrs.nMCInfosTPC || GetProcessingSettings().runMC)));
  if (needQA && GetQA()->IsInitialized() == false) {
    if (GetQA()->InitQA(GetProcessingSettings().runQA <= 0 ? -GetProcessingSettings().runQA : gpudatatypes::gpuqa::tasksAutomatic)) {
      return 1;
    }
  }
  if (needQA) {
    mFractionalQAEnabled = GetProcessingSettings().qcRunFraction == 100.f || (uint32_t)(rand() % 10000) < (uint32_t)(GetProcessingSettings().qcRunFraction * 100);
  }
  if (GetProcessingSettings().debugLevel >= 6) {
    *mDebugFile << "\n\nProcessing event " << mRec->getNEventsProcessed() << std::endl;
  }
  DoQueuedUpdates(0);

  mRec->getGeneralStepTimer(GeneralStep::Prepare).Start();
  try {
    mRec->PrepareEvent();
  } catch (const std::bad_alloc& e) {
    GPUError("Memory Allocation Error");
    return (1);
  }
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Stop();

  PrepareKernelDebugOutput();

  SynchronizeStream(0); // Synchronize all init copies that might be ongoing

  if (GetProcessingSettings().debugOnFailure) {
    mRec->setDebugDumpCallback([this]() { DoDebugRawDump(); });
  }

  if (mIOPtrs.tpcCompressedClusters) {
    if (runRecoStep(RecoStep::TPCDecompression, &GPUChainTracking::RunTPCDecompression)) {
      return 1;
    }
  } else if (mIOPtrs.tpcPackedDigits || mIOPtrs.tpcZS) {
    if (runRecoStep(RecoStep::TPCClusterFinding, &GPUChainTracking::RunTPCClusterizer, false)) {
      return 1;
    }
  }

  if (GetProcessingSettings().autoAdjustHostThreads && !mRec->IsGPU() && mIOPtrs.clustersNative) {
    mRec->SetNActiveThreads(mIOPtrs.clustersNative->nClustersTotal / 1500);
  }

  if (mIOPtrs.clustersNative && runRecoStep(RecoStep::TPCConversion, &GPUChainTracking::ConvertNativeToClusterData)) {
    return 1;
  }

  mRec->PushNonPersistentMemory(qStr2Tag("TPCSLCD1")); // 1st stack level for TPC tracking sector data
  mTPCSectorScratchOnStack = true;
  if (runRecoStep(RecoStep::TPCSectorTracking, &GPUChainTracking::RunTPCTrackingSectors)) {
    return 1;
  }

  if (runRecoStep(RecoStep::TPCMerging, &GPUChainTracking::RunTPCTrackingMerger, false)) {
    return 1;
  }
  if (mTPCSectorScratchOnStack) {
    mRec->PopNonPersistentMemory(RecoStep::TPCSectorTracking, qStr2Tag("TPCSLCD1")); // Release 1st stack level, TPC sector data not needed after merger
    mTPCSectorScratchOnStack = false;
  }

  if (mIOPtrs.clustersNative) {
    if (GetProcessingSettings().doublePipeline) {
      GPUChainTracking* foreignChain = (GPUChainTracking*)GetNextChainInQueue();
      if (foreignChain && foreignChain->mIOPtrs.tpcZS) {
        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Preempting tpcZS input of foreign chain");
        }
        mPipelineFinalizationCtx.reset(new GPUChainTrackingFinalContext);
        mPipelineFinalizationCtx->rec = this->mRec;
        foreignChain->mPipelineNotifyCtx = mPipelineFinalizationCtx.get();
      }
    }
    if (runRecoStep(RecoStep::TPCCompression, &GPUChainTracking::RunTPCCompression)) {
      return 1;
    }
  }

  if (runRecoStep(RecoStep::TRDTracking, &GPUChainTracking::RunTRDTracking)) {
    return 1;
  }

  if (runRecoStep(RecoStep::Refit, &GPUChainTracking::RunRefit)) {
    return 1;
  }

  if (!GetProcessingSettings().doublePipeline) { // Synchronize with output copies running asynchronously
    SynchronizeStream(OutputStream());
  }

  if (GetProcessingSettings().autoAdjustHostThreads && !mRec->IsGPU()) {
    mRec->SetNActiveThreads(-1);
  }

  int32_t retVal = 0;
  if (CheckErrorCodes(false, false, mRec->getErrorCodeOutput())) { // TODO: Eventually, we should use GPUReconstruction::CheckErrorCodes
    retVal = 3;
    if (!GetProcessingSettings().ignoreNonFatalGPUErrors) {
      return retVal;
    }
  }

  if (GetProcessingSettings().doublePipeline) {
    return retVal;
  }
  int32_t retVal2 = RunChainFinalize();
  return retVal2 ? retVal2 : retVal;
}

int32_t GPUChainTracking::RunChainFinalize()
{
  if (mIOPtrs.clustersNative && (GetRecoSteps() & RecoStep::TPCCompression) && GetProcessingSettings().runCompressionStatistics) {
    CompressedClusters c = *mIOPtrs.tpcCompressedClusters;
    mCompressionStatistics->RunStatistics(mIOPtrs.clustersNative, &c, param());
  }

  if (GetProcessingSettings().outputSanityCheck) {
    OutputSanityCheck();
  }

  const bool needQA = GPUQA::QAAvailable() && (GetProcessingSettings().runQA || (GetProcessingSettings().eventDisplay && mIOPtrs.nMCInfosTPC));
  if (needQA && mFractionalQAEnabled) {
    mRec->getGeneralStepTimer(GeneralStep::QA).Start();
    GetQA()->UpdateChain(this);
    GetQA()->RunQA(!GetProcessingSettings().runQA);
    mRec->getGeneralStepTimer(GeneralStep::QA).Stop();
    if (GetProcessingSettings().debugLevel == 0) {
      GPUInfo("Total QA runtime: %d us", (int32_t)(mRec->getGeneralStepTimer(GeneralStep::QA).GetElapsedTime() * 1000000));
    }
  }

  if (GetProcessingSettings().showOutputStat) {
    PrintOutputStat();
  }

  PrintKernelDebugOutput();

  // PrintMemoryRelations();

  if (GetProcessingSettings().eventDisplay) {
    if (!mDisplayRunning) {
      GPUInfo("Starting Event Display...");
      if (mEventDisplay->StartDisplay()) {
        GPUError("Error starting Event Display");
        return (1);
      }
      mDisplayRunning = true;
    } else {
      mEventDisplay->ShowNextEvent();
    }

    mEventDisplay->WaitTillEventShown();

    if (GetProcessingSettings().eventDisplay->EnableSendKey()) {
      while (kbhit()) {
        getch();
      }
      GPUInfo("Press key for next event!");
    }

    int32_t iKey;
    do {
      usleep(10000);
      if (GetProcessingSettings().eventDisplay->EnableSendKey()) {
        iKey = kbhit() ? getch() : 0;
        if (iKey == 27) {
          GetProcessingSettings().eventDisplay->setDisplayControl(2);
        } else if (iKey == 'n') {
          break;
        } else if (iKey) {
          while (GetProcessingSettings().eventDisplay->getSendKey() != 0) {
            usleep(1000);
          }
          GetProcessingSettings().eventDisplay->setSendKey(iKey);
        }
      }
    } while (GetProcessingSettings().eventDisplay->getDisplayControl() == 0);
    if (GetProcessingSettings().eventDisplay->getDisplayControl() == 2) {
      mDisplayRunning = false;
      GetProcessingSettings().eventDisplay->DisplayExit();
      const_cast<GPUSettingsProcessing&>(GetProcessingSettings()).eventDisplay = nullptr; // TODO: fixme - eventDisplay should probably not be put into ProcessingSettings in the first place
      return (2);
    }
    GetProcessingSettings().eventDisplay->setDisplayControl(0);
    GPUInfo("Loading next event...");

    mEventDisplay->BlockTillNextEvent();
  }

  return 0;
}

int32_t GPUChainTracking::FinalizePipelinedProcessing()
{
  if (mPipelineFinalizationCtx) {
    {
      std::unique_lock<std::mutex> lock(mPipelineFinalizationCtx->mutex);
      auto* ctx = mPipelineFinalizationCtx.get();
      mPipelineFinalizationCtx->cond.wait(lock, [ctx]() { return ctx->ready; });
    }
    mPipelineFinalizationCtx.reset();
  }
  return RunChainFinalize();
}

int32_t GPUChainTracking::CheckErrorCodes(bool cpuOnly, bool forceShowErrors, std::vector<std::array<uint32_t, 4>>* fillErrors)
{
  int32_t retVal = 0;
  bool hasDebugError = false;
  for (int32_t i = 0; i < 1 + (!cpuOnly && mRec->IsGPU()); i++) {
    if (i) {
      const auto& threadContext = GetThreadContext();
      if (GetProcessingSettings().doublePipeline) {
        TransferMemoryResourceLinkToHost(RecoStep::NoRecoStep, mInputsHost->mResourceErrorCodes, 0);
        SynchronizeStream(0);
      } else {
        TransferMemoryResourceLinkToHost(RecoStep::NoRecoStep, mInputsHost->mResourceErrorCodes);
      }
    }
    if (processors()->errorCodes.hasError()) {
      static int32_t errorsShown = 0;
      static bool quiet = false;
      static std::chrono::time_point<std::chrono::steady_clock> silenceFrom;
      if (!quiet && errorsShown++ >= 10 && GetProcessingSettings().throttleAlarms && !forceShowErrors) {
        silenceFrom = std::chrono::steady_clock::now();
        quiet = true;
      } else if (quiet) {
        auto currentTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = currentTime - silenceFrom;
        if (elapsed_seconds.count() > 60 * 10) {
          quiet = false;
          errorsShown = 1;
        }
      }
      retVal = 1;
      if (GetProcessingSettings().throttleAlarms && !forceShowErrors) {
        GPUWarning("GPUReconstruction suffered from an error in the %s part", i ? "GPU" : "CPU");
      } else {
        GPUError("GPUReconstruction suffered from an error in the %s part", i ? "GPU" : "CPU");
      }
      if (!quiet) {
        processors()->errorCodes.printErrors(GetProcessingSettings().throttleAlarms && !forceShowErrors);
      }
      if (fillErrors) {
        uint32_t nErrors = processors()->errorCodes.getNErrors();
        const uint32_t* pErrors = processors()->errorCodes.getErrorPtr();
        for (uint32_t j = 0; j < nErrors; j++) {
          fillErrors->emplace_back(std::array<uint32_t, 4>{pErrors[4 * j], pErrors[4 * j + 1], pErrors[4 * j + 2], pErrors[4 * j + 3]});
        }
      }
      if ((GetProcessingSettings().debugOnFailure & 1) || (GetProcessingSettings().debugOnFailure & 4)) {
        if (GetProcessingSettings().debugOnFailureErrorMask == (uint64_t)-1) {
          hasDebugError = true;
        } else {
          uint32_t nErrors = processors()->errorCodes.getNErrors();
          const uint32_t* pErrors = processors()->errorCodes.getErrorPtr();
          for (uint32_t j = 0; j < nErrors; j++) {
            if (GetProcessingSettings().debugOnFailureErrorMask & (1 << pErrors[4 * j])) {
              hasDebugError = true;
              break;
            }
          }
        }
      }
    }
  }
  ClearErrorCodes(cpuOnly);
  if (hasDebugError) {
    mRec->triggerDebugDump();
  }
  return retVal;
}

void GPUChainTracking::ClearErrorCodes(bool cpuOnly)
{
  processors()->errorCodes.clear();
  if (mRec->IsGPU() && !cpuOnly) {
    const auto& threadContext = GetThreadContext();
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->errorCodes - (char*)processors(), &processorsShadow()->errorCodes, sizeof(processorsShadow()->errorCodes), 0);
    TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mInputsHost->mResourceErrorCodes, 0);
  }
}

void GPUChainTracking::SetUpdateCalibObjects(const GPUCalibObjectsConst& obj, const GPUNewCalibValues& vals)
{
  std::lock_guard lk(mMutexUpdateCalib);
  if (mNewCalibObjects) {
    void* const* pSrc = (void* const*)&obj;
    void** pDst = (void**)mNewCalibObjects.get();
    for (uint32_t i = 0; i < sizeof(*mNewCalibObjects) / sizeof(void*); i++) {
      if (pSrc[i]) {
        pDst[i] = pSrc[i];
      }
    }
  } else {
    mNewCalibObjects.reset(new GPUCalibObjectsConst(obj));
  }
  if (mNewCalibValues) {
    mNewCalibValues->updateFrom(&vals);
  } else {
    mNewCalibValues.reset(new GPUNewCalibValues(vals));
  }
  mUpdateNewCalibObjects = true;
}

const o2::base::Propagator* GPUChainTracking::GetDeviceO2Propagator()
{
  return (mRec->IsGPU() ? processorsShadow() : processors())->calibObjects.o2Propagator;
}

void GPUChainTracking::SetO2Propagator(const o2::base::Propagator* prop)
{
  processors()->calibObjects.o2Propagator = prop;
  if ((prop->getGPUField() != nullptr) ^ GetProcessingSettings().o2PropagatorUseGPUField) {
    GPUFatal("GPU magnetic field for propagator requested, but received an O2 propagator without GPU field");
  }
}

void GPUChainTracking::ApplySyncSettings(GPUSettingsProcessing& proc, GPUSettingsRec& rec, gpudatatypes::RecoStepField& steps, bool syncMode, int32_t dEdxMode)
{
  if (syncMode) {
    rec.useMatLUT = false;
  }
  if (proc.rtc.optSpecialCode == -1) {
    proc.rtc.optSpecialCode = syncMode;
  }
  if (dEdxMode != -2) {
    steps.setBits(gpudatatypes::RecoStep::TPCdEdx, dEdxMode == -1 ? !syncMode : (dEdxMode > 0));
  }
}
