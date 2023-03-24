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

#ifdef GPUCA_HAVE_O2HEADERS
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif
#include <fstream>
#include <chrono>

#include "GPUChainTracking.h"
#include "GPUChainTrackingDefs.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCSliceOutCluster.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTPCTrack.h"
#include "GPUTPCHitId.h"
#include "TPCZSLinkMapping.h"
#include "GPUTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUTRDTrackletLabels.h"
#include "display/GPUDisplayInterface.h"
#include "GPUQA.h"
#include "GPULogging.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"
#include "GPUNewCalibValues.h"

#ifdef GPUCA_HAVE_O2HEADERS
#include "GPUTPCClusterStatistics.h"
#include "GPUHostDataTypes.h"
#include "GPUTPCCFChainContext.h"
#include "GPUTrackingRefit.h"
#include "CalibdEdxContainer.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#include "TPCFastTransform.h"
#include "CorrectionMapsHelper.h"

#include "utils/linux_helpers.h"
#include "utils/strtag.h"
using namespace GPUCA_NAMESPACE::gpu;

#include "GPUO2DataTypes.h"

using namespace o2::tpc;
using namespace o2::trd;

GPUChainTracking::GPUChainTracking(GPUReconstruction* rec, unsigned int maxTPCHits, unsigned int maxTRDTracklets) : GPUChain(rec), mIOPtrs(processors()->ioPtrs), mInputsHost(new GPUTrackingInputProvider), mInputsShadow(new GPUTrackingInputProvider), mClusterNativeAccess(new ClusterNativeAccess), mMaxTPCHits(maxTPCHits), mMaxTRDTracklets(maxTRDTracklets), mDebugFile(new std::ofstream)
{
  ClearIOPointers();
  mFlatObjectsShadow.mChainTracking = this;
  mFlatObjectsDevice.mChainTracking = this;
}

GPUChainTracking::~GPUChainTracking() = default;

void GPUChainTracking::RegisterPermanentMemoryAndProcessors()
{
  mFlatObjectsShadow.InitGPUProcessor(mRec, GPUProcessor::PROCESSOR_TYPE_SLAVE);
  mFlatObjectsDevice.InitGPUProcessor(mRec, GPUProcessor::PROCESSOR_TYPE_DEVICE, &mFlatObjectsShadow);
  mFlatObjectsShadow.mMemoryResFlat = mRec->RegisterMemoryAllocation(&mFlatObjectsShadow, &GPUTrackingFlatObjects::SetPointersFlatObjects, GPUMemoryResource::MEMORY_PERMANENT, "CalibObjects");

  mRec->RegisterGPUProcessor(mInputsHost.get(), mRec->IsGPU());
  if (GetRecoSteps() & RecoStep::TPCSliceTracking) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      mRec->RegisterGPUProcessor(&processors()->tpcTrackers[i], GetRecoStepsGPU() & RecoStep::TPCSliceTracking);
    }
  }
  if (GetRecoSteps() & RecoStep::TPCMerging) {
    mRec->RegisterGPUProcessor(&processors()->tpcMerger, GetRecoStepsGPU() & RecoStep::TPCMerging);
  }
  if (GetRecoSteps() & RecoStep::TRDTracking) {
    mRec->RegisterGPUProcessor(&processors()->trdTrackerGPU, GetRecoStepsGPU() & RecoStep::TRDTracking);
  }
#ifdef GPUCA_HAVE_O2HEADERS
  if (GetRecoSteps() & RecoStep::TRDTracking) {
    mRec->RegisterGPUProcessor(&processors()->trdTrackerO2, GetRecoStepsGPU() & RecoStep::TRDTracking);
  }
  if (GetRecoSteps() & RecoStep::TPCConversion) {
    mRec->RegisterGPUProcessor(&processors()->tpcConverter, GetRecoStepsGPU() & RecoStep::TPCConversion);
  }
  if (GetRecoSteps() & RecoStep::TPCCompression) {
    mRec->RegisterGPUProcessor(&processors()->tpcCompressor, GetRecoStepsGPU() & RecoStep::TPCCompression);
  }
  if (GetRecoSteps() & RecoStep::TPCClusterFinding) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      mRec->RegisterGPUProcessor(&processors()->tpcClusterer[i], GetRecoStepsGPU() & RecoStep::TPCClusterFinding);
    }
  }
  if (GetRecoSteps() & RecoStep::Refit) {
    mRec->RegisterGPUProcessor(&processors()->trackingRefit, GetRecoStepsGPU() & RecoStep::Refit);
  }
#endif
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
  if (GetRecoStepsGPU() & RecoStep::TPCSliceTracking) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcTrackers[i], &processors()->tpcTrackers[i]);
    }
  }
  if (GetRecoStepsGPU() & RecoStep::TPCMerging) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcMerger, &processors()->tpcMerger);
  }
  if (GetRecoStepsGPU() & RecoStep::TRDTracking) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->trdTrackerGPU, &processors()->trdTrackerGPU);
  }

#ifdef GPUCA_HAVE_O2HEADERS
  memcpy((void*)&processorsShadow()->trdTrackerO2, (const void*)&processors()->trdTrackerO2, sizeof(processors()->trdTrackerO2));
  if (GetRecoStepsGPU() & RecoStep::TRDTracking) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->trdTrackerO2, &processors()->trdTrackerO2);
  }
  if (GetRecoStepsGPU() & RecoStep::TPCConversion) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcConverter, &processors()->tpcConverter);
  }
  if (GetRecoStepsGPU() & RecoStep::TPCCompression) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcCompressor, &processors()->tpcCompressor);
  }
  if (GetRecoStepsGPU() & RecoStep::TPCClusterFinding) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcClusterer[i], &processors()->tpcClusterer[i]);
    }
  }
  if (GetRecoStepsGPU() & RecoStep::Refit) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->trackingRefit, &processors()->trackingRefit);
  }
#endif
#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
  mRec->RegisterGPUDeviceProcessor(&processorsShadow()->debugOutput, &processors()->debugOutput);
#endif
}

void GPUChainTracking::MemorySize(size_t& gpuMem, size_t& pageLockedHostMem)
{
  gpuMem = GPUCA_MEMORY_SIZE;
  pageLockedHostMem = GPUCA_HOST_MEMORY_SIZE;
}

bool GPUChainTracking::ValidateSteps()
{
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCdEdx) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging)) {
    GPUError("Invalid Reconstruction Step Setting: dEdx requires TPC Merger to be active");
    return false;
  }
  if ((GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCdEdx) && !(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging)) {
    GPUError("Invalid GPU Reconstruction Step Setting: dEdx requires TPC Merger to be active");
    return false;
  }
  if (!param().par.earlyTpcTransform) {
    if (((GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging)) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCConversion)) {
      GPUError("Invalid Reconstruction Step Setting: Tracking without early transform requires TPC Conversion to be active");
      return false;
    }
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) && !(GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCRaw)) {
    GPUError("Invalid input, TPC Clusterizer needs TPC raw input");
    return false;
  }
  if (param().rec.tpc.mergerReadFromTrackerDirectly && (GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging) && ((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) || !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCConversion))) {
    GPUError("Invalid input / output / step, mergerReadFromTrackerDirectly cannot read/store sectors tracks and needs TPC conversion");
    return false;
  }
  if (!GetProcessingSettings().fullMergerOnGPU && (param().rec.tpc.mergerReadFromTrackerDirectly || GetProcessingSettings().createO2Output) && (GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging)) {
    GPUError("createO2Output and mergerReadFromTrackerDirectly works only in combination with fullMergerOnGPU if the merger is to run on GPU");
    return false;
  }
  bool tpcClustersAvail = (GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCClusters) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCDecompression);
#ifndef GPUCA_ALIROOT_LIB
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging) && !tpcClustersAvail) {
    GPUError("Invalid Inputs for track merging, TPC Clusters required");
    return false;
  }
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
  if (GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) {
    GPUError("Can not run TPC GPU Cluster Finding with Run 2 Data");
    return false;
  }
#endif
  if (((GetRecoSteps() & GPUDataTypes::RecoStep::TPCConversion) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCCompression) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCdEdx)) && !tpcClustersAvail) {
    GPUError("Missing input for TPC Cluster conversion / sector tracking / compression / dEdx: TPC Clusters required");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging) && !((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking))) {
    GPUError("Input for TPC merger missing");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCCompression) && !((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCMergedTracks) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging))) {
    GPUError("Input for TPC compressor missing");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TRDTracking) && (!((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCMergedTracks) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging)) || !(GetRecoStepsInputs() & GPUDataTypes::InOutType::TRDTracklets))) {
    GPUError("Input for TRD Tracker missing");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCRaw) || (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TRDTracklets)) {
    GPUError("TPC Raw / TPC Clusters / TRD Tracklets cannot be output");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking)) {
    GPUError("No TPC Slice Tracker Output available");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCMergedTracks) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging)) {
    GPUError("No TPC Merged Track Output available");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCCompressedClusters) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCCompression)) {
    GPUError("No TPC Compression Output available");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TRDTracks) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TRDTracking)) {
    GPUError("No TRD Tracker Output available");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCdEdx) && (processors()->calibObjects.dEdxCalibContainer == nullptr)) {
    GPUError("Cannot run dE/dx without dE/dx calibration container object");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) && processors()->calibObjects.tpcPadGain == nullptr) {
    GPUError("Cannot run gain calibration without calibration object");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) && processors()->calibObjects.tpcZSLinkMapping == nullptr && mIOPtrs.tpcZS != nullptr) {
    GPUError("Cannot run TPC ZS Decoder without mapping object. (tpczslinkmapping.dump missing?)");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::Refit) && !param().rec.trackingRefitGPUModel && ((processors()->calibObjects.o2Propagator == nullptr && !ProcessingSettings().internalO2PropagatorGPUField) || processors()->calibObjects.matLUT == nullptr)) {
    GPUError("Cannot run refit with o2 track model without o2 propagator");
    return false;
  }
  return true;
}

bool GPUChainTracking::ValidateSettings()
{
  if ((param().rec.tpc.nWays & 1) == 0) {
    GPUError("nWay setting musst be odd number!");
    return false;
  }
  if (param().rec.tpc.mergerInterpolateErrors && param().rec.tpc.nWays == 1) {
    GPUError("Cannot do error interpolation with NWays = 1!");
    return false;
  }
  if ((param().rec.tpc.mergerReadFromTrackerDirectly || !param().par.earlyTpcTransform) && param().rec.nonConsecutiveIDs) {
    GPUError("incompatible settings for non consecutive ids");
    return false;
  }
  if (!param().rec.tpc.mergerReadFromTrackerDirectly && GetProcessingSettings().ompKernels) {
    GPUError("OMP Kernels require mergerReadFromTrackerDirectly");
    return false;
  }
  if (param().par.continuousMaxTimeBin > (int)GPUSettings::TPC_MAX_TF_TIME_BIN) {
    GPUError("configured max time bin exceeds 256 orbits");
    return false;
  }
  if ((GetRecoStepsGPU() & RecoStep::TPCClusterFinding) && std::max(GetProcessingSettings().nTPCClustererLanes + 1, GetProcessingSettings().nTPCClustererLanes * 2) + (GetProcessingSettings().doublePipeline ? 1 : 0) > mRec->NStreams()) {
    GPUError("NStreams (%d) must be > nTPCClustererLanes (%d)", mRec->NStreams(), (int)GetProcessingSettings().nTPCClustererLanes);
    return false;
  }
  if (GetProcessingSettings().noGPUMemoryRegistration && GetProcessingSettings().tpcCompressionGatherMode != 3) {
    GPUError("noGPUMemoryRegistration only possible with gather mode 3");
    return false;
  }
  if (GetProcessingSettings().doublePipeline) {
    if (!GetRecoStepsOutputs().isOnlySet(GPUDataTypes::InOutType::TPCMergedTracks, GPUDataTypes::InOutType::TPCCompressedClusters, GPUDataTypes::InOutType::TPCClusters)) {
      GPUError("Invalid outputs for double pipeline mode 0x%x", (unsigned int)GetRecoStepsOutputs());
      return false;
    }
    if (((GetRecoStepsOutputs().isSet(GPUDataTypes::InOutType::TPCCompressedClusters) && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::compressedClusters)] == nullptr) ||
         (GetRecoStepsOutputs().isSet(GPUDataTypes::InOutType::TPCClusters) && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)] == nullptr) ||
         (GetRecoStepsOutputs().isSet(GPUDataTypes::InOutType::TPCMergedTracks) && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTracks)] == nullptr) ||
         (GetProcessingSettings().outputSharedClusterMap && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::sharedClusterMap)] == nullptr))) {
      GPUError("Must use external output for double pipeline mode");
      return false;
    }
    if (ProcessingSettings().tpcCompressionGatherMode == 1) {
      GPUError("Double pipeline incompatible to compression mode 1");
      return false;
    }
    if (!(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression) || !(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding) || param().rec.fwdTPCDigitsAsClusters) {
      GPUError("Invalid reconstruction settings for double pipeline");
      return false;
    }
  }
  if ((GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression) && !(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression) && (ProcessingSettings().tpcCompressionGatherMode == 1 || ProcessingSettings().tpcCompressionGatherMode == 3)) {
    GPUError("Invalid tpcCompressionGatherMode for compression on CPU");
    return false;
  }
  if (GetRecoSteps() & RecoStep::TRDTracking) {
    if (GetProcessingSettings().trdTrackModelO2 && (GetProcessingSettings().createO2Output == 0 || param().rec.tpc.nWaysOuter == 0 || GetMatLUT() == nullptr)) {
      GPUError("TRD tracking can only run on O2 TPC tracks if createO2Output is enabled (%d), nWaysOuter is set (%d), and matBudLUT is available (0x%p)", (int)GetProcessingSettings().createO2Output, (int)param().rec.tpc.nWaysOuter, (void*)GetMatLUT());
      return false;
    }
    if ((GetRecoStepsGPU() & RecoStep::TRDTracking) && !GetProcessingSettings().trdTrackModelO2 && GetProcessingSettings().createO2Output > 1) {
      GPUError("TRD tracking can only run on GPU TPC tracks if the createO2Output setting does not suppress them");
      return false;
    }
    if ((GetRecoStepsGPU() & RecoStep::TRDTracking) && (param().rec.trd.useExternalO2DefaultPropagator || !GetProcessingSettings().internalO2PropagatorGPUField)) {
      GPUError("Cannot use TRD tracking on GPU with external default o2::Propagator or without GPU polynomial field map");
      return false;
    }
  }
  return true;
}

int GPUChainTracking::EarlyConfigure()
{
  if (GetProcessingSettings().useInternalO2Propagator) {
    SetDefaultInternalO2Propagator(GetProcessingSettings().internalO2PropagatorGPUField);
  }
  return 0;
}

int GPUChainTracking::Init()
{
  const auto& threadContext = GetThreadContext();
  if (GetProcessingSettings().debugLevel >= 1) {
    printf("Enabled Reconstruction Steps: 0x%x (on GPU: 0x%x)", (int)GetRecoSteps().get(), (int)GetRecoStepsGPU().get());
    for (unsigned int i = 0; i < sizeof(GPUDataTypes::RECO_STEP_NAMES) / sizeof(GPUDataTypes::RECO_STEP_NAMES[0]); i++) {
      if (GetRecoSteps().isSet(1u << i)) {
        printf(" - %s", GPUDataTypes::RECO_STEP_NAMES[i]);
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

  for (unsigned int i = 0; i < mSubOutputControls.size(); i++) {
    if (mSubOutputControls[i] == nullptr) {
      mSubOutputControls[i] = &mRec->OutputControl();
    }
  }

  if (!ValidateSettings()) {
    return 1;
  }

  if (GPUQA::QAAvailable() && (GetProcessingSettings().runQA || GetProcessingSettings().eventDisplay)) {
    mQA.reset(new GPUQA(this));
  }
  if (GetProcessingSettings().eventDisplay) {
#ifndef GPUCA_ALIROOT_LIB
    mEventDisplay.reset(GPUDisplayInterface::getDisplay(GetProcessingSettings().eventDisplay, this, mQA.get()));
#endif
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
    mDebugFile->open(mRec->IsGPU() ? "GPU.out" : "CPU.out");
  }

  return 0;
}

void GPUChainTracking::UpdateGPUCalibObjects(int stream)
{
  if (processors()->calibObjects.fastTransform) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.fastTransform, (const void*)processors()->calibObjects.fastTransform, sizeof(*processors()->calibObjects.fastTransform));
    memcpy((void*)mFlatObjectsShadow.mTpcTransformBuffer, (const void*)processors()->calibObjects.fastTransform->getFlatBufferPtr(), processors()->calibObjects.fastTransform->getFlatBufferSize());
    mFlatObjectsShadow.mCalibObjects.fastTransform->clearInternalBufferPtr();
    mFlatObjectsShadow.mCalibObjects.fastTransform->setActualBufferAddress(mFlatObjectsShadow.mTpcTransformBuffer);
    mFlatObjectsShadow.mCalibObjects.fastTransform->setFutureBufferAddress(mFlatObjectsDevice.mTpcTransformBuffer);
  }
  if (processors()->calibObjects.fastTransformRef) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.fastTransformRef, (const void*)processors()->calibObjects.fastTransformRef, sizeof(*processors()->calibObjects.fastTransformRef));
    memcpy((void*)mFlatObjectsShadow.mTpcTransformRefBuffer, (const void*)processors()->calibObjects.fastTransformRef->getFlatBufferPtr(), processors()->calibObjects.fastTransformRef->getFlatBufferSize());
    mFlatObjectsShadow.mCalibObjects.fastTransformRef->clearInternalBufferPtr();
    mFlatObjectsShadow.mCalibObjects.fastTransformRef->setActualBufferAddress(mFlatObjectsShadow.mTpcTransformRefBuffer);
    mFlatObjectsShadow.mCalibObjects.fastTransformRef->setFutureBufferAddress(mFlatObjectsDevice.mTpcTransformRefBuffer);
  }
  if (processors()->calibObjects.fastTransformHelper) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.fastTransformHelper, (const void*)processors()->calibObjects.fastTransformHelper, sizeof(*processors()->calibObjects.fastTransformHelper));
    mFlatObjectsShadow.mCalibObjects.fastTransformHelper->setCorrMap(mFlatObjectsShadow.mCalibObjects.fastTransform);
    mFlatObjectsShadow.mCalibObjects.fastTransformHelper->setCorrMapRef(mFlatObjectsShadow.mCalibObjects.fastTransformRef);
  }
#ifdef GPUCA_HAVE_O2HEADERS
  if (processors()->calibObjects.dEdxCalibContainer) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.dEdxCalibContainer, (const void*)processors()->calibObjects.dEdxCalibContainer, sizeof(*processors()->calibObjects.dEdxCalibContainer));
    memcpy((void*)mFlatObjectsShadow.mdEdxSplinesBuffer, (const void*)processors()->calibObjects.dEdxCalibContainer->getFlatBufferPtr(), processors()->calibObjects.dEdxCalibContainer->getFlatBufferSize());
    mFlatObjectsShadow.mCalibObjects.dEdxCalibContainer->clearInternalBufferPtr();
    mFlatObjectsShadow.mCalibObjects.dEdxCalibContainer->setActualBufferAddress(mFlatObjectsShadow.mdEdxSplinesBuffer);
    mFlatObjectsShadow.mCalibObjects.dEdxCalibContainer->setFutureBufferAddress(mFlatObjectsDevice.mdEdxSplinesBuffer);
  }
  if (processors()->calibObjects.matLUT) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.matLUT, (const void*)processors()->calibObjects.matLUT, sizeof(*processors()->calibObjects.matLUT));
    memcpy((void*)mFlatObjectsShadow.mMatLUTBuffer, (const void*)processors()->calibObjects.matLUT->getFlatBufferPtr(), processors()->calibObjects.matLUT->getFlatBufferSize());
    mFlatObjectsShadow.mCalibObjects.matLUT->clearInternalBufferPtr();
    mFlatObjectsShadow.mCalibObjects.matLUT->setActualBufferAddress(mFlatObjectsShadow.mMatLUTBuffer);
    mFlatObjectsShadow.mCalibObjects.matLUT->setFutureBufferAddress(mFlatObjectsDevice.mMatLUTBuffer);
  }
  if (processors()->calibObjects.trdGeometry) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.trdGeometry, (const void*)processors()->calibObjects.trdGeometry, sizeof(*processors()->calibObjects.trdGeometry));
    mFlatObjectsShadow.mCalibObjects.trdGeometry->clearInternalBufferPtr();
  }
  if (processors()->calibObjects.tpcPadGain) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.tpcPadGain, (const void*)processors()->calibObjects.tpcPadGain, sizeof(*processors()->calibObjects.tpcPadGain));
  }
  if (processors()->calibObjects.tpcZSLinkMapping) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.tpcZSLinkMapping, (const void*)processors()->calibObjects.tpcZSLinkMapping, sizeof(*processors()->calibObjects.tpcZSLinkMapping));
  }
  if (processors()->calibObjects.o2Propagator) {
    memcpy((void*)mFlatObjectsShadow.mCalibObjects.o2Propagator, (const void*)processors()->calibObjects.o2Propagator, sizeof(*processors()->calibObjects.o2Propagator));
    mFlatObjectsShadow.mCalibObjects.o2Propagator->setGPUField(&processorsDevice()->param.polynomialField);
    mFlatObjectsShadow.mCalibObjects.o2Propagator->setBz(param().polynomialField.GetNominalBz());
    mFlatObjectsShadow.mCalibObjects.o2Propagator->setMatLUT(mFlatObjectsShadow.mCalibObjects.matLUT);
  }
#endif
  TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mFlatObjectsShadow.mMemoryResFlat, stream);
  memcpy((void*)&processorsShadow()->calibObjects, (void*)&mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects));
}

void GPUChainTracking::UpdateGPUCalibObjectsPtrs(int stream)
{
  WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->calibObjects - (char*)processors(), &mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects), stream);
}

int GPUChainTracking::PrepareEvent()
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

int GPUChainTracking::ForceInitQA()
{
  if (!mQA) {
    mQA.reset(new GPUQA(this));
  }
  if (!mQA->IsInitialized()) {
    return mQA->InitQA();
  }
  return 0;
}

int GPUChainTracking::Finalize()
{
  if (GetProcessingSettings().runQA && mQA->IsInitialized() && !(mConfigQA && mConfigQA->shipToQC)) {
    mQA->DrawQAHistograms();
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
  if (mChainTracking->processors()->calibObjects.fastTransform) {
    computePointerWithAlignment(mem, mCalibObjects.fastTransform, 1);
    computePointerWithAlignment(mem, mTpcTransformBuffer, mChainTracking->processors()->calibObjects.fastTransform->getFlatBufferSize());
  }
  if (mChainTracking->processors()->calibObjects.fastTransformRef) {
    computePointerWithAlignment(mem, mCalibObjects.fastTransformRef, 1);
    computePointerWithAlignment(mem, mTpcTransformRefBuffer, mChainTracking->processors()->calibObjects.fastTransformRef->getFlatBufferSize());
  }
  if (mChainTracking->processors()->calibObjects.fastTransformHelper) {
    computePointerWithAlignment(mem, mCalibObjects.fastTransformHelper, 1);
  }
  if (mChainTracking->processors()->calibObjects.tpcPadGain) {
    computePointerWithAlignment(mem, mCalibObjects.tpcPadGain, 1);
  }
  if (mChainTracking->processors()->calibObjects.tpcZSLinkMapping) {
    computePointerWithAlignment(mem, mCalibObjects.tpcZSLinkMapping, 1);
  }
#ifdef GPUCA_HAVE_O2HEADERS
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
  if (mChainTracking->processors()->calibObjects.o2Propagator) {
    computePointerWithAlignment(mem, mCalibObjects.o2Propagator, 1);
  } else if (mChainTracking->GetProcessingSettings().internalO2PropagatorGPUField) {
    computePointerWithAlignment(mem, dummyPtr, sizeof(*mCalibObjects.o2Propagator));
  }
#endif
  if (!mChainTracking->mUpdateNewCalibObjects) {
    mem = (char*)mem + mChainTracking->GetProcessingSettings().calibObjectsExtraMemorySize;
  }
  return mem;
}

void GPUChainTracking::ClearIOPointers()
{
  std::memset((void*)&mIOPtrs, 0, sizeof(mIOPtrs));
  mIOMem.~InOutMemory();
  new (&mIOMem) InOutMemory;
}

void GPUChainTracking::AllocateIOMemory()
{
  for (unsigned int i = 0; i < NSLICES; i++) {
    AllocateIOMemoryHelper(mIOPtrs.nClusterData[i], mIOPtrs.clusterData[i], mIOMem.clusterData[i]);
    AllocateIOMemoryHelper(mIOPtrs.nRawClusters[i], mIOPtrs.rawClusters[i], mIOMem.rawClusters[i]);
    AllocateIOMemoryHelper(mIOPtrs.nSliceTracks[i], mIOPtrs.sliceTracks[i], mIOMem.sliceTracks[i]);
    AllocateIOMemoryHelper(mIOPtrs.nSliceClusters[i], mIOPtrs.sliceClusters[i], mIOMem.sliceClusters[i]);
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
  AllocateIOMemoryHelper(mIOPtrs.nMergedTrackHits, mIOPtrs.mergedTrackHitsXYZ, mIOMem.mergedTrackHitsXYZ);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracks, mIOPtrs.trdTracks, mIOMem.trdTracks);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdTracklets, mIOMem.trdTracklets);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdSpacePoints, mIOMem.trdSpacePoints);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTriggerRecords, mIOPtrs.trdTrigRecMask, mIOMem.trdTrigRecMask);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTriggerRecords, mIOPtrs.trdTriggerTimes, mIOMem.trdTriggerTimes);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTriggerRecords, mIOPtrs.trdTrackletIdxFirst, mIOMem.trdTrackletIdxFirst);
}

void GPUChainTracking::LoadClusterErrors() { param().LoadClusterErrors(); }

void GPUChainTracking::SetTPCFastTransform(std::unique_ptr<TPCFastTransform>&& tpcFastTransform, std::unique_ptr<CorrectionMapsHelper>&& tpcTransformHelper)
{
  mTPCFastTransformU = std::move(tpcFastTransform);
  mTPCFastTransformHelperU = std::move(tpcTransformHelper);
  processors()->calibObjects.fastTransform = mTPCFastTransformU.get();
  processors()->calibObjects.fastTransformHelper = mTPCFastTransformHelperU.get();
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

void GPUChainTracking::DoQueuedCalibUpdates(int stream)
{
  if (mUpdateNewCalibObjects) {
    void** pSrc = (void**)&mNewCalibObjects;
    void** pDst = (void**)&processors()->calibObjects;
    for (unsigned int i = 0; i < sizeof(mNewCalibObjects) / sizeof(void*); i++) {
      if (pSrc[i]) {
        pDst[i] = pSrc[i];
      }
    }
    if (mRec->IsGPU()) {
      mRec->ResetRegisteredMemoryPointers(mFlatObjectsShadow.mMemoryResFlat);
      UpdateGPUCalibObjects(stream);
    }
    if (mNewCalibValues->newSolenoidField || mNewCalibValues->newContinuousMaxTimeBin) {
      GPUSettingsGRP grp = mRec->GetGRPSettings();
      if (mNewCalibValues->newSolenoidField) {
        grp.solenoidBz = mNewCalibValues->solenoidField;
      }
      if (mNewCalibValues->newContinuousMaxTimeBin) {
        grp.continuousMaxTimeBin = mNewCalibValues->continuousMaxTimeBin;
      }
      mRec->UpdateGRPSettings(&grp);
    }
  }
  if ((mUpdateNewCalibObjects || mRec->slavesExist()) && mRec->IsGPU()) {
    UpdateGPUCalibObjectsPtrs(stream); // Reinitialize
  }
  mUpdateNewCalibObjects = false;
}

int GPUChainTracking::RunChain()
{
  if (GetProcessingSettings().ompAutoNThreads && !mRec->IsGPU()) {
    mRec->SetNOMPThreads(-1);
  }
  const auto threadContext = GetThreadContext();
  if (GetProcessingSettings().runCompressionStatistics && mCompressionStatistics == nullptr) {
    mCompressionStatistics.reset(new GPUTPCClusterStatistics);
  }
  const bool needQA = GPUQA::QAAvailable() && (GetProcessingSettings().runQA || (GetProcessingSettings().eventDisplay && (mIOPtrs.nMCInfosTPC || GetProcessingSettings().runMC)));
  if (needQA && mQA->IsInitialized() == false) {
    if (mQA->InitQA(GetProcessingSettings().runQA ? -GetProcessingSettings().runQA : -1)) {
      return 1;
    }
  }
  if (needQA) {
    mFractionalQAEnabled = GetProcessingSettings().qcRunFraction == 100.f || (unsigned int)(rand() % 10000) < (unsigned int)(GetProcessingSettings().qcRunFraction * 100);
  }
  if (GetProcessingSettings().debugLevel >= 6) {
    *mDebugFile << "\n\nProcessing event " << mRec->getNEventsProcessed() << std::endl;
  }
  DoQueuedCalibUpdates(0);

  mRec->getGeneralStepTimer(GeneralStep::Prepare).Start();
  try {
    mRec->PrepareEvent();
  } catch (const std::bad_alloc& e) {
    GPUError("Memory Allocation Error");
    return (1);
  }
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Stop();

  PrepareDebugOutput();

  SynchronizeStream(0); // Synchronize all init copies that might be ongoing

  if (mIOPtrs.tpcCompressedClusters) {
    if (runRecoStep(RecoStep::TPCDecompression, &GPUChainTracking::RunTPCDecompression)) {
      return 1;
    }
  } else if (mIOPtrs.tpcPackedDigits || mIOPtrs.tpcZS) {
    if (runRecoStep(RecoStep::TPCClusterFinding, &GPUChainTracking::RunTPCClusterizer, false)) {
      return 1;
    }
  }

  if (GetProcessingSettings().ompAutoNThreads && !mRec->IsGPU() && mIOPtrs.clustersNative) {
    mRec->SetNOMPThreads(mIOPtrs.clustersNative->nClustersTotal / 5000);
  }

  if (mIOPtrs.clustersNative && runRecoStep(RecoStep::TPCConversion, &GPUChainTracking::ConvertNativeToClusterData)) {
    return 1;
  }

  mRec->PushNonPersistentMemory(qStr2Tag("TPCSLCD1")); // 1st stack level for TPC tracking slice data
  mTPCSliceScratchOnStack = true;
  if (runRecoStep(RecoStep::TPCSliceTracking, &GPUChainTracking::RunTPCTrackingSlices)) {
    return 1;
  }

  for (unsigned int i = 0; i < NSLICES; i++) {
    // GPUInfo("slice %d clusters %d tracks %d", i, mClusterData[i].NumberOfClusters(), processors()->tpcTrackers[i].Output()->NTracks());
    processors()->tpcMerger.SetSliceData(i, param().rec.tpc.mergerReadFromTrackerDirectly ? nullptr : processors()->tpcTrackers[i].Output());
  }
  if (runRecoStep(RecoStep::TPCMerging, &GPUChainTracking::RunTPCTrackingMerger, false)) {
    return 1;
  }
  if (mTPCSliceScratchOnStack) {
    mRec->PopNonPersistentMemory(RecoStep::TPCSliceTracking, qStr2Tag("TPCSLCD1")); // Release 1st stack level, TPC slice data not needed after merger
    mTPCSliceScratchOnStack = false;
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

  if (GetProcessingSettings().trdTrackModelO2 ? runRecoStep(RecoStep::TRDTracking, &GPUChainTracking::RunTRDTracking<GPUTRDTrackerKernels::o2Version>) : runRecoStep(RecoStep::TRDTracking, &GPUChainTracking::RunTRDTracking<GPUTRDTrackerKernels::gpuVersion>)) {
    return 1;
  }

  if (runRecoStep(RecoStep::Refit, &GPUChainTracking::RunRefit)) {
    return 1;
  }

  if (!GetProcessingSettings().doublePipeline) { // Synchronize with output copies running asynchronously
    SynchronizeStream(mRec->NStreams() - 2);
  }

  if (GetProcessingSettings().ompAutoNThreads && !mRec->IsGPU()) {
    mRec->SetNOMPThreads(-1);
  }

  if (CheckErrorCodes()) {
    return 3;
  }

  return GetProcessingSettings().doublePipeline ? 0 : RunChainFinalize();
}

int GPUChainTracking::RunChainFinalize()
{
#ifdef GPUCA_HAVE_O2HEADERS
  if (mIOPtrs.clustersNative && (GetRecoSteps() & RecoStep::TPCCompression) && GetProcessingSettings().runCompressionStatistics) {
    CompressedClusters c = *mIOPtrs.tpcCompressedClusters;
    mCompressionStatistics->RunStatistics(mIOPtrs.clustersNative, &c, param());
  }
#endif

  if (GetProcessingSettings().outputSanityCheck) {
    SanityCheck();
  }

  const bool needQA = GPUQA::QAAvailable() && (GetProcessingSettings().runQA || (GetProcessingSettings().eventDisplay && mIOPtrs.nMCInfosTPC));
  if (needQA && mFractionalQAEnabled) {
    mRec->getGeneralStepTimer(GeneralStep::QA).Start();
    mQA->RunQA(!GetProcessingSettings().runQA);
    mRec->getGeneralStepTimer(GeneralStep::QA).Stop();
    if (GetProcessingSettings().debugLevel == 0) {
      GPUInfo("Total QA runtime: %d us", (int)(mRec->getGeneralStepTimer(GeneralStep::QA).GetElapsedTime() * 1000000));
    }
  }

  if (GetProcessingSettings().showOutputStat) {
    PrintOutputStat();
  }

  PrintDebugOutput();

  //PrintMemoryRelations();

  if (GetProcessingSettings().eventDisplay) {
    if (!mDisplayRunning) {
      if (mEventDisplay->StartDisplay()) {
        return (1);
      }
      mDisplayRunning = true;
    } else {
      mEventDisplay->ShowNextEvent();
    }

    if (GetProcessingSettings().eventDisplay->EnableSendKey()) {
      while (kbhit()) {
        getch();
      }
      GPUInfo("Press key for next event!");
    }

    int iKey;
    do {
      Sleep(10);
      if (GetProcessingSettings().eventDisplay->EnableSendKey()) {
        iKey = kbhit() ? getch() : 0;
        if (iKey == 'q') {
          GetProcessingSettings().eventDisplay->setDisplayControl(2);
        } else if (iKey == 'n') {
          break;
        } else if (iKey) {
          while (GetProcessingSettings().eventDisplay->getSendKey() != 0) {
            Sleep(1);
          }
          GetProcessingSettings().eventDisplay->setSendKey(iKey);
        }
      }
    } while (GetProcessingSettings().eventDisplay->getDisplayControl() == 0);
    if (GetProcessingSettings().eventDisplay->getDisplayControl() == 2) {
      mDisplayRunning = false;
      GetProcessingSettings().eventDisplay->DisplayExit();
      ProcessingSettings().eventDisplay = nullptr;
      return (2);
    }
    GetProcessingSettings().eventDisplay->setDisplayControl(0);
    GPUInfo("Loading next event");

    mEventDisplay->WaitForNextEvent();
  }

  return 0;
}

int GPUChainTracking::FinalizePipelinedProcessing()
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

int GPUChainTracking::HelperReadEvent(int iSlice, int threadId, GPUReconstructionHelpers::helperParam* par) { return ReadEvent(iSlice, threadId); }

int GPUChainTracking::HelperOutput(int iSlice, int threadId, GPUReconstructionHelpers::helperParam* par)
{
  if (param().rec.tpc.globalTracking) {
    unsigned int tmpSlice = GPUTPCGlobalTracking::GlobalTrackingSliceOrder(iSlice);
    unsigned int sliceLeft, sliceRight;
    GPUTPCGlobalTracking::GlobalTrackingSliceLeftRight(tmpSlice, sliceLeft, sliceRight);

    while (mSliceSelectorReady < (int)tmpSlice || mSliceSelectorReady < (int)sliceLeft || mSliceSelectorReady < (int)sliceRight) {
      if (par->reset) {
        return 1;
      }
    }
    GlobalTracking(tmpSlice, 0);
    WriteOutput(tmpSlice, 0);
  } else {
    while (mSliceSelectorReady < iSlice) {
      if (par->reset) {
        return 1;
      }
    }
    WriteOutput(iSlice, threadId);
  }
  return 0;
}

int GPUChainTracking::CheckErrorCodes(bool cpuOnly, bool forceShowErrors)
{
  int retVal = 0;
  for (int i = 0; i < 1 + (!cpuOnly && mRec->IsGPU()); i++) {
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
      static int errorsShown = 0;
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
    }
  }
  ClearErrorCodes(cpuOnly);
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

void GPUChainTracking::SetDefaultInternalO2Propagator(bool useGPUField)
{
#ifdef GPUCA_HAVE_O2HEADERS
  o2::base::Propagator* prop = param().GetDefaultO2Propagator(useGPUField);
  prop->setMatLUT(processors()->calibObjects.matLUT);
  SetO2Propagator(prop);
#endif
}

void GPUChainTracking::SetUpdateCalibObjects(const GPUCalibObjectsConst& obj, const GPUNewCalibValues& vals)
{
  mNewCalibObjects = obj;
  mNewCalibValues.reset(new GPUNewCalibValues(vals));
  mUpdateNewCalibObjects = true;
}
