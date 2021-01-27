// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTracking.cxx
/// \author David Rohr

#ifdef HAVE_O2HEADERS
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif
#include <fstream>

#include "GPUChainTracking.h"
#include "GPUChainTrackingDefs.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCSliceOutCluster.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTPCTrack.h"
#include "GPUTPCHitId.h"
#include "GPUTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUTRDTrackletLabels.h"
#include "GPUDisplay.h"
#include "GPUQA.h"
#include "GPULogging.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"

#ifdef HAVE_O2HEADERS
#include "GPUTPCClusterStatistics.h"
#include "GPUHostDataTypes.h"
#include "TPCdEdxCalibrationSplines.h"
#include "GPUTPCCFChainContext.h"
#include "GPUTrackingRefit.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#include "TPCFastTransform.h"

#include "utils/linux_helpers.h"
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
  mFlatObjectsShadow.mMemoryResFlat = mRec->RegisterMemoryAllocation(&mFlatObjectsShadow, &GPUTrackingFlatObjects::SetPointersFlatObjects, GPUMemoryResource::MEMORY_PERMANENT, "Processors");

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
    mRec->RegisterGPUProcessor(&processors()->trdTracker, GetRecoStepsGPU() & RecoStep::TRDTracking);
  }
#ifdef HAVE_O2HEADERS
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
  memcpy((void*)&processorsShadow()->trdTracker, (const void*)&processors()->trdTracker, sizeof(processors()->trdTracker));
  if (GetRecoStepsGPU() & RecoStep::TPCSliceTracking) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcTrackers[i], &processors()->tpcTrackers[i]);
    }
  }
  if (GetRecoStepsGPU() & RecoStep::TPCMerging) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcMerger, &processors()->tpcMerger);
  }
  if (GetRecoStepsGPU() & RecoStep::TRDTracking) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->trdTracker, &processors()->trdTracker);
  }

#ifdef HAVE_O2HEADERS
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
    if (((GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) || (GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging)) && !(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion)) {
      GPUError("Invalid GPU Reconstruction Step Setting: Tracking without early transform requires TPC Conversion to be active");
      return false;
    }
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) && !(GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCRaw)) {
    GPUError("Invalid input, TPC Clusterizer needs TPC raw input");
    return false;
  }
  if (param().rec.mergerReadFromTrackerDirectly && (GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging) && ((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) || !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCConversion))) {
    GPUError("Invalid input / output / step, mergerReadFromTrackerDirectly cannot read/store sectors tracks and needs TPC conversion");
    return false;
  }
  bool tpcClustersAvail = (GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCClusters) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCDecompression);
#ifndef GPUCA_ALIROOT_LIB
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging) && !tpcClustersAvail) {
    GPUError("Invalid Inputs, TPC Clusters required");
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
    GPUError("Invalid Inputs, TPC Clusters required");
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
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCdEdx) && processors()->calibObjects.dEdxSplines == nullptr) {
    GPUError("Cannot run dE/dx without calibration splines");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) && processors()->calibObjects.tpcPadGain == nullptr) {
    GPUError("Cannot run gain calibration without calibration object");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::Refit) && !param().rec.trackingRefitGPUModel && (processors()->calibObjects.o2Propagator == nullptr || processors()->calibObjects.matLUT == nullptr)) {
    GPUError("Cannot run refit with o2 track model without o2 propagator");
    return false;
  }
  return true;
}

bool GPUChainTracking::ValidateSettings()
{
  if ((param().rec.NWays & 1) == 0) {
    GPUError("nWay setting musst be odd number!");
    return false;
  }
  if (param().rec.mergerInterpolateErrors && param().rec.NWays == 1) {
    GPUError("Cannot do error interpolation with NWays = 1!");
    return false;
  }
  if ((param().rec.mergerReadFromTrackerDirectly || !param().par.earlyTpcTransform) && param().rec.NonConsecutiveIDs) {
    GPUError("incompatible settings for non consecutive ids");
    return false;
  }
  if (!param().rec.mergerReadFromTrackerDirectly && GetProcessingSettings().ompKernels) {
    GPUError("OMP Kernels require mergerReadFromTrackerDirectly");
    return false;
  }
  if (param().par.continuousMaxTimeBin > (int)GPUSettings::TPC_MAX_TF_TIME_BIN) {
    GPUError("configure max time bin exceeds 256 orbits");
    return false;
  }
  if (mRec->IsGPU() && std::max(GetProcessingSettings().nTPCClustererLanes + 1, GetProcessingSettings().nTPCClustererLanes * 2) + (GetProcessingSettings().doublePipeline ? 1 : 0) > mRec->NStreams()) {
    GPUError("NStreams must be > nTPCClustererLanes");
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
  if (!(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression) && (ProcessingSettings().tpcCompressionGatherMode == 1 || ProcessingSettings().tpcCompressionGatherMode == 3)) {
    GPUError("Invalid tpcCompressionGatherMode for compression on CPU");
    return false;
  }
  return true;
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
    mEventDisplay.reset(new GPUDisplay(GetProcessingSettings().eventDisplay, this, mQA.get()));
  }

  processors()->errorCodes.setMemory(mInputsHost->mErrorCodes);
  processors()->errorCodes.clear();

  if (mRec->IsGPU()) {
    if (processors()->calibObjects.fastTransform) {
      memcpy((void*)mFlatObjectsShadow.mCalibObjects.fastTransform, (const void*)processors()->calibObjects.fastTransform, sizeof(*processors()->calibObjects.fastTransform));
      memcpy((void*)mFlatObjectsShadow.mTpcTransformBuffer, (const void*)processors()->calibObjects.fastTransform->getFlatBufferPtr(), processors()->calibObjects.fastTransform->getFlatBufferSize());
      mFlatObjectsShadow.mCalibObjects.fastTransform->clearInternalBufferPtr();
      mFlatObjectsShadow.mCalibObjects.fastTransform->setActualBufferAddress(mFlatObjectsShadow.mTpcTransformBuffer);
      mFlatObjectsShadow.mCalibObjects.fastTransform->setFutureBufferAddress(mFlatObjectsDevice.mTpcTransformBuffer);
    }
#ifdef HAVE_O2HEADERS
    if (processors()->calibObjects.dEdxSplines) {
      memcpy((void*)mFlatObjectsShadow.mCalibObjects.dEdxSplines, (const void*)processors()->calibObjects.dEdxSplines, sizeof(*processors()->calibObjects.dEdxSplines));
      memcpy((void*)mFlatObjectsShadow.mdEdxSplinesBuffer, (const void*)processors()->calibObjects.dEdxSplines->getFlatBufferPtr(), processors()->calibObjects.dEdxSplines->getFlatBufferSize());
      mFlatObjectsShadow.mCalibObjects.dEdxSplines->clearInternalBufferPtr();
      mFlatObjectsShadow.mCalibObjects.dEdxSplines->setActualBufferAddress(mFlatObjectsShadow.mdEdxSplinesBuffer);
      mFlatObjectsShadow.mCalibObjects.dEdxSplines->setFutureBufferAddress(mFlatObjectsDevice.mdEdxSplinesBuffer);
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
    if (processors()->calibObjects.o2Propagator) {
      memcpy((void*)mFlatObjectsShadow.mCalibObjects.o2Propagator, (const void*)processors()->calibObjects.o2Propagator, sizeof(*processors()->calibObjects.o2Propagator));
      mFlatObjectsShadow.mCalibObjects.o2Propagator->setGPUField(&processorsDevice()->param.polynomialField);
      mFlatObjectsShadow.mCalibObjects.o2Propagator->setBz(param().polynomialField.GetNominalBz());
      mFlatObjectsShadow.mCalibObjects.o2Propagator->setMatLUT(mFlatObjectsShadow.mCalibObjects.matLUT);
    }
#endif
    TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mFlatObjectsShadow.mMemoryResFlat);
    memcpy((void*)&processorsShadow()->calibObjects, (void*)&mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects));
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->calibObjects - (char*)processors(), &mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects), -1); // First initialization, for users not using RunChain
    processorsShadow()->errorCodes.setMemory(mInputsShadow->mErrorCodes);
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->errorCodes - (char*)processors(), &processorsShadow()->errorCodes, sizeof(processorsShadow()->errorCodes), -1);
    TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mInputsHost->mResourceErrorCodes);
  }

  if (GetProcessingSettings().debugLevel >= 6) {
    mDebugFile->open(mRec->IsGPU() ? "GPU.out" : "CPU.out");
  }

  return 0;
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
  return mQA->InitQA();
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
  if (mChainTracking->GetTPCTransform()) {
    computePointerWithAlignment(mem, mCalibObjects.fastTransform, 1);
    computePointerWithAlignment(mem, mTpcTransformBuffer, mChainTracking->GetTPCTransform()->getFlatBufferSize());
  }
  if (mChainTracking->GetTPCPadGainCalib()) {
    computePointerWithAlignment(mem, mCalibObjects.tpcPadGain, 1);
  }
#ifdef HAVE_O2HEADERS
  if (mChainTracking->GetdEdxSplines()) {
    computePointerWithAlignment(mem, mCalibObjects.dEdxSplines, 1);
    computePointerWithAlignment(mem, mdEdxSplinesBuffer, mChainTracking->GetdEdxSplines()->getFlatBufferSize());
  }
  if (mChainTracking->GetMatLUT()) {
    computePointerWithAlignment(mem, mCalibObjects.matLUT, 1);
    computePointerWithAlignment(mem, mMatLUTBuffer, mChainTracking->GetMatLUT()->getFlatBufferSize());
  }
  if (mChainTracking->GetTRDGeometry()) {
    computePointerWithAlignment(mem, mCalibObjects.trdGeometry, 1);
  }
  if (mChainTracking->GetO2Propagator()) {
    computePointerWithAlignment(mem, mCalibObjects.o2Propagator, 1);
  }
#endif
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
  AllocateIOMemoryHelper(mIOPtrs.nMergedTracks, mIOPtrs.mergedTracks, mIOMem.mergedTracks);
  AllocateIOMemoryHelper(mIOPtrs.nMergedTrackHits, mIOPtrs.mergedTrackHits, mIOMem.mergedTrackHits);
  AllocateIOMemoryHelper(mIOPtrs.nMergedTrackHits, mIOPtrs.mergedTrackHitsXYZ, mIOMem.mergedTrackHitsXYZ);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracks, mIOPtrs.trdTracks, mIOMem.trdTracks);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdTracklets, mIOMem.trdTracklets);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTrackletsMC, mIOPtrs.trdTrackletsMC, mIOMem.trdTrackletsMC);
}

void GPUChainTracking::LoadClusterErrors() { param().LoadClusterErrors(); }

void GPUChainTracking::SetTPCFastTransform(std::unique_ptr<TPCFastTransform>&& tpcFastTransform)
{
  mTPCFastTransformU = std::move(tpcFastTransform);
  processors()->calibObjects.fastTransform = mTPCFastTransformU.get();
}

void GPUChainTracking::SetdEdxSplines(std::unique_ptr<TPCdEdxCalibrationSplines>&& dEdxSplines)
{
  mdEdxSplinesU = std::move(dEdxSplines);
  processors()->calibObjects.dEdxSplines = mdEdxSplinesU.get();
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

int GPUChainTracking::RunChain()
{
  if (GetProcessingSettings().ompAutoNThreads && !mRec->IsGPU()) {
    mRec->SetNOMPThreads(-1);
  }
  const auto threadContext = GetThreadContext();
  if (GetProcessingSettings().runCompressionStatistics && mCompressionStatistics == nullptr) {
    mCompressionStatistics.reset(new GPUTPCClusterStatistics);
  }
  const bool needQA = GPUQA::QAAvailable() && (GetProcessingSettings().runQA || (GetProcessingSettings().eventDisplay && mIOPtrs.nMCInfosTPC));
  if (needQA && mQA->IsInitialized() == false) {
    if (mQA->InitQA(GetProcessingSettings().runQA ? -GetProcessingSettings().runQA : -1)) {
      return 1;
    }
  }
  if (GetProcessingSettings().debugLevel >= 6) {
    *mDebugFile << "\n\nProcessing event " << mRec->getNEventsProcessed() << std::endl;
  }
  if (mRec->slavesExist() && mRec->IsGPU()) {
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->calibObjects - (char*)processors(), &mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects), 0); // Reinitialize
  }

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

  mRec->PushNonPersistentMemory(); // 1st stack level for TPC tracking slice data
  if (runRecoStep(RecoStep::TPCSliceTracking, &GPUChainTracking::RunTPCTrackingSlices)) {
    return 1;
  }

  for (unsigned int i = 0; i < NSLICES; i++) {
    // GPUInfo("slice %d clusters %d tracks %d", i, mClusterData[i].NumberOfClusters(), processors()->tpcTrackers[i].Output()->NTracks());
    processors()->tpcMerger.SetSliceData(i, param().rec.mergerReadFromTrackerDirectly ? nullptr : processors()->tpcTrackers[i].Output());
  }
  if (runRecoStep(RecoStep::TPCMerging, &GPUChainTracking::RunTPCTrackingMerger, false)) {
    return 1;
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCSliceTracking); // Release 1st stack level, TPC slice data not needed after merger

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
    SynchronizeStream(mRec->NStreams() - 2);
  }

  if (CheckErrorCodes()) {
    return 1;
  }

  if (GetProcessingSettings().ompAutoNThreads && !mRec->IsGPU()) {
    mRec->SetNOMPThreads(-1);
  }

  return GetProcessingSettings().doublePipeline ? 0 : RunChainFinalize();
}

int GPUChainTracking::RunChainFinalize()
{
#ifdef HAVE_O2HEADERS
  if (mIOPtrs.clustersNative && (GetRecoSteps() & RecoStep::TPCCompression) && GetProcessingSettings().runCompressionStatistics) {
    CompressedClusters c = *mIOPtrs.tpcCompressedClusters;
    mCompressionStatistics->RunStatistics(mIOPtrs.clustersNative, &c, param());
  }
#endif

  const bool needQA = GPUQA::QAAvailable() && (GetProcessingSettings().runQA || (GetProcessingSettings().eventDisplay && mIOPtrs.nMCInfosTPC));
  if (needQA) {
    mRec->getGeneralStepTimer(GeneralStep::QA).Start();
    mQA->RunQA(!GetProcessingSettings().runQA);
    mRec->getGeneralStepTimer(GeneralStep::QA).Stop();
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
          GetProcessingSettings().eventDisplay->mDisplayControl = 2;
        } else if (iKey == 'n') {
          break;
        } else if (iKey) {
          while (GetProcessingSettings().eventDisplay->mSendKey != 0) {
            Sleep(1);
          }
          GetProcessingSettings().eventDisplay->mSendKey = iKey;
        }
      }
    } while (GetProcessingSettings().eventDisplay->mDisplayControl == 0);
    if (GetProcessingSettings().eventDisplay->mDisplayControl == 2) {
      mDisplayRunning = false;
      GetProcessingSettings().eventDisplay->DisplayExit();
      ProcessingSettings().eventDisplay = nullptr;
      return (2);
    }
    GetProcessingSettings().eventDisplay->mDisplayControl = 0;
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
  if (param().rec.GlobalTracking) {
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

int GPUChainTracking::CheckErrorCodes(bool cpuOnly)
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
      retVal = 1;
      GPUError("GPUReconstruction suffered from an error in the %s part", i ? "GPU" : "CPU");
      processors()->errorCodes.printErrors();
    }
  }
  return retVal;
}

void GPUChainTracking::ClearErrorCodes()
{
  processors()->errorCodes.clear();
  const auto& threadContext = GetThreadContext();
  if (mRec->IsGPU()) {
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->errorCodes - (char*)processors(), &processorsShadow()->errorCodes, sizeof(processorsShadow()->errorCodes), 0);
  }
  TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mInputsHost->mResourceErrorCodes, 0);
}

void GPUChainTracking::SetDefaultO2PropagatorForGPU()
{
#ifdef HAVE_O2HEADERS
  o2::base::Propagator* prop = param().GetDefaultO2Propagator(true);
  prop->setMatLUT(processors()->calibObjects.matLUT);
  SetO2Propagator(prop);
#endif
}
