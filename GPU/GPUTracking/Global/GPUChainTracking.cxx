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

#ifdef GPUCA_O2_LIB
#include "CommonDataFormat/InteractionRecord.h"
#endif
#ifdef HAVE_O2HEADERS
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif
#ifndef GPUCA_NO_VC
#include <Vc/Vc>
#endif
#include <fstream>
#include <mutex>
#include <condition_variable>

#include "GPUChainTracking.h"
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
#include "GPUReconstructionConvert.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"

#ifdef HAVE_O2HEADERS
#include "GPUTPCClusterStatistics.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "GPURawData.h"
#include "DetectorsRaw/RDHUtils.h"
#include "GPUHostDataTypes.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCdEdxCalibrationSplines.h"
#include "TPCClusterDecompressor.h"
#include "GPUTPCCFChainContext.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#include "TPCFastTransform.h"

#include "utils/linux_helpers.h"
using namespace GPUCA_NAMESPACE::gpu;

#include "GPUO2DataTypes.h"

using namespace o2::tpc;
using namespace o2::trd;
using namespace o2::tpc::constants;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUChainTrackingFinalContext {
  GPUReconstruction* rec = nullptr;
  std::mutex mutex;
  std::condition_variable cond;
  bool ready = false;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

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
  if (!param().earlyTpcTransform) {
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
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) && processors()->calibObjects.tpcCalibration == nullptr) {
    GPUError("Cannot run gain calibration without calibration object");
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
  if ((param().rec.mergerReadFromTrackerDirectly || !param().earlyTpcTransform) && param().rec.NonConsecutiveIDs) {
    GPUError("incompatible settings for non consecutive ids");
    return false;
  }
  if (!param().rec.mergerReadFromTrackerDirectly && GetProcessingSettings().ompKernels) {
    GPUError("OMP Kernels require mergerReadFromTrackerDirectly");
    return false;
  }
  if (param().continuousMaxTimeBin > (int)GPUSettings::TPC_MAX_TF_TIME_BIN) {
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
    if (((GetRecoStepsOutputs().isSet(GPUDataTypes::InOutType::TPCCompressedClusters) && mOutputCompressedClusters == nullptr) || (GetRecoStepsOutputs().isSet(GPUDataTypes::InOutType::TPCClusters) && mOutputClustersNative == nullptr) || (GetRecoStepsOutputs().isSet(GPUDataTypes::InOutType::TPCMergedTracks) && mOutputTPCTracks == nullptr))) {
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

  if (mOutputCompressedClusters == nullptr) {
    mOutputCompressedClusters = &mRec->OutputControl();
  }
  if (mOutputClustersNative == nullptr) {
    mOutputClustersNative = &mRec->OutputControl();
  }
  if (mOutputTPCTracks == nullptr) {
    mOutputTPCTracks = &mRec->OutputControl();
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
    if (processors()->calibObjects.tpcCalibration) {
      memcpy((void*)mFlatObjectsShadow.mCalibObjects.tpcCalibration, (const void*)processors()->calibObjects.tpcCalibration, sizeof(*processors()->calibObjects.tpcCalibration));
    }
#endif
    TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mFlatObjectsShadow.mMemoryResFlat);
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
  if (GetProcessingSettings().runQA && mQA->IsInitialized()) {
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
  if (mChainTracking->GetTPCCalibration()) {
    computePointerWithAlignment(mem, mCalibObjects.tpcCalibration, 1);
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

int GPUChainTracking::ConvertNativeToClusterData()
{
#ifdef HAVE_O2HEADERS
  mRec->PushNonPersistentMemory();
  const auto& threadContext = GetThreadContext();
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCConversion;
  GPUTPCConvert& convert = processors()->tpcConverter;
  GPUTPCConvert& convertShadow = doGPU ? processorsShadow()->tpcConverter : convert;

  SetupGPUProcessor(&convert, true);
  if (doGPU) {
    if (!(mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
      mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mIOPtrs.clustersNative->nClustersTotal;
      AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
      processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
      WriteToConstantMemory(RecoStep::TPCConversion, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
      *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
      mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
      mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
      GPUMemCpy(RecoStep::TPCConversion, mInputsShadow->mPclusterNativeBuffer, mIOPtrs.clustersNative->clustersLinear, sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * mIOPtrs.clustersNative->nClustersTotal, 0, true);
      TransferMemoryResourceLinkToGPU(RecoStep::TPCConversion, mInputsHost->mResourceClusterNativeAccess, 0);
    }
  }
  if (!param().earlyTpcTransform) {
    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Early transform inactive, skipping TPC Early transformation kernel, transformed on the fly during slice data creation / refit");
    }
    return 0;
  }
  for (unsigned int i = 0; i < NSLICES; i++) {
    convert.mMemory->clusters[i] = convertShadow.mClusters + mIOPtrs.clustersNative->clusterOffset[i][0];
  }

  WriteToConstantMemory(RecoStep::TPCConversion, (char*)&processors()->tpcConverter - (char*)processors(), &convertShadow, sizeof(convertShadow), 0);
  TransferMemoryResourcesToGPU(RecoStep::TPCConversion, &convert, 0);
  runKernel<GPUTPCConvertKernel>(GetGridBlk(NSLICES * GPUCA_ROW_COUNT, 0), krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(RecoStep::TPCConversion, &convert, 0);
  SynchronizeStream(0);

  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.nClusterData[i] = (i == NSLICES - 1 ? mIOPtrs.clustersNative->nClustersTotal : mIOPtrs.clustersNative->clusterOffset[i + 1][0]) - mIOPtrs.clustersNative->clusterOffset[i][0];
    mIOPtrs.clusterData[i] = convert.mClusters + mIOPtrs.clustersNative->clusterOffset[i][0];
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCConversion);
#endif
  return 0;
}

void GPUChainTracking::ConvertNativeToClusterDataLegacy()
{
  ClusterNativeAccess* tmp = mIOMem.clusterNativeAccess.get();
  if (tmp != mIOPtrs.clustersNative) {
    *tmp = *mIOPtrs.clustersNative;
  }
  GPUReconstructionConvert::ConvertNativeToClusterData(mIOMem.clusterNativeAccess.get(), mIOMem.clusterData, mIOPtrs.nClusterData, processors()->calibObjects.fastTransform, param().continuousMaxTimeBin);
  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.clusterData[i] = mIOMem.clusterData[i].get();
    if (GetProcessingSettings().registerStandaloneInputMemory) {
      if (mRec->registerMemoryForGPU(mIOMem.clusterData[i].get(), mIOPtrs.nClusterData[i] * sizeof(*mIOPtrs.clusterData[i]))) {
        throw std::runtime_error("Error registering memory for GPU");
      }
    }
  }
  mIOPtrs.clustersNative = nullptr;
  mIOMem.clustersNative.reset(nullptr);
}

void GPUChainTracking::ConvertRun2RawToNative()
{
  GPUReconstructionConvert::ConvertRun2RawToNative(*mIOMem.clusterNativeAccess, mIOMem.clustersNative, mIOPtrs.rawClusters, mIOPtrs.nRawClusters);
  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.rawClusters[i] = nullptr;
    mIOPtrs.nRawClusters[i] = 0;
    mIOMem.rawClusters[i].reset(nullptr);
    mIOPtrs.clusterData[i] = nullptr;
    mIOPtrs.nClusterData[i] = 0;
    mIOMem.clusterData[i].reset(nullptr);
  }
  mIOPtrs.clustersNative = mIOMem.clusterNativeAccess.get();
  if (GetProcessingSettings().registerStandaloneInputMemory) {
    if (mRec->registerMemoryForGPU(mIOMem.clustersNative.get(), mIOMem.clusterNativeAccess->nClustersTotal * sizeof(*mIOMem.clusterNativeAccess->clustersLinear))) {
      throw std::runtime_error("Error registering memory for GPU");
    }
  }
}

void GPUChainTracking::ConvertZSEncoder(bool zs12bit)
{
#ifdef HAVE_O2HEADERS
  mIOMem.tpcZSmeta2.reset(new GPUTrackingInOutZS::GPUTrackingInOutZSMeta);
  mIOMem.tpcZSmeta.reset(new GPUTrackingInOutZS);
  GPUReconstructionConvert::RunZSEncoder<o2::tpc::Digit>(*mIOPtrs.tpcPackedDigits, &mIOMem.tpcZSpages, &mIOMem.tpcZSmeta2->n[0][0], nullptr, nullptr, param(), zs12bit, true);
  GPUReconstructionConvert::RunZSEncoderCreateMeta(mIOMem.tpcZSpages.get(), &mIOMem.tpcZSmeta2->n[0][0], &mIOMem.tpcZSmeta2->ptr[0][0], mIOMem.tpcZSmeta.get());
  mIOPtrs.tpcZS = mIOMem.tpcZSmeta.get();
  if (GetProcessingSettings().registerStandaloneInputMemory) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[i].count[j]; k++) {
          if (mRec->registerMemoryForGPU(mIOPtrs.tpcZS->slice[i].zsPtr[j][k], mIOPtrs.tpcZS->slice[i].nZSPtr[j][k] * TPCZSHDR::TPC_ZS_PAGE_SIZE)) {
            throw std::runtime_error("Error registering memory for GPU");
          }
        }
      }
    }
  }
#endif
}

void GPUChainTracking::ConvertZSFilter(bool zs12bit)
{
  GPUReconstructionConvert::RunZSFilter(mIOMem.tpcDigits, mIOPtrs.tpcPackedDigits->tpcDigits, mIOMem.digitMap->nTPCDigits, mIOPtrs.tpcPackedDigits->nTPCDigits, param(), zs12bit, param().rec.tpcZSthreshold);
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

void GPUChainTracking::SetTRDGeometry(std::unique_ptr<o2::trd::TRDGeometryFlat>&& geo)
{
  mTRDGeometryU = std::move(geo);
  processors()->calibObjects.trdGeometry = mTRDGeometryU.get();
}

int GPUChainTracking::ReadEvent(unsigned int iSlice, int threadId)
{
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Running ReadEvent for slice %d on thread %d\n", iSlice, threadId);
  }
  runKernel<GPUTPCCreateSliceData>({GetGridAuto(0, GPUReconstruction::krnlDeviceType::CPU)}, {iSlice});
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Finished ReadEvent for slice %d on thread %d\n", iSlice, threadId);
  }
  return (0);
}

void GPUChainTracking::WriteOutput(int iSlice, int threadId)
{
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Running WriteOutput for slice %d on thread %d\n", iSlice, threadId);
  }
  if (GetProcessingSettings().nDeviceHelperThreads) {
    while (mLockAtomic.test_and_set(std::memory_order_acquire)) {
      ;
    }
  }
  processors()->tpcTrackers[iSlice].WriteOutputPrepare();
  if (GetProcessingSettings().nDeviceHelperThreads) {
    mLockAtomic.clear();
  }
  processors()->tpcTrackers[iSlice].WriteOutput();
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Finished WriteOutput for slice %d on thread %d\n", iSlice, threadId);
  }
}

int GPUChainTracking::ForwardTPCDigits()
{
#ifdef HAVE_O2HEADERS
  if (GetRecoStepsGPU() & RecoStep::TPCClusterFinding) {
    throw std::runtime_error("Cannot forward TPC digits with Clusterizer on GPU");
  }
  std::vector<ClusterNative> tmp[NSLICES][GPUCA_ROW_COUNT];
  unsigned int nTotal = 0;
  const float zsThreshold = param().rec.tpcZSthreshold;
  for (int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < mIOPtrs.tpcPackedDigits->nTPCDigits[i]; j++) {
      const auto& d = mIOPtrs.tpcPackedDigits->tpcDigits[i][j];
      if (d.getChargeFloat() >= zsThreshold) {
        ClusterNative c;
        c.setTimeFlags(d.getTimeStamp(), 0);
        c.setPad(d.getPad());
        c.setSigmaTime(1);
        c.setSigmaPad(1);
        c.qTot = c.qMax = d.getChargeFloat();
        tmp[i][d.getRow()].emplace_back(c);
        nTotal++;
      }
    }
  }
  mIOMem.clustersNative.reset(new ClusterNative[nTotal]);
  nTotal = 0;
  mClusterNativeAccess->clustersLinear = mIOMem.clustersNative.get();
  for (int i = 0; i < NSLICES; i++) {
    for (int j = 0; j < GPUCA_ROW_COUNT; j++) {
      mClusterNativeAccess->nClusters[i][j] = tmp[i][j].size();
      memcpy(&mIOMem.clustersNative[nTotal], tmp[i][j].data(), tmp[i][j].size() * sizeof(*mClusterNativeAccess->clustersLinear));
      nTotal += tmp[i][j].size();
    }
  }
  mClusterNativeAccess->setOffsetPtrs();
  mIOPtrs.tpcPackedDigits = nullptr;
  mIOPtrs.clustersNative = mClusterNativeAccess.get();
  GPUInfo("Forwarded %u TPC clusters", nTotal);
  mRec->MemoryScalers()->nTPCHits = nTotal;
#endif
  return 0;
}

int GPUChainTracking::GlobalTracking(unsigned int iSlice, int threadId, bool synchronizeOutput)
{
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("GPU Tracker running Global Tracking for slice %u on thread %d\n", iSlice, threadId);
  }

  GPUReconstruction::krnlDeviceType deviceType = GetProcessingSettings().fullMergerOnGPU ? GPUReconstruction::krnlDeviceType::Auto : GPUReconstruction::krnlDeviceType::CPU;
  runKernel<GPUTPCGlobalTracking>(GetGridBlk(256, iSlice % mRec->NStreams(), deviceType), {iSlice});
  if (GetProcessingSettings().fullMergerOnGPU) {
    TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, processors()->tpcTrackers[iSlice].MemoryResCommon(), iSlice % mRec->NStreams());
  }
  if (synchronizeOutput) {
    SynchronizeStream(iSlice % mRec->NStreams());
  }

  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("GPU Tracker finished Global Tracking for slice %u on thread %d\n", iSlice, threadId);
  }
  return (0);
}

#ifdef GPUCA_TPC_GEOMETRY_O2
std::pair<unsigned int, unsigned int> GPUChainTracking::TPCClusterizerDecodeZSCountUpdate(unsigned int iSlice, const CfFragment& fragment)
{
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;
  GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
  GPUTPCClusterFinder::ZSOffset* o = processors()->tpcClusterer[iSlice].mPzsOffsets;
  unsigned int digits = 0;
  unsigned short pages = 0;
  for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    unsigned short posInEndpoint = 0;
    unsigned short pagesEndpoint = 0;
    clusterer.mMinMaxCN[j] = mCFContext->fragmentData[fragment.index].minMaxCN[iSlice][j];
    if (doGPU) {
      for (unsigned int k = clusterer.mMinMaxCN[j].minC; k < clusterer.mMinMaxCN[j].maxC; k++) {
        const unsigned int minL = (k == clusterer.mMinMaxCN[j].minC) ? clusterer.mMinMaxCN[j].minN : 0;
        const unsigned int maxL = (k + 1 == clusterer.mMinMaxCN[j].maxC) ? clusterer.mMinMaxCN[j].maxN : mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
        for (unsigned int l = minL; l < maxL; l++) {
          unsigned short pageDigits = mCFContext->fragmentData[fragment.index].pageDigits[iSlice][j][posInEndpoint++];
          if (pageDigits) {
            *(o++) = GPUTPCClusterFinder::ZSOffset{digits, j, pagesEndpoint};
            digits += pageDigits;
          }
          pagesEndpoint++;
        }
      }
      pages += pagesEndpoint;
    } else {
      clusterer.mPzsOffsets[j] = GPUTPCClusterFinder::ZSOffset{digits, j, 0};
      digits += mCFContext->fragmentData[fragment.index].nDigits[iSlice][j];
      pages += mCFContext->fragmentData[fragment.index].nPages[iSlice][j];
    }
  }

  return {digits, pages};
}

std::pair<unsigned int, unsigned int> GPUChainTracking::TPCClusterizerDecodeZSCount(unsigned int iSlice, const CfFragment& fragment)
{
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Start();
  unsigned int nDigits = 0;
  unsigned int nPages = 0;
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;
  int firstHBF = o2::raw::RDHUtils::getHeartBeatOrbit(*(const RAWDataHeaderGPU*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0]);

  for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    for (unsigned int k = 0; k < mCFContext->nFragments; k++) {
      mCFContext->fragmentData[k].minMaxCN[iSlice][j].maxC = mIOPtrs.tpcZS->slice[iSlice].count[j];
      mCFContext->fragmentData[k].minMaxCN[iSlice][j].minC = mCFContext->fragmentData[k].minMaxCN[iSlice][j].maxC;
      mCFContext->fragmentData[k].minMaxCN[iSlice][j].maxN = mIOPtrs.tpcZS->slice[iSlice].count[j] ? mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][mIOPtrs.tpcZS->slice[iSlice].count[j] - 1] : 0;
      mCFContext->fragmentData[k].minMaxCN[iSlice][j].minN = mCFContext->fragmentData[k].minMaxCN[iSlice][j].maxN;
    }

#ifndef GPUCA_NO_VC
    if (GetProcessingSettings().prefetchTPCpageScan >= 3 && j < GPUTrackingInOutZS::NENDPOINTS - 1) {
      for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j + 1]; k++) {
        for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j + 1][k]; l++) {
          Vc::Common::prefetchMid(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j + 1][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          Vc::Common::prefetchMid(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j + 1][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(RAWDataHeaderGPU));
        }
      }
    }
#endif

    bool firstNextFound = false;
    CfFragment f = fragment;
    CfFragment fNext = f.next();
    bool firstSegment = true;

    for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
      for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]; l++) {
        if (f.isEnd()) {
          GPUError("Time bin passed last fragment");
          return {0, 0};
        }
#ifndef GPUCA_NO_VC
        if (GetProcessingSettings().prefetchTPCpageScan >= 2 && l + 1 < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]) {
          Vc::Common::prefetchForOneRead(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + (l + 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          Vc::Common::prefetchForOneRead(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + (l + 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(RAWDataHeaderGPU));
        }
#endif
        const unsigned char* const page = ((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        const RAWDataHeaderGPU* rdh = (const RAWDataHeaderGPU*)page;
        if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(RAWDataHeaderGPU)) {
          nPages++;
          if (mCFContext->fragmentData[f.index].nPages[iSlice][j]) {
            mCFContext->fragmentData[f.index].nPages[iSlice][j]++;
            mCFContext->fragmentData[f.index].pageDigits[iSlice][j].emplace_back(0);
          }
          continue;
        }
        const TPCZSHDR* const hdr = (const TPCZSHDR*)(page + sizeof(RAWDataHeaderGPU));
        unsigned int timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
        for (unsigned int m = 0; m < 2; m++) {
          CfFragment& fm = m ? fNext : f;
          if (timeBin + hdr->nTimeBins >= (unsigned int)fm.first() && timeBin < (unsigned int)fm.last() && !fm.isEnd()) {
            mCFContext->fragmentData[fm.index].nPages[iSlice][j]++;
            mCFContext->fragmentData[fm.index].nDigits[iSlice][j] += hdr->nADCsamples;
            if (doGPU) {
              mCFContext->fragmentData[fm.index].pageDigits[iSlice][j].emplace_back(hdr->nADCsamples);
            }
          }
        }
        if (firstSegment && timeBin + hdr->nTimeBins >= (unsigned int)f.first()) {
          mCFContext->fragmentData[f.index].minMaxCN[iSlice][j].minC = k;
          mCFContext->fragmentData[f.index].minMaxCN[iSlice][j].minN = l;
          firstSegment = false;
        }
        if (!firstNextFound && timeBin + hdr->nTimeBins >= (unsigned int)fNext.first() && !fNext.isEnd()) {
          mCFContext->fragmentData[fNext.index].minMaxCN[iSlice][j].minC = k;
          mCFContext->fragmentData[fNext.index].minMaxCN[iSlice][j].minN = l;
          firstNextFound = true;
        }
        while (!f.isEnd() && timeBin >= (unsigned int)f.last()) {
          if (!firstNextFound && !fNext.isEnd()) {
            mCFContext->fragmentData[fNext.index].minMaxCN[iSlice][j].minC = k;
            mCFContext->fragmentData[fNext.index].minMaxCN[iSlice][j].minN = l;
          }
          mCFContext->fragmentData[f.index].minMaxCN[iSlice][j].maxC = k + 1;
          mCFContext->fragmentData[f.index].minMaxCN[iSlice][j].maxN = l;
          f = fNext;
          fNext = f.next();
          firstNextFound = false;
        }
        if (timeBin + hdr->nTimeBins > mCFContext->tpcMaxTimeBin) {
          mCFContext->tpcMaxTimeBin = timeBin + hdr->nTimeBins;
        }

        nPages++;
        nDigits += hdr->nADCsamples;
      }
    }
  }
  mCFContext->nPagesTotal += nPages;
  mCFContext->nPagesSector[iSlice] = nPages;
  mCFContext->nPagesSectorMax = std::max(mCFContext->nPagesSectorMax, nPages);

  unsigned int digitsFragment = 0;
  for (unsigned int i = 0; i < mCFContext->nFragments; i++) {
    unsigned int pages = 0;
    unsigned int digits = 0;
    for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      pages += mCFContext->fragmentData[i].nPages[iSlice][j];
      digits += mCFContext->fragmentData[i].nDigits[iSlice][j];
    }
    mCFContext->nPagesFragmentMax = std::max(mCFContext->nPagesSectorMax, pages);
    digitsFragment = std::max(digitsFragment, digits);
  }
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Stop();
  return {nDigits, digitsFragment};
}

void GPUChainTracking::RunTPCClusterizer_compactPeaks(GPUTPCClusterFinder& clusterer, GPUTPCClusterFinder& clustererShadow, int stage, bool doGPU, int lane)
{
  auto& in = stage ? clustererShadow.mPpeakPositions : clustererShadow.mPpositions;
  auto& out = stage ? clustererShadow.mPfilteredPeakPositions : clustererShadow.mPpeakPositions;
  if (doGPU) {
    const unsigned int iSlice = clusterer.mISlice;
    auto& count = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;

    std::vector<size_t> counts;

    unsigned int nSteps = clusterer.getNSteps(count);
    if (nSteps > clusterer.mNBufs) {
      GPUError("Clusterer buffers exceeded (%u > %u)", nSteps, (int)clusterer.mNBufs);
      exit(1);
    }

    size_t tmpCount = count;
    if (nSteps > 1) {
      for (unsigned int i = 1; i < nSteps; i++) {
        counts.push_back(tmpCount);
        if (i == 1) {
          runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanStart>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, stage);
        } else {
          runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanUp>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, tmpCount);
        }
        tmpCount = (tmpCount + clusterer.mScanWorkGroupSize - 1) / clusterer.mScanWorkGroupSize;
      }

      runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanTop>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, nSteps, tmpCount);

      for (unsigned int i = nSteps - 1; i > 1; i--) {
        tmpCount = counts[i - 1];
        runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanDown>(GetGrid(tmpCount - clusterer.mScanWorkGroupSize, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, clusterer.mScanWorkGroupSize, tmpCount);
      }
    }

    runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::compactDigits>(GetGrid(count, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, 1, stage, in, out);
  } else {
    auto& nOut = stage ? clusterer.mPmemory->counters.nClusters : clusterer.mPmemory->counters.nPeaks;
    auto& nIn = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;
    size_t count = 0;
    for (size_t i = 0; i < nIn; i++) {
      if (clusterer.mPisPeak[i]) {
        out[count++] = in[i];
      }
    }
    nOut = count;
  }
}

std::pair<unsigned int, unsigned int> GPUChainTracking::RunTPCClusterizer_transferZS(int iSlice, const CfFragment& fragment, int lane)
{
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;
  const auto& retVal = TPCClusterizerDecodeZSCountUpdate(iSlice, fragment);
  if (doGPU) {
    GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
    GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
    unsigned int nPagesSector = 0;
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      unsigned int nPages = 0;
      mInputsHost->mPzsMeta->slice[iSlice].zsPtr[j] = &mInputsShadow->mPzsPtrs[iSlice * GPUTrackingInOutZS::NENDPOINTS + j];
      mInputsHost->mPzsPtrs[iSlice * GPUTrackingInOutZS::NENDPOINTS + j] = clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE;
      for (unsigned int k = clusterer.mMinMaxCN[j].minC; k < clusterer.mMinMaxCN[j].maxC; k++) {
        const unsigned int min = (k == clusterer.mMinMaxCN[j].minC) ? clusterer.mMinMaxCN[j].minN : 0;
        const unsigned int max = (k + 1 == clusterer.mMinMaxCN[j].maxC) ? clusterer.mMinMaxCN[j].maxN : mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
        if (max > min) {
          char* src = (char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k] + min * TPCZSHDR::TPC_ZS_PAGE_SIZE;
          size_t size = o2::raw::RDHUtils::getMemorySize(*(const RAWDataHeaderGPU*)src);
          size = (max - min - 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE + (size ? TPCZSHDR::TPC_ZS_PAGE_SIZE : size);
          GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE, src, size, lane, true);
        }
        nPages += max - min;
      }
      mInputsHost->mPzsMeta->slice[iSlice].nZSPtr[j] = &mInputsShadow->mPzsSizes[iSlice * GPUTrackingInOutZS::NENDPOINTS + j];
      mInputsHost->mPzsSizes[iSlice * GPUTrackingInOutZS::NENDPOINTS + j] = nPages;
      mInputsHost->mPzsMeta->slice[iSlice].count[j] = 1;
      nPagesSector += nPages;
    }
    GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPzsOffsets, clusterer.mPzsOffsets, clusterer.mNMaxPages * sizeof(*clusterer.mPzsOffsets), lane, true);
  }
  return retVal;
}

int GPUChainTracking::RunTPCClusterizer_prepare(bool restorePointers)
{
  if (restorePointers) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processors()->tpcClusterer[iSlice].mPzsOffsets = mCFContext->ptrSave[iSlice].zsOffsetHost;
      processorsShadow()->tpcClusterer[iSlice].mPzsOffsets = mCFContext->ptrSave[iSlice].zsOffsetDevice;
      processorsShadow()->tpcClusterer[iSlice].mPzs = mCFContext->ptrSave[iSlice].zsDevice;
    }
    processorsShadow()->ioPtrs.clustersNative = mCFContext->ptrClusterNativeSave;
    return 0;
  }
  const auto& threadContext = GetThreadContext();
  mRec->MemoryScalers()->nTPCdigits = 0;
  if (mCFContext == nullptr) {
    mCFContext.reset(new GPUTPCCFChainContext);
  }
  mCFContext->tpcMaxTimeBin = std::max<int>(param().continuousMaxTimeBin, TPC_MAX_FRAGMENT_LEN);
  const CfFragment fragmentMax{(tpccf::TPCTime)mCFContext->tpcMaxTimeBin + 1, TPC_MAX_FRAGMENT_LEN};
  mCFContext->prepare(mIOPtrs.tpcZS, fragmentMax);
  if (mIOPtrs.tpcZS) {
    unsigned int nDigitsFragment[NSLICES];
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (mIOPtrs.tpcZS->slice[iSlice].count[0] == 0) {
        GPUError("No ZS data present, must contain at least empty HBF");
        return 1;
      }
      const void* rdh = mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0];
      if (o2::raw::RDHUtils::getVersion<o2::gpu::RAWDataHeaderGPU>() != o2::raw::RDHUtils::getVersion(rdh)) {
        GPUError("Data has invalid RDH version %d, %d required\n", o2::raw::RDHUtils::getVersion(rdh), o2::raw::RDHUtils::getVersion<o2::gpu::RAWDataHeaderGPU>());
        return 1;
      }
#ifndef GPUCA_NO_VC
      if (GetProcessingSettings().prefetchTPCpageScan >= 1 && iSlice < NSLICES - 1) {
        for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
          for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
            for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]; l++) {
              Vc::Common::prefetchFar(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice + 1].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
              Vc::Common::prefetchFar(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice + 1].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(RAWDataHeaderGPU));
            }
          }
        }
      }
#endif
      const auto& x = TPCClusterizerDecodeZSCount(iSlice, fragmentMax);
      nDigitsFragment[iSlice] = x.second;
      processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits = x.first;
      mRec->MemoryScalers()->nTPCdigits += x.first;
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processors()->tpcClusterer[iSlice].SetNMaxDigits(processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsFragment[iSlice]);
      if (mRec->IsGPU()) {
        processorsShadow()->tpcClusterer[iSlice].SetNMaxDigits(processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsFragment[iSlice]);
      }
      if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
        mPipelineNotifyCtx->rec->AllocateRegisteredForeignMemory(processors()->tpcClusterer[iSlice].mZSOffsetId, mRec);
        mPipelineNotifyCtx->rec->AllocateRegisteredForeignMemory(processors()->tpcClusterer[iSlice].mZSId, mRec);
      } else {
        AllocateRegisteredMemory(processors()->tpcClusterer[iSlice].mZSOffsetId);
        AllocateRegisteredMemory(processors()->tpcClusterer[iSlice].mZSId);
      }
    }
  } else {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      unsigned int nDigits = mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice];
      mRec->MemoryScalers()->nTPCdigits += nDigits;
      processors()->tpcClusterer[iSlice].SetNMaxDigits(nDigits, mCFContext->nPagesFragmentMax, nDigits);
    }
  }
  if (mIOPtrs.tpcZS) {
    GPUInfo("Event has %u 8kb TPC ZS pages, %lld digits", mCFContext->nPagesTotal, (long long int)mRec->MemoryScalers()->nTPCdigits);
  } else {
    GPUInfo("Event has %lld TPC Digits", (long long int)mRec->MemoryScalers()->nTPCdigits);
  }
  mCFContext->fragmentFirst = CfFragment{std::max<int>(mCFContext->tpcMaxTimeBin + 1, TPC_MAX_FRAGMENT_LEN), TPC_MAX_FRAGMENT_LEN};
  for (int iSlice = 0; iSlice < GetProcessingSettings().nTPCClustererLanes && iSlice < NSLICES; iSlice++) {
    if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
      mCFContext->nextPos[iSlice] = RunTPCClusterizer_transferZS(iSlice, mCFContext->fragmentFirst, GetProcessingSettings().nTPCClustererLanes + iSlice);
    }
  }

  if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      mCFContext->ptrSave[iSlice].zsOffsetHost = processors()->tpcClusterer[iSlice].mPzsOffsets;
      mCFContext->ptrSave[iSlice].zsOffsetDevice = processorsShadow()->tpcClusterer[iSlice].mPzsOffsets;
      mCFContext->ptrSave[iSlice].zsDevice = processorsShadow()->tpcClusterer[iSlice].mPzs;
    }
  }
  return 0;
}
#endif

int GPUChainTracking::RunTPCClusterizer(bool synchronizeOutput)
{
  if (param().rec.fwdTPCDigitsAsClusters) {
    return ForwardTPCDigits();
  }
#ifdef GPUCA_TPC_GEOMETRY_O2
  mRec->PushNonPersistentMemory();
  const auto& threadContext = GetThreadContext();
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;
  if (RunTPCClusterizer_prepare(mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer)) {
    return 1;
  }

  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    processors()->tpcClusterer[iSlice].SetMaxData(mIOPtrs); // First iteration to set data sizes
  }
  mRec->ComputeReuseMax(nullptr); // Resolve maximums for shared buffers
  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    SetupGPUProcessor(&processors()->tpcClusterer[iSlice], true); // Now we allocate
  }
  if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
    RunTPCClusterizer_prepare(true); // Restore some pointers, allocated by the other pipeline, and set to 0 by SetupGPUProcessor (since not allocated in this pipeline)
  }

  if (doGPU && mIOPtrs.tpcZS) {
    processorsShadow()->ioPtrs.tpcZS = mInputsShadow->mPzsMeta;
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), mRec->NStreams() - 1);
  }
  if (doGPU) {
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)processors()->tpcClusterer - (char*)processors(), processorsShadow()->tpcClusterer, sizeof(GPUTPCClusterFinder) * NSLICES, mRec->NStreams() - 1, &mEvents->init);
  }

  size_t nClsTotal = 0;
  ClusterNativeAccess* tmpNative = mClusterNativeAccess.get();

  // setup MC Labels
  bool propagateMCLabels = GetProcessingSettings().runMC && processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC != nullptr;

  auto* digitsMC = propagateMCLabels ? processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC : nullptr;

  if (param().continuousMaxTimeBin > 0 && mCFContext->tpcMaxTimeBin >= (unsigned int)std::max(param().continuousMaxTimeBin + 1, TPC_MAX_FRAGMENT_LEN)) {
    GPUError("Input data has invalid time bin\n");
    return 1;
  }
  bool buildNativeGPU = (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression);
  bool buildNativeHost = mRec->GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCClusters; // TODO: Should do this also when clusters are needed for later steps on the host but not requested as output
  mRec->MemoryScalers()->nTPCHits = mRec->MemoryScalers()->NTPCClusters(mRec->MemoryScalers()->nTPCdigits);
  mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mRec->MemoryScalers()->nTPCHits;
  if (buildNativeGPU) {
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
  }
  if (buildNativeHost && !(buildNativeGPU && GetProcessingSettings().delayedOutput)) {
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput, mOutputClustersNative);
  }

  GPUTPCLinearLabels mcLinearLabels;
  if (propagateMCLabels) {
    // No need to overallocate here, nTPCHits is anyway an upper bound used for the GPU cluster buffer, and we can always enlarge the buffer anyway
    mcLinearLabels.header.reserve(mRec->MemoryScalers()->nTPCHits / 2);
    mcLinearLabels.data.reserve(mRec->MemoryScalers()->nTPCHits);
  }

  char transferRunning[NSLICES] = {0};
  unsigned int outputQueueStart = mOutputQueue.size();

  for (unsigned int iSliceBase = 0; iSliceBase < NSLICES; iSliceBase += GetProcessingSettings().nTPCClustererLanes) {
    std::vector<bool> laneHasData(GetProcessingSettings().nTPCClustererLanes, false);
    for (CfFragment fragment = mCFContext->fragmentFirst; !fragment.isEnd(); fragment = fragment.next()) {
      if (GetProcessingSettings().debugLevel >= 3) {
        GPUInfo("Processing time bins [%d, %d) for sectors %d to %d", fragment.start, fragment.last(), iSliceBase, iSliceBase + GetProcessingSettings().nTPCClustererLanes - 1);
      }
      for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        if (fragment.index != 0) {
          SynchronizeStream(lane); // Don't overwrite charge map from previous iteration until cluster computation is finished
        }

        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        clusterer.mPmemory->counters.nPeaks = clusterer.mPmemory->counters.nClusters = 0;
        clusterer.mPmemory->fragment = fragment;

        if (propagateMCLabels && fragment.index == 0) {
          clusterer.PrepareMC();
          clusterer.mPinputLabels = digitsMC->v[iSlice];
          // TODO: Why is the number of header entries in truth container
          // sometimes larger than the number of digits?
          assert(clusterer.mPinputLabels->getIndexedSize() >= mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice]);
        }

        if (mIOPtrs.tpcPackedDigits) {
          bool setDigitsOnGPU = doGPU && not mIOPtrs.tpcZS;
          bool setDigitsOnHost = (not doGPU && not mIOPtrs.tpcZS) || propagateMCLabels;
          auto* inDigits = mIOPtrs.tpcPackedDigits;
          size_t numDigits = inDigits->nTPCDigits[iSlice];
          if (setDigitsOnGPU) {
            GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPdigits, inDigits->tpcDigits[iSlice], sizeof(clustererShadow.mPdigits[0]) * numDigits, lane, true);
            clusterer.mPmemory->counters.nDigits = numDigits;
          } else if (setDigitsOnHost) {
            clusterer.mPdigits = const_cast<o2::tpc::Digit*>(inDigits->tpcDigits[iSlice]); // TODO: Needs fixing (invalid const cast)
            clusterer.mPmemory->counters.nDigits = numDigits;
          }
        }

        if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
          clusterer.mPmemory->counters.nPositions = mCFContext->nextPos[iSlice].first;
          clusterer.mPmemory->counters.nPagesSubslice = mCFContext->nextPos[iSlice].second;
        }
        TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);

        using ChargeMapType = decltype(*clustererShadow.mPchargeMap);
        using PeakMapType = decltype(*clustererShadow.mPpeakMap);
        runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPchargeMap, TPCMapMemoryLayout<ChargeMapType>::items() * sizeof(ChargeMapType));
        runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPpeakMap, TPCMapMemoryLayout<PeakMapType>::items() * sizeof(PeakMapType));
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Zeroed Charges");

        if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceZS, lane);
          SynchronizeStream(GetProcessingSettings().nTPCClustererLanes + lane);
        }

        SynchronizeStream(mRec->NStreams() - 1); // Wait for copying to constant memory

        if (mIOPtrs.tpcZS && !mCFContext->nPagesSector[iSlice]) {
          continue;
        }

        if (not mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane), {iSlice}, {}, mIOPtrs.tpcZS == nullptr);
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        } else if (propagateMCLabels) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, mIOPtrs.tpcZS == nullptr);
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }

        if (mIOPtrs.tpcZS) {
          int firstHBF = o2::raw::RDHUtils::getHeartBeatOrbit(*(const RAWDataHeaderGPU*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0]);
          runKernel<GPUTPCCFDecodeZS, GPUTPCCFDecodeZS::decodeZS>(GetGridBlk(doGPU ? clusterer.mPmemory->counters.nPagesSubslice : GPUTrackingInOutZS::NENDPOINTS, lane), {iSlice}, {}, firstHBF);
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }
      }
      for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        SynchronizeStream(lane);
        if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
          CfFragment f = fragment.next();
          int nextSlice = iSlice;
          if (f.isEnd()) {
            nextSlice += GetProcessingSettings().nTPCClustererLanes;
            f = mCFContext->fragmentFirst;
          }
          if (nextSlice < NSLICES && mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
            mCFContext->nextPos[nextSlice] = RunTPCClusterizer_transferZS(nextSlice, f, GetProcessingSettings().nTPCClustererLanes + lane);
          }
        }
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        if (propagateMCLabels || not mIOPtrs.tpcZS) {
          if (clusterer.mPmemory->counters.nPositions == 0) {
            continue;
          }
        }
        if (!mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillFromDigits>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        }
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpDigits, *mDebugFile)) {
          clusterer.DumpChargeMap(*mDebugFile, "Charges");
        }

        if (propagateMCLabels) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillIndexMap>(GetGrid(clusterer.mPmemory->counters.nDigitsInFragment, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {});
        }

        runKernel<GPUTPCCFPeakFinder>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpPeaks, *mDebugFile);

        RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 0, doGPU, lane);
        TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpPeaksCompacted, *mDebugFile);
      }
      for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        SynchronizeStream(lane);
        if (clusterer.mPmemory->counters.nPeaks == 0) {
          continue;
        }
        runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::noiseSuppression>(GetGrid(clusterer.mPmemory->counters.nPeaks, lane), {iSlice}, {});
        runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::updatePeaks>(GetGrid(clusterer.mPmemory->counters.nPeaks, lane), {iSlice}, {});
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaks, *mDebugFile);

        RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 1, doGPU, lane);
        TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaksCompacted, *mDebugFile);
      }
      for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        SynchronizeStream(lane);

        if (fragment.index == 0) {
          runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {nullptr, transferRunning[lane] == 1 ? &mEvents->stream[lane] : nullptr}, clustererShadow.mPclusterInRow, GPUCA_ROW_COUNT * sizeof(*clustererShadow.mPclusterInRow));
          transferRunning[lane] = 2;
        }

        if (clusterer.mPmemory->counters.nClusters == 0) {
          continue;
        }

        runKernel<GPUTPCCFDeconvolution>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Split Charges");

        runKernel<GPUTPCCFClusterizer>(GetGrid(clusterer.mPmemory->counters.nClusters, lane), {iSlice}, {}, 0);
        if (doGPU && propagateMCLabels) {
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mScratchId, lane);
          SynchronizeStream(lane);
          runKernel<GPUTPCCFClusterizer>(GetGrid(clusterer.mPmemory->counters.nClusters, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, 1);
        }
        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Lane %d: Found clusters: digits %u peaks %u clusters %u", lane, (int)clusterer.mPmemory->counters.nPositions, (int)clusterer.mPmemory->counters.nPeaks, (int)clusterer.mPmemory->counters.nClusters);
        }

        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, lane);
        laneHasData[lane] = true;
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpCountedPeaks, *mDebugFile)) {
          clusterer.DumpClusters(*mDebugFile);
        }
      }
    }
    size_t nClsFirst = nClsTotal;
    bool anyLaneHasData = false;
    for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      std::fill(&tmpNative->nClusters[iSlice][0], &tmpNative->nClusters[iSlice][0] + MAXGLOBALPADROW, 0);
      SynchronizeStream(lane);
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
      if (laneHasData[lane]) {
        anyLaneHasData = true;
        if (buildNativeGPU && GetProcessingSettings().tpccfGatherKernel) {
          runKernel<GPUTPCCFGather>(GetGridBlk(GPUCA_ROW_COUNT, mRec->NStreams() - 1), {iSlice}, {}, &mInputsShadow->mPclusterNativeBuffer[nClsTotal]);
        }
        for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
          if (buildNativeGPU) {
            if (!GetProcessingSettings().tpccfGatherKernel) {
              GPUMemCpyAlways(RecoStep::TPCClusterFinding, (void*)&mInputsShadow->mPclusterNativeBuffer[nClsTotal], (const void*)&clustererShadow.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * clusterer.mPclusterInRow[j], mRec->NStreams() - 1, -2);
            }
          } else if (buildNativeHost) {
            GPUMemCpyAlways(RecoStep::TPCClusterFinding, (void*)&mInputsHost->mPclusterNativeOutput[nClsTotal], (const void*)&clustererShadow.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * clusterer.mPclusterInRow[j], mRec->NStreams() - 1, false);
          }
          tmpNative->nClusters[iSlice][j] += clusterer.mPclusterInRow[j];
          nClsTotal += clusterer.mPclusterInRow[j];
        }
        if (transferRunning[lane]) {
          ReleaseEvent(&mEvents->stream[lane]);
        }
        RecordMarker(&mEvents->stream[lane], mRec->NStreams() - 1);
        transferRunning[lane] = 1;
      }

      if (not propagateMCLabels) {
        continue;
      }

      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::setRowOffsets>(GetGrid(GPUCA_ROW_COUNT, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {});
      GPUTPCCFMCLabelFlattener::setGlobalOffsetsAndAllocate(clusterer, mcLinearLabels);
      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::flatten>(GetGrid(GPUCA_ROW_COUNT, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, &mcLinearLabels);
      clusterer.clearMCMemory();
    }
    if (buildNativeHost && buildNativeGPU && anyLaneHasData) {
      if (GetProcessingSettings().delayedOutput) {
        mOutputQueue.emplace_back(outputQueueEntry{(void*)((char*)&mInputsHost->mPclusterNativeOutput[nClsFirst] - (char*)&mInputsHost->mPclusterNativeOutput[0]), &mInputsShadow->mPclusterNativeBuffer[nClsFirst], (nClsTotal - nClsFirst) * sizeof(mInputsHost->mPclusterNativeOutput[nClsFirst]), RecoStep::TPCClusterFinding});
      } else {
        GPUMemCpy(RecoStep::TPCClusterFinding, (void*)&mInputsHost->mPclusterNativeOutput[nClsFirst], (void*)&mInputsShadow->mPclusterNativeBuffer[nClsFirst], (nClsTotal - nClsFirst) * sizeof(mInputsHost->mPclusterNativeOutput[nClsFirst]), mRec->NStreams() - 1, false);
      }
    }
  }
  for (int i = 0; i < GetProcessingSettings().nTPCClustererLanes; i++) {
    if (transferRunning[i]) {
      ReleaseEvent(&mEvents->stream[i]);
    }
  }

  ClusterNativeAccess::ConstMCLabelContainerView* mcLabelsConstView = nullptr;
  if (propagateMCLabels) {
    // TODO: write to buffer directly
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcLabels;
    if (mOutputClusterLabels == nullptr || !mOutputClusterLabels->OutputAllocator) {
      throw std::runtime_error("Cluster MC Label buffer missing");
    }
    ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer* container = reinterpret_cast<ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer*>(mOutputClusterLabels->OutputAllocator(0));

    assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);
    assert(propagateMCLabels ? mcLinearLabels.data.size() >= nClsTotal : true);

    mcLabels.setFrom(mcLinearLabels.header, mcLinearLabels.data);
    mcLabels.flatten_to(container->first);
    container->second = container->first;
    mcLabelsConstView = &container->second;
  }

  if (buildNativeHost && buildNativeGPU && GetProcessingSettings().delayedOutput) {
    mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = nClsTotal;
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput, mOutputClustersNative);
    for (unsigned int i = outputQueueStart; i < mOutputQueue.size(); i++) {
      mOutputQueue[i].dst = (char*)mInputsHost->mPclusterNativeOutput + (size_t)mOutputQueue[i].dst;
    }
  }

  if (buildNativeHost) {
    tmpNative->clustersLinear = mInputsHost->mPclusterNativeOutput;
    tmpNative->clustersMCTruth = mcLabelsConstView;
    tmpNative->setOffsetPtrs();
    mIOPtrs.clustersNative = tmpNative;
  }

  if (mPipelineNotifyCtx) {
    SynchronizeStream(mRec->NStreams() - 2); // Must finish before updating ioPtrs in (global) constant memory
    std::lock_guard<std::mutex> lock(mPipelineNotifyCtx->mutex);
    mPipelineNotifyCtx->ready = true;
    mPipelineNotifyCtx->cond.notify_one();
  }

  if (buildNativeGPU) {
    processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
    mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
    mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
    TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceClusterNativeAccess, 0);
  }
  if (synchronizeOutput) {
    SynchronizeStream(mRec->NStreams() - 1);
  }
  if (buildNativeHost && GetProcessingSettings().debugLevel >= 4) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        std::sort(&mInputsHost->mPclusterNativeOutput[tmpNative->clusterOffset[i][j]], &mInputsHost->mPclusterNativeOutput[tmpNative->clusterOffset[i][j] + tmpNative->nClusters[i][j]]);
      }
    }
  }
  mRec->MemoryScalers()->nTPCHits = nClsTotal;
  mRec->PopNonPersistentMemory(RecoStep::TPCClusterFinding);
  if (mPipelineNotifyCtx) {
    mRec->UnblockStackedMemory();
    mPipelineNotifyCtx = nullptr;
  }

#endif
  return 0;
}

int GPUChainTracking::RunTPCTrackingSlices()
{
  if (mRec->GPUStuck()) {
    GPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
    return (1);
  }

  const auto& threadContext = GetThreadContext();

  int retVal = RunTPCTrackingSlices_internal();
  if (retVal) {
    SynchronizeGPU();
  }
  if (retVal >= 2) {
    ResetHelperThreads(retVal >= 3);
  }
  return (retVal != 0);
}

int GPUChainTracking::RunTPCTrackingSlices_internal()
{
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Slice Tracker");
  }
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCSliceTracking;
  bool doSliceDataOnGPU = processors()->tpcTrackers[0].SliceDataOnGPU();
  if (!param().earlyTpcTransform) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      processors()->tpcTrackers[i].Data().SetClusterData(nullptr, mIOPtrs.clustersNative->nClustersSector[i], mIOPtrs.clustersNative->clusterOffset[i][0]);
      if (doGPU) {
        processorsShadow()->tpcTrackers[i].Data().SetClusterData(nullptr, mIOPtrs.clustersNative->nClustersSector[i], mIOPtrs.clustersNative->clusterOffset[i][0]); // TODO: not needed I think, anyway copied in SetupGPUProcessor
      }
    }
    mRec->MemoryScalers()->nTPCHits = mIOPtrs.clustersNative->nClustersTotal;
  } else {
    int offset = 0;
    for (unsigned int i = 0; i < NSLICES; i++) {
      processors()->tpcTrackers[i].Data().SetClusterData(mIOPtrs.clusterData[i], mIOPtrs.nClusterData[i], offset);
#ifdef HAVE_O2HEADERS
      if (doGPU && GetRecoSteps().isSet(RecoStep::TPCConversion)) {
        processorsShadow()->tpcTrackers[i].Data().SetClusterData(processorsShadow()->tpcConverter.mClusters + processors()->tpcTrackers[i].Data().ClusterIdOffset(), processors()->tpcTrackers[i].NHitsTotal(), processors()->tpcTrackers[i].Data().ClusterIdOffset());
      }
#endif
      offset += mIOPtrs.nClusterData[i];
    }
    mRec->MemoryScalers()->nTPCHits = offset;
  }
  GPUInfo("Event has %u TPC Clusters, %d TRD Tracklets", (unsigned int)mRec->MemoryScalers()->nTPCHits, mIOPtrs.nTRDTracklets);

  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    processors()->tpcTrackers[iSlice].SetMaxData(mIOPtrs); // First iteration to set data sizes
  }
  mRec->ComputeReuseMax(nullptr); // Resolve maximums for shared buffers
  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    SetupGPUProcessor(&processors()->tpcTrackers[iSlice], false); // Prepare custom allocation for 1st stack level
    mRec->AllocateRegisteredMemory(processors()->tpcTrackers[iSlice].MemoryResSliceScratch());
    mRec->AllocateRegisteredMemory(processors()->tpcTrackers[iSlice].MemoryResSliceInput());
  }
  mRec->PushNonPersistentMemory();
  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    SetupGPUProcessor(&processors()->tpcTrackers[iSlice], true);             // Now we allocate
    mRec->ResetRegisteredMemoryPointers(&processors()->tpcTrackers[iSlice]); // TODO: The above call breaks the GPU ptrs to already allocated memory. This fixes them. Should actually be cleaned up at the source.
    processors()->tpcTrackers[iSlice].SetupCommonMemory();
  }

  bool streamInit[GPUCA_MAX_STREAMS] = {false};
  if (doGPU) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processorsShadow()->tpcTrackers[iSlice].GPUParametersConst()->gpumem = (char*)mRec->DeviceMemoryBase();
      // Initialize Startup Constants
      processors()->tpcTrackers[iSlice].GPUParameters()->nextStartHit = (((getKernelProperties<GPUTPCTrackletConstructor, GPUTPCTrackletConstructor::allSlices>().minBlocks * BlockCount()) + NSLICES - 1 - iSlice) / NSLICES) * getKernelProperties<GPUTPCTrackletConstructor, GPUTPCTrackletConstructor::allSlices>().nThreads;
      processorsShadow()->tpcTrackers[iSlice].SetGPUTextureBase(mRec->DeviceMemoryBase());
    }

    if (!doSliceDataOnGPU) {
      RunHelperThreads(&GPUChainTracking::HelperReadEvent, this, NSLICES);
    }
    if (PrepareTextures()) {
      return (2);
    }

    // Copy Tracker Object to GPU Memory
    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Copying Tracker objects to GPU");
    }
    if (PrepareProfile()) {
      return 2;
    }

    WriteToConstantMemory(RecoStep::TPCSliceTracking, (char*)processors()->tpcTrackers - (char*)processors(), processorsShadow()->tpcTrackers, sizeof(GPUTPCTracker) * NSLICES, mRec->NStreams() - 1, &mEvents->init);

    for (int i = 0; i < mRec->NStreams() - 1; i++) {
      streamInit[i] = false;
    }
    streamInit[mRec->NStreams() - 1] = true;
  }
  if (GPUDebug("Initialization (1)", 0)) {
    return (2);
  }

  int streamMap[NSLICES];

  bool error = false;
  GPUCA_OPENMP(parallel for if(!(doGPU || GetProcessingSettings().ompKernels)) num_threads(GetProcessingSettings().ompThreads))
  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    if (mRec->GetDeviceType() == GPUReconstruction::DeviceType::HIP) {
      SynchronizeGPU(); // BUG: Workaround for probable bug in AMD runtime, crashes randomly if not synchronized here
    }
    GPUTPCTracker& trk = processors()->tpcTrackers[iSlice];
    GPUTPCTracker& trkShadow = doGPU ? processorsShadow()->tpcTrackers[iSlice] : trk;
    int useStream = (iSlice % mRec->NStreams());

    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Creating Slice Data (Slice %d)", iSlice);
    }
    if (doSliceDataOnGPU) {
      TransferMemoryResourcesToGPU(RecoStep::TPCSliceTracking, &trk, useStream);
      runKernel<GPUTPCCreateSliceData>(GetGridBlk(GPUCA_ROW_COUNT, useStream), {iSlice}, {nullptr, streamInit[useStream] ? nullptr : &mEvents->init});
      streamInit[useStream] = true;
    } else if (!doGPU || iSlice % (GetProcessingSettings().nDeviceHelperThreads + 1) == 0) {
      if (ReadEvent(iSlice, 0)) {
        GPUError("Error reading event");
        error = 1;
        continue;
      }
    } else {
      if (GetProcessingSettings().debugLevel >= 3) {
        GPUInfo("Waiting for helper thread %d", iSlice % (GetProcessingSettings().nDeviceHelperThreads + 1) - 1);
      }
      while (HelperDone(iSlice % (GetProcessingSettings().nDeviceHelperThreads + 1) - 1) < (int)iSlice) {
        ;
      }
      if (HelperError(iSlice % (GetProcessingSettings().nDeviceHelperThreads + 1) - 1)) {
        error = 1;
        continue;
      }
    }
    if (!doGPU && trk.CheckEmptySlice() && GetProcessingSettings().debugLevel == 0) {
      continue;
    }

    if (GetProcessingSettings().debugLevel >= 6) {
      *mDebugFile << "\n\nReconstruction: Slice " << iSlice << "/" << NSLICES << std::endl;
      if (GetProcessingSettings().debugMask & 1) {
        if (doSliceDataOnGPU) {
          TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &trk, -1, true);
        }
        trk.DumpSliceData(*mDebugFile);
      }
    }

    // Initialize temporary memory where needed
    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Copying Slice Data to GPU and initializing temporary memory");
    }
    if (GetProcessingSettings().keepAllMemory && !doSliceDataOnGPU) {
      memset((void*)trk.Data().HitWeights(), 0, trkShadow.Data().NumberOfHitsPlusAlign() * sizeof(*trkShadow.Data().HitWeights()));
    } else {
      runKernel<GPUMemClean16>(GetGridAutoStep(useStream, RecoStep::TPCSliceTracking), krnlRunRangeNone, {}, trkShadow.Data().HitWeights(), trkShadow.Data().NumberOfHitsPlusAlign() * sizeof(*trkShadow.Data().HitWeights()));
    }

    // Copy Data to GPU Global Memory
    if (!doSliceDataOnGPU) {
      TransferMemoryResourcesToGPU(RecoStep::TPCSliceTracking, &trk, useStream);
    }
    if (GPUDebug("Initialization (3)", useStream)) {
      throw std::runtime_error("memcpy failure");
    }

    runKernel<GPUTPCNeighboursFinder>(GetGridBlk(GPUCA_ROW_COUNT, useStream), {iSlice}, {nullptr, streamInit[useStream] ? nullptr : &mEvents->init});
    streamInit[useStream] = true;

    if (GetProcessingSettings().keepDisplayMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &trk, -1, true);
      memcpy(trk.LinkTmpMemory(), mRec->Res(trk.MemoryResLinks()).Ptr(), mRec->Res(trk.MemoryResLinks()).Size());
      if (GetProcessingSettings().debugMask & 2) {
        trk.DumpLinks(*mDebugFile);
      }
    }

    runKernel<GPUTPCNeighboursCleaner>(GetGridBlk(GPUCA_ROW_COUNT - 2, useStream), {iSlice});
    DoDebugAndDump(RecoStep::TPCSliceTracking, 4, trk, &GPUTPCTracker::DumpLinks, *mDebugFile);

    runKernel<GPUTPCStartHitsFinder>(GetGridBlk(GPUCA_ROW_COUNT - 6, useStream), {iSlice});
#ifdef GPUCA_SORT_STARTHITS_GPU
    if (doGPU) {
      runKernel<GPUTPCStartHitsSorter>(GetGridAuto(useStream), {iSlice});
    }
#endif
    DoDebugAndDump(RecoStep::TPCSliceTracking, 32, trk, &GPUTPCTracker::DumpStartHits, *mDebugFile);

    if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
      trk.UpdateMaxData();
      AllocateRegisteredMemory(trk.MemoryResTracklets());
      AllocateRegisteredMemory(trk.MemoryResOutput());
    }

    if (!(doGPU || GetProcessingSettings().debugLevel >= 1) || GetProcessingSettings().trackletConstructorInPipeline) {
      runKernel<GPUTPCTrackletConstructor>(GetGridAuto(useStream), {iSlice});
      DoDebugAndDump(RecoStep::TPCSliceTracking, 128, trk, &GPUTPCTracker::DumpTrackletHits, *mDebugFile);
      if (GetProcessingSettings().debugMask & 256 && !GetProcessingSettings().comparableDebutOutput) {
        trk.DumpHitWeights(*mDebugFile);
      }
    }

    if (!(doGPU || GetProcessingSettings().debugLevel >= 1) || GetProcessingSettings().trackletSelectorInPipeline) {
      runKernel<GPUTPCTrackletSelector>(GetGridAuto(useStream), {iSlice});
      runKernel<GPUTPCGlobalTrackingCopyNumbers>({1, -ThreadCount(), useStream}, {iSlice}, {}, 1);
      TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, trk.MemoryResCommon(), useStream, &mEvents->slice[iSlice]);
      streamMap[iSlice] = useStream;
      if (GetProcessingSettings().debugLevel >= 3) {
        GPUInfo("Slice %u, Number of tracks: %d", iSlice, *trk.NTracks());
      }
      DoDebugAndDump(RecoStep::TPCSliceTracking, 512, trk, &GPUTPCTracker::DumpTrackHits, *mDebugFile);
    }
  }
  if (error) {
    return (3);
  }

  if (doGPU || GetProcessingSettings().debugLevel >= 1) {
    ReleaseEvent(&mEvents->init);
    if (!doSliceDataOnGPU) {
      WaitForHelperThreads();
    }

    if (!GetProcessingSettings().trackletSelectorInPipeline) {
      if (GetProcessingSettings().trackletConstructorInPipeline) {
        SynchronizeGPU();
      } else {
        for (int i = 0; i < mRec->NStreams(); i++) {
          RecordMarker(&mEvents->stream[i], i);
        }
        runKernel<GPUTPCTrackletConstructor, 1>(GetGridAuto(0), krnlRunRangeNone, {&mEvents->single, mEvents->stream, mRec->NStreams()});
        for (int i = 0; i < mRec->NStreams(); i++) {
          ReleaseEvent(&mEvents->stream[i]);
        }
        SynchronizeEvents(&mEvents->single);
        ReleaseEvent(&mEvents->single);
      }

      if (GetProcessingSettings().debugLevel >= 4) {
        for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
          DoDebugAndDump(RecoStep::TPCSliceTracking, 128, processors()->tpcTrackers[iSlice], &GPUTPCTracker::DumpTrackletHits, *mDebugFile);
        }
      }

      int runSlices = 0;
      int useStream = 0;
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice += runSlices) {
        if (runSlices < GetProcessingSettings().trackletSelectorSlices) {
          runSlices++;
        }
        runSlices = CAMath::Min<int>(runSlices, NSLICES - iSlice);
        if (getKernelProperties<GPUTPCTrackletSelector>().minBlocks * BlockCount() < (unsigned int)runSlices) {
          runSlices = getKernelProperties<GPUTPCTrackletSelector>().minBlocks * BlockCount();
        }

        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Running TPC Tracklet selector (Stream %d, Slice %d to %d)", useStream, iSlice, iSlice + runSlices);
        }
        runKernel<GPUTPCTrackletSelector>(GetGridAuto(useStream), {iSlice, runSlices});
        runKernel<GPUTPCGlobalTrackingCopyNumbers>({1, -ThreadCount(), useStream}, {iSlice}, {}, runSlices);
        for (unsigned int k = iSlice; k < iSlice + runSlices; k++) {
          TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, processors()->tpcTrackers[k].MemoryResCommon(), useStream, &mEvents->slice[k]);
          streamMap[k] = useStream;
        }
        useStream++;
        if (useStream >= mRec->NStreams()) {
          useStream = 0;
        }
      }
    }

    mSliceSelectorReady = 0;

    std::array<bool, NSLICES> transferRunning;
    transferRunning.fill(true);
    if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (doGPU && !(GetRecoStepsGPU() & RecoStep::TPCMerging))) {
      if (param().rec.GlobalTracking) {
        mWriteOutputDone.fill(0);
      }
      RunHelperThreads(&GPUChainTracking::HelperOutput, this, NSLICES);

      unsigned int tmpSlice = 0;
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Transfering Tracks from GPU to Host");
        }

        if (tmpSlice == iSlice) {
          SynchronizeEvents(&mEvents->slice[iSlice]);
        }
        while (tmpSlice < NSLICES && (tmpSlice == iSlice || IsEventDone(&mEvents->slice[tmpSlice]))) {
          ReleaseEvent(&mEvents->slice[tmpSlice]);
          if (*processors()->tpcTrackers[tmpSlice].NTracks() > 0) {
            TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, processors()->tpcTrackers[tmpSlice].MemoryResOutput(), streamMap[tmpSlice], &mEvents->slice[tmpSlice]);
          } else {
            transferRunning[tmpSlice] = false;
          }
          tmpSlice++;
        }

        if (GetProcessingSettings().keepAllMemory) {
          TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &processors()->tpcTrackers[iSlice], -1, true);
          if (!GetProcessingSettings().trackletConstructorInPipeline) {
            if (GetProcessingSettings().debugMask & 256 && !GetProcessingSettings().comparableDebutOutput) {
              processors()->tpcTrackers[iSlice].DumpHitWeights(*mDebugFile);
            }
          }
          if (!GetProcessingSettings().trackletSelectorInPipeline) {
            if (GetProcessingSettings().debugMask & 512) {
              processors()->tpcTrackers[iSlice].DumpTrackHits(*mDebugFile);
            }
          }
        }

        if (transferRunning[iSlice]) {
          SynchronizeEvents(&mEvents->slice[iSlice]);
        }
        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Tracks Transfered: %d / %d", *processors()->tpcTrackers[iSlice].NTracks(), *processors()->tpcTrackers[iSlice].NTrackHits());
        }

        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Data ready for slice %d, helper thread %d", iSlice, iSlice % (GetProcessingSettings().nDeviceHelperThreads + 1));
        }
        mSliceSelectorReady = iSlice;

        if (param().rec.GlobalTracking) {
          for (unsigned int tmpSlice2a = 0; tmpSlice2a <= iSlice; tmpSlice2a += GetProcessingSettings().nDeviceHelperThreads + 1) {
            unsigned int tmpSlice2 = GPUTPCGlobalTracking::GlobalTrackingSliceOrder(tmpSlice2a);
            unsigned int sliceLeft, sliceRight;
            GPUTPCGlobalTracking::GlobalTrackingSliceLeftRight(tmpSlice2, sliceLeft, sliceRight);

            if (tmpSlice2 <= iSlice && sliceLeft <= iSlice && sliceRight <= iSlice && mWriteOutputDone[tmpSlice2] == 0) {
              GlobalTracking(tmpSlice2, 0);
              WriteOutput(tmpSlice2, 0);
              mWriteOutputDone[tmpSlice2] = 1;
            }
          }
        } else {
          if (iSlice % (GetProcessingSettings().nDeviceHelperThreads + 1) == 0) {
            WriteOutput(iSlice, 0);
          }
        }
      }
      WaitForHelperThreads();
    }
    if (!(GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) && param().rec.GlobalTracking) {
      std::vector<bool> blocking(NSLICES * mRec->NStreams());
      for (int i = 0; i < NSLICES; i++) {
        for (int j = 0; j < mRec->NStreams(); j++) {
          blocking[i * mRec->NStreams() + j] = i % mRec->NStreams() == j;
        }
      }
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
        unsigned int tmpSlice = GPUTPCGlobalTracking::GlobalTrackingSliceOrder(iSlice);
        if (!((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (doGPU && !(GetRecoStepsGPU() & RecoStep::TPCMerging)))) {
          unsigned int sliceLeft, sliceRight;
          GPUTPCGlobalTracking::GlobalTrackingSliceLeftRight(tmpSlice, sliceLeft, sliceRight);
          if (!blocking[tmpSlice * mRec->NStreams() + sliceLeft % mRec->NStreams()]) {
            StreamWaitForEvents(tmpSlice % mRec->NStreams(), &mEvents->slice[sliceLeft]);
            blocking[tmpSlice * mRec->NStreams() + sliceLeft % mRec->NStreams()] = true;
          }
          if (!blocking[tmpSlice * mRec->NStreams() + sliceRight % mRec->NStreams()]) {
            StreamWaitForEvents(tmpSlice % mRec->NStreams(), &mEvents->slice[sliceRight]);
            blocking[tmpSlice * mRec->NStreams() + sliceRight % mRec->NStreams()] = true;
          }
        }
        GlobalTracking(tmpSlice, 0, !GetProcessingSettings().fullMergerOnGPU);
      }
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (transferRunning[iSlice]) {
        ReleaseEvent(&mEvents->slice[iSlice]);
      }
    }
  } else {
    mSliceSelectorReady = NSLICES;
    GPUCA_OPENMP(parallel for if(!(doGPU || GetProcessingSettings().ompKernels)) num_threads(GetProcessingSettings().ompThreads))
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (param().rec.GlobalTracking) {
        GlobalTracking(iSlice, 0);
      }
      if (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) {
        WriteOutput(iSlice, 0);
      }
    }
  }

  if (param().rec.GlobalTracking && GetProcessingSettings().debugLevel >= 3) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      GPUInfo("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d", iSlice,
              processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTracks, processors()->tpcTrackers[iSlice].CommonMemory()->nTracks, processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTrackHits, processors()->tpcTrackers[iSlice].CommonMemory()->nTrackHits);
    }
  }

  if (GetProcessingSettings().debugMask & 1024 && !GetProcessingSettings().comparableDebutOutput) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      processors()->tpcTrackers[i].DumpOutput(*mDebugFile);
    }
  }

  if (DoProfile()) {
    return (1);
  }
  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.nSliceTracks[i] = *processors()->tpcTrackers[i].NTracks();
    mIOPtrs.sliceTracks[i] = processors()->tpcTrackers[i].Tracks();
    mIOPtrs.nSliceClusters[i] = *processors()->tpcTrackers[i].NTrackHits();
    mIOPtrs.sliceClusters[i] = processors()->tpcTrackers[i].TrackHits();
    if (GetProcessingSettings().keepDisplayMemory && !GetProcessingSettings().keepAllMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &processors()->tpcTrackers[i], -1, true);
    }
  }
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Slice Tracker finished");
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCSliceTracking);
  return 0;
}

void GPUChainTracking::RunTPCTrackingMerger_MergeBorderTracks(char withinSlice, char mergeMode, GPUReconstruction::krnlDeviceType deviceType)
{
  unsigned int n = withinSlice == -1 ? NSLICES / 2 : NSLICES;
  bool doGPUall = GetRecoStepsGPU() & RecoStep::TPCMerging && GetProcessingSettings().fullMergerOnGPU;
  if (GetProcessingSettings().alternateBorderSort && (!mRec->IsGPU() || doGPUall)) {
    GPUTPCGMMerger& Merger = processors()->tpcMerger;
    GPUTPCGMMerger& MergerShadow = doGPUall ? processorsShadow()->tpcMerger : Merger;
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->init);
    RecordMarker(&mEvents->single, 0);
    for (unsigned int i = 0; i < n; i++) {
      int stream = i % mRec->NStreams();
      runKernel<GPUTPCGMMergerMergeBorders, 0>(GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, stream && i < (unsigned int)mRec->NStreams() ? &mEvents->single : nullptr}, i, withinSlice, mergeMode);
    }
    ReleaseEvent(&mEvents->single);
    SynchronizeEvents(&mEvents->init);
    ReleaseEvent(&mEvents->init);
    for (unsigned int i = 0; i < n; i++) {
      int stream = i % mRec->NStreams();
      int n1, n2;
      GPUTPCGMBorderTrack *b1, *b2;
      int jSlice;
      Merger.MergeBorderTracksSetup(n1, n2, b1, b2, jSlice, i, withinSlice, mergeMode);
      gputpcgmmergertypes::GPUTPCGMBorderRange* range1 = MergerShadow.BorderRange(i);
      gputpcgmmergertypes::GPUTPCGMBorderRange* range2 = MergerShadow.BorderRange(jSlice) + *processors()->tpcTrackers[jSlice].NTracks();
      runKernel<GPUTPCGMMergerMergeBorders, 3>({1, -WarpSize(), stream, deviceType}, krnlRunRangeNone, krnlEventNone, range1, n1, 0);
      runKernel<GPUTPCGMMergerMergeBorders, 3>({1, -WarpSize(), stream, deviceType}, krnlRunRangeNone, krnlEventNone, range2, n2, 1);
      deviceEvent** e = nullptr;
      int ne = 0;
      if (i == n - 1) { // Synchronize all execution on stream 0 with the last kernel
        ne = std::min<int>(n, mRec->NStreams());
        for (int j = 1; j < ne; j++) {
          RecordMarker(&mEvents->slice[j], j);
        }
        e = &mEvents->slice[1];
        ne--;
        stream = 0;
      }
      runKernel<GPUTPCGMMergerMergeBorders, 2>(GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, e, ne}, i, withinSlice, mergeMode);
    }
  } else {
    for (unsigned int i = 0; i < n; i++) {
      runKernel<GPUTPCGMMergerMergeBorders, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i, withinSlice, mergeMode);
    }
    runKernel<GPUTPCGMMergerMergeBorders, 1>({2 * n, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, 0, withinSlice, mergeMode);
    for (unsigned int i = 0; i < n; i++) {
      runKernel<GPUTPCGMMergerMergeBorders, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i, withinSlice, mergeMode);
    }
  }
  mRec->ReturnVolatileDeviceMemory();
}

void GPUChainTracking::RunTPCTrackingMerger_Resolve(char useOrigTrackParam, char mergeAll, GPUReconstruction::krnlDeviceType deviceType)
{
  runKernel<GPUTPCGMMergerResolve, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerResolve, 1>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerResolve, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerResolve, 3>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerResolve, 4>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, useOrigTrackParam, mergeAll);
}

int GPUChainTracking::RunTPCTrackingMerger(bool synchronizeOutput)
{
  if (GetProcessingSettings().debugLevel >= 6 && GetProcessingSettings().comparableDebutOutput && param().rec.mergerReadFromTrackerDirectly) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      GPUTPCTracker& trk = processors()->tpcTrackers[i];
      TransferMemoryResourcesToHost(RecoStep::NoRecoStep, &trk);
      auto sorter = [&trk](GPUTPCTrack& trk1, GPUTPCTrack& trk2) {
        if (trk1.NHits() == trk2.NHits()) {
          return trk1.Param().Y() > trk2.Param().Y();
        }
        return trk1.NHits() > trk2.NHits();
      };
      std::sort(trk.Tracks(), trk.Tracks() + trk.CommonMemory()->nLocalTracks, sorter);
      std::sort(trk.Tracks() + trk.CommonMemory()->nLocalTracks, trk.Tracks() + *trk.NTracks(), sorter);
      TransferMemoryResourcesToGPU(RecoStep::NoRecoStep, &trk, 0);
    }
  }
  mRec->PushNonPersistentMemory();
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCMerging;
  bool doGPUall = doGPU && GetProcessingSettings().fullMergerOnGPU;
  GPUReconstruction::krnlDeviceType deviceType = doGPUall ? GPUReconstruction::krnlDeviceType::Auto : GPUReconstruction::krnlDeviceType::CPU;
  unsigned int numBlocks = (!mRec->IsGPU() || doGPUall) ? BlockCount() : 1;
  GPUTPCGMMerger& Merger = processors()->tpcMerger;
  GPUTPCGMMerger& MergerShadow = doGPU ? processorsShadow()->tpcMerger : Merger;
  GPUTPCGMMerger& MergerShadowAll = doGPUall ? processorsShadow()->tpcMerger : Merger;
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Merger");
  }
  const auto& threadContext = GetThreadContext();

  SynchronizeGPU(); // Need to know the full number of slice tracks
  SetupGPUProcessor(&Merger, true);
  AllocateRegisteredMemory(Merger.MemoryResOutput(), mOutputTPCTracks);

  if (Merger.CheckSlices()) {
    return 1;
  }

  memset(Merger.Memory(), 0, sizeof(*Merger.Memory()));
  WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
  if (doGPUall) {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }

  for (unsigned int i = 0; i < NSLICES; i++) {
    runKernel<GPUTPCGMMergerUnpackResetIds>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i);
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, i);
    runKernel<GPUTPCGMMergerSliceRefit>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i);
  }
  for (unsigned int i = 0; i < NSLICES; i++) {
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, NSLICES + i);
    runKernel<GPUTPCGMMergerUnpackGlobal>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, i);
  }
  runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, 2 * NSLICES);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpSliceTracks, *mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, 0);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeWithinPrepare>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  RunTPCTrackingMerger_MergeBorderTracks(1, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergedWithinSlices, *mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, 0);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 2, 3, 0);
  RunTPCTrackingMerger_MergeBorderTracks(0, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 1, 0);
  RunTPCTrackingMerger_MergeBorderTracks(0, 0, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 1, 1);
  RunTPCTrackingMerger_MergeBorderTracks(0, -1, deviceType);
  RunTPCTrackingMerger_Resolve(0, 1, deviceType);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergedBetweenSlices, *mDebugFile);

  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));

  runKernel<GPUTPCGMMergerLinkGlobalTracks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerCollect>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpCollected, *mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone, 1);
  RunTPCTrackingMerger_MergeBorderTracks(-1, 1, deviceType);
  RunTPCTrackingMerger_MergeBorderTracks(-1, 2, deviceType);
  runKernel<GPUTPCGMMergerMergeCE>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergeCE, *mDebugFile);
  int waitForTransfer = 0;
  if (doGPUall) {
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, &mEvents->single);
    waitForTransfer = 1;
  }

  if (GetProcessingSettings().mergerSortTracks) {
    runKernel<GPUTPCGMMergerSortTracksPrepare>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    CondWaitEvent(waitForTransfer, &mEvents->single);
    runKernel<GPUTPCGMMergerSortTracks>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  }

  unsigned int maxId = param().rec.NonConsecutiveIDs ? Merger.Memory()->nOutputTrackClusters : Merger.NMaxClusters();
  if (maxId > Merger.NMaxClusters()) {
    throw std::runtime_error("mNMaxClusters too small");
  }
  if (!param().rec.NonConsecutiveIDs) {
    unsigned int* sharedCount = (unsigned int*)MergerShadowAll.TmpMem() + CAMath::nextMultipleOf<4>(Merger.Memory()->nOutputTracks);
    runKernel<GPUMemClean16>({numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, sharedCount, maxId * sizeof(*sharedCount));
    runKernel<GPUMemClean16>({numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.ClusterAttachment(), maxId * sizeof(*MergerShadowAll.ClusterAttachment()));
    runKernel<GPUTPCGMMergerPrepareClusters, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    CondWaitEvent(waitForTransfer, &mEvents->single);
    runKernel<GPUTPCGMMergerSortTracksQPt>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerPrepareClusters, 1>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerPrepareClusters, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  }

  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpFitPrepare, *mDebugFile);

  if (doGPUall) {
    CondWaitEvent(waitForTransfer, &mEvents->single);
    if (waitForTransfer) {
      ReleaseEvent(&mEvents->single);
    }
  } else if (doGPU) {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }

  if (GetProcessingSettings().delayedOutput) {
    for (unsigned int i = 0; i < mOutputQueue.size(); i++) {
      GPUMemCpy(mOutputQueue[i].step, mOutputQueue[i].dst, mOutputQueue[i].src, mOutputQueue[i].size, mRec->NStreams() - 2, false);
    }
    mOutputQueue.clear();
  }

  runKernel<GPUTPCGMMergerTrackFit>(doGPU ? GetGrid(Merger.NOutputTracks(), 0) : GetGridAuto(0), krnlRunRangeNone, krnlEventNone, GetProcessingSettings().mergerSortTracks ? 1 : 0);
  if (param().rec.retryRefit == 1) {
    runKernel<GPUTPCGMMergerTrackFit>(GetGridAuto(0), krnlRunRangeNone, krnlEventNone, -1);
  }
  if (param().rec.loopInterpolationInExtraPass) {
    runKernel<GPUTPCGMMergerFollowLoopers>(GetGridAuto(0), krnlRunRangeNone, krnlEventNone);
  }
  if (doGPU && !doGPUall) {
    TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger, 0);
    SynchronizeStream(0);
  }

  DoDebugAndDump(RecoStep::TPCMerging, 0, Merger, &GPUTPCGMMerger::DumpRefit, *mDebugFile);
  runKernel<GPUTPCGMMergerFinalize, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  if (!param().rec.NonConsecutiveIDs) {
    runKernel<GPUTPCGMMergerFinalize, 1>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerFinalize, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  }
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpFinal, *mDebugFile);

  if (doGPUall) {
    RecordMarker(&mEvents->single, 0);
    if (!GetProcessingSettings().fullMergerOnGPU) {
      TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResOutput(), mRec->NStreams() - 2, nullptr, &mEvents->single);
    } else {
      GPUMemCpy(RecoStep::TPCMerging, Merger.OutputTracks(), MergerShadowAll.OutputTracks(), Merger.NOutputTracks() * sizeof(*Merger.OutputTracks()), mRec->NStreams() - 2, 0, nullptr, &mEvents->single);
      GPUMemCpy(RecoStep::TPCMerging, Merger.Clusters(), MergerShadowAll.Clusters(), Merger.NOutputTrackClusters() * sizeof(*Merger.Clusters()), mRec->NStreams() - 2, 0);
      if (param().earlyTpcTransform) {
        GPUMemCpy(RecoStep::TPCMerging, Merger.ClustersXYZ(), MergerShadowAll.ClustersXYZ(), Merger.NOutputTrackClusters() * sizeof(*Merger.ClustersXYZ()), mRec->NStreams() - 2, 0);
      }
      GPUMemCpy(RecoStep::TPCMerging, Merger.ClusterAttachment(), MergerShadowAll.ClusterAttachment(), Merger.NMaxClusters() * sizeof(*Merger.ClusterAttachment()), mRec->NStreams() - 2, 0);
    }
    ReleaseEvent(&mEvents->single);
    if (synchronizeOutput) {
      SynchronizeStream(mRec->NStreams() - 2);
    }
  } else {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }
  if (GetProcessingSettings().keepDisplayMemory && !GetProcessingSettings().keepAllMemory) {
    TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger, -1, true);
  }
  mRec->ReturnVolatileDeviceMemory();

  mIOPtrs.mergedTracks = Merger.OutputTracks();
  mIOPtrs.nMergedTracks = Merger.NOutputTracks();
  mIOPtrs.mergedTrackHits = Merger.Clusters();
  mIOPtrs.nMergedTrackHits = Merger.NOutputTrackClusters();
  mIOPtrs.mergedTrackHitAttachment = Merger.ClusterAttachment();

  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Merger Finished (output clusters %d / input clusters %d)", Merger.NOutputTrackClusters(), Merger.NClusters());
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCMerging);
  return 0;
}

int GPUChainTracking::RunTPCCompression()
{
#ifdef HAVE_O2HEADERS
  mRec->PushNonPersistentMemory();
  RecoStep myStep = RecoStep::TPCCompression;
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCCompression;
  GPUTPCCompression& Compressor = processors()->tpcCompressor;
  GPUTPCCompression& CompressorShadow = doGPU ? processorsShadow()->tpcCompressor : Compressor;
  const auto& threadContext = GetThreadContext();
  if (mPipelineFinalizationCtx && GetProcessingSettings().doublePipelineClusterizer) {
    RecordMarker(&mEvents->single, 0);
  }
  Compressor.mNMaxClusterSliceRow = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      if (mIOPtrs.clustersNative->nClusters[i][j] > Compressor.mNMaxClusterSliceRow) {
        Compressor.mNMaxClusterSliceRow = mIOPtrs.clustersNative->nClusters[i][j];
      }
    }
  }
  SetupGPUProcessor(&Compressor, true);
  new (Compressor.mMemory) GPUTPCCompression::memory;

  WriteToConstantMemory(myStep, (char*)&processors()->tpcCompressor - (char*)processors(), &CompressorShadow, sizeof(CompressorShadow), 0);
  TransferMemoryResourcesToGPU(myStep, &Compressor, 0);
  runKernel<GPUMemClean16>(GetGridAutoStep(0, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone, CompressorShadow.mClusterStatus, Compressor.mMaxClusters * sizeof(CompressorShadow.mClusterStatus[0]));
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step0attached>(GetGridAuto(0), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step1unattached>(GetGridAuto(0), krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(myStep, &Compressor, 0);
#ifdef GPUCA_TPC_GEOMETRY_O2
  if (mPipelineFinalizationCtx && GetProcessingSettings().doublePipelineClusterizer) {
    SynchronizeEvents(&mEvents->single);
    ReleaseEvent(&mEvents->single);
    ((GPUChainTracking*)GetNextChainInQueue())->RunTPCClusterizer_prepare(false);
    ((GPUChainTracking*)GetNextChainInQueue())->mCFContext->ptrClusterNativeSave = processorsShadow()->ioPtrs.clustersNative;
  }
#endif
  SynchronizeStream(0);
  o2::tpc::CompressedClusters* O = Compressor.mOutput;
  memset((void*)O, 0, sizeof(*O));
  O->nTracks = Compressor.mMemory->nStoredTracks;
  O->nAttachedClusters = Compressor.mMemory->nStoredAttachedClusters;
  O->nUnattachedClusters = Compressor.mMemory->nStoredUnattachedClusters;
  O->nAttachedClustersReduced = O->nAttachedClusters - O->nTracks;
  O->nSliceRows = NSLICES * GPUCA_ROW_COUNT;
  O->nComppressionModes = param().rec.tpcCompressionModes;
  size_t outputSize = AllocateRegisteredMemory(Compressor.mMemoryResOutputHost, mOutputCompressedClusters);
  Compressor.mOutputFlat->set(outputSize, *Compressor.mOutput);
  const o2::tpc::CompressedClustersPtrs* P = nullptr;
  HighResTimer* gatherTimer = nullptr;
  int outputStream = 0;
  if (ProcessingSettings().doublePipeline) {
    SynchronizeStream(mRec->NStreams() - 2); // Synchronize output copies running in parallel from memory that might be released, only the following async copy from stacked memory is safe after the chain finishes.
    outputStream = mRec->NStreams() - 2;
  }

  if (ProcessingSettings().tpcCompressionGatherMode == 2) {
    void* devicePtr = mRec->getGPUPointer(Compressor.mOutputFlat);
    if (devicePtr != Compressor.mOutputFlat) {
      CompressedClustersPtrs& ptrs = *Compressor.mOutput; // We need to update the ptrs with the gpu-mapped version of the host address space
      for (unsigned int i = 0; i < sizeof(ptrs) / sizeof(void*); i++) {
        reinterpret_cast<char**>(&ptrs)[i] = reinterpret_cast<char**>(&ptrs)[i] + (reinterpret_cast<char*>(devicePtr) - reinterpret_cast<char*>(Compressor.mOutputFlat));
      }
    }
    TransferMemoryResourcesToGPU(myStep, &Compressor, outputStream);
    unsigned int nBlocks = 2;
    switch (ProcessingSettings().tpcCompressionGatherModeKernel) {
      case 0:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::unbuffered>(GetGridBlkStep(nBlocks, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::unbuffered>(RecoStep::TPCCompression, 0, outputSize);
        break;
      case 1:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered32>(GetGridBlkStep(nBlocks, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered32>(RecoStep::TPCCompression, 0, outputSize);
        break;
      case 2:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered64>(GetGridBlkStep(nBlocks, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered64>(RecoStep::TPCCompression, 0, outputSize);
        break;
      case 3:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered128>(GetGridBlkStep(nBlocks, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered128>(RecoStep::TPCCompression, 0, outputSize);
        break;
      default:
        GPUError("Invalid compression kernel selected.");
        return 1;
    }

  } else {
    char direction = 0;
    if (ProcessingSettings().tpcCompressionGatherMode == 0) {
      P = &CompressorShadow.mPtrs;
    } else if (ProcessingSettings().tpcCompressionGatherMode == 1) {
      P = &Compressor.mPtrs;
      direction = -1;
      gatherTimer = &getTimer<GPUTPCCompressionKernels>("GPUTPCCompression_GatherOnCPU", 0);
      gatherTimer->Start();
    }
    GPUMemCpyAlways(myStep, O->nSliceRowClusters, P->nSliceRowClusters, NSLICES * GPUCA_ROW_COUNT * sizeof(O->nSliceRowClusters[0]), outputStream, direction);
    GPUMemCpyAlways(myStep, O->nTrackClusters, P->nTrackClusters, O->nTracks * sizeof(O->nTrackClusters[0]), outputStream, direction);
    SynchronizeStream(outputStream);
    unsigned int offset = 0;
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        GPUMemCpyAlways(myStep, O->qTotU + offset, P->qTotU + mIOPtrs.clustersNative->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->qTotU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->qMaxU + offset, P->qMaxU + mIOPtrs.clustersNative->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->qMaxU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->flagsU + offset, P->flagsU + mIOPtrs.clustersNative->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->flagsU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->padDiffU + offset, P->padDiffU + mIOPtrs.clustersNative->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->padDiffU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->timeDiffU + offset, P->timeDiffU + mIOPtrs.clustersNative->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->timeDiffU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->sigmaPadU + offset, P->sigmaPadU + mIOPtrs.clustersNative->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->sigmaPadU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->sigmaTimeU + offset, P->sigmaTimeU + mIOPtrs.clustersNative->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->sigmaTimeU[0]), outputStream, direction);
        offset += O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
      }
    }
    offset = 0;
    for (unsigned int i = 0; i < O->nTracks; i++) {
      GPUMemCpyAlways(myStep, O->qTotA + offset, P->qTotA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->qTotA[0]), outputStream, direction);
      GPUMemCpyAlways(myStep, O->qMaxA + offset, P->qMaxA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->qMaxA[0]), outputStream, direction);
      GPUMemCpyAlways(myStep, O->flagsA + offset, P->flagsA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->flagsA[0]), outputStream, direction);
      GPUMemCpyAlways(myStep, O->sigmaPadA + offset, P->sigmaPadA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->sigmaPadA[0]), outputStream, direction);
      GPUMemCpyAlways(myStep, O->sigmaTimeA + offset, P->sigmaTimeA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->sigmaTimeA[0]), outputStream, direction);

      // First index stored with track
      GPUMemCpyAlways(myStep, O->rowDiffA + offset - i, P->rowDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (O->nTrackClusters[i] - 1) * sizeof(O->rowDiffA[0]), outputStream, direction);
      GPUMemCpyAlways(myStep, O->sliceLegDiffA + offset - i, P->sliceLegDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (O->nTrackClusters[i] - 1) * sizeof(O->sliceLegDiffA[0]), outputStream, direction);
      GPUMemCpyAlways(myStep, O->padResA + offset - i, P->padResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (O->nTrackClusters[i] - 1) * sizeof(O->padResA[0]), outputStream, direction);
      GPUMemCpyAlways(myStep, O->timeResA + offset - i, P->timeResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (O->nTrackClusters[i] - 1) * sizeof(O->timeResA[0]), outputStream, direction);
      offset += O->nTrackClusters[i];
    }
    GPUMemCpyAlways(myStep, O->qPtA, P->qPtA, O->nTracks * sizeof(O->qPtA[0]), outputStream, direction);
    GPUMemCpyAlways(myStep, O->rowA, P->rowA, O->nTracks * sizeof(O->rowA[0]), outputStream, direction);
    GPUMemCpyAlways(myStep, O->sliceA, P->sliceA, O->nTracks * sizeof(O->sliceA[0]), outputStream, direction);
    GPUMemCpyAlways(myStep, O->timeA, P->timeA, O->nTracks * sizeof(O->timeA[0]), outputStream, direction);
    GPUMemCpyAlways(myStep, O->padA, P->padA, O->nTracks * sizeof(O->padA[0]), outputStream, direction);
  }
  if (ProcessingSettings().tpcCompressionGatherMode == 1) {
    gatherTimer->Stop();
  }
  mIOPtrs.tpcCompressedClusters = Compressor.mOutputFlat;
  if (mPipelineFinalizationCtx == nullptr) {
    SynchronizeStream(outputStream);
  } else {
    ((GPUChainTracking*)GetNextChainInQueue())->mRec->BlockStackedMemory(mRec);
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCCompression);
#endif
  return 0;
}

int GPUChainTracking::RunTPCDecompression()
{
#ifdef HAVE_O2HEADERS
  const auto& threadContext = GetThreadContext();
  TPCClusterDecompressor decomp;
  auto allocator = [this](size_t size) {
    this->mInputsHost->mNClusterNative = this->mInputsShadow->mNClusterNative = size;
    this->AllocateRegisteredMemory(this->mInputsHost->mResourceClusterNativeOutput, this->mOutputClustersNative);
    return this->mInputsHost->mPclusterNativeOutput;
  };
  auto& gatherTimer = getTimer<TPCClusterDecompressor>("TPCDecompression", 0);
  gatherTimer.Start();
  if (decomp.decompress(mIOPtrs.tpcCompressedClusters, *mClusterNativeAccess, allocator, param())) {
    GPUError("Error decompressing clusters");
    return 1;
  }
  gatherTimer.Stop();
  mIOPtrs.clustersNative = mClusterNativeAccess.get();
  if (mRec->IsGPU()) {
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
    processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
    WriteToConstantMemory(RecoStep::TPCDecompression, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
    mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
    mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
    GPUMemCpy(RecoStep::TPCDecompression, mInputsShadow->mPclusterNativeBuffer, mIOPtrs.clustersNative->clustersLinear, sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * mIOPtrs.clustersNative->nClustersTotal, 0, true);
    TransferMemoryResourceLinkToGPU(RecoStep::TPCDecompression, mInputsHost->mResourceClusterNativeAccess, 0);
    SynchronizeStream(0);
  }
#endif
  return 0;
}

int GPUChainTracking::RunTRDTracking()
{
  if (!processors()->trdTracker.IsInitialized()) {
    return 1;
  }

  GPUTRDTrackerGPU& Tracker = processors()->trdTracker;
  Tracker.Reset();
  if (mIOPtrs.nTRDTracklets == 0) {
    return 0;
  }

  mRec->PushNonPersistentMemory();
  SetupGPUProcessor(&Tracker, true);

  for (unsigned int iTracklet = 0; iTracklet < mIOPtrs.nTRDTracklets; ++iTracklet) {
    if (Tracker.LoadTracklet(mIOPtrs.trdTracklets[iTracklet], mIOPtrs.trdTrackletsMC ? mIOPtrs.trdTrackletsMC[iTracklet].mLabel : nullptr)) {
      return 1;
    }
  }

  for (unsigned int i = 0; i < mIOPtrs.nMergedTracks; i++) {
    const GPUTPCGMMergedTrack& trk = mIOPtrs.mergedTracks[i];
    if (!trk.OK()) {
      continue;
    }
    if (trk.Looper()) {
      continue;
    }

    const GPUTRDTrackGPU& trktrd = param().rec.NWaysOuter ? (GPUTRDTrackGPU)trk.OuterParam() : (GPUTRDTrackGPU)trk;

    if (Tracker.LoadTrack(trktrd, -1, nullptr, -1, i)) {
      return 1;
    }
  }

  Tracker.DoTracking(this);

  mIOPtrs.nTRDTracks = Tracker.NTracks();
  mIOPtrs.trdTracks = Tracker.Tracks();
  mRec->PopNonPersistentMemory(RecoStep::TRDTracking);

  return 0;
}

int GPUChainTracking::DoTRDGPUTracking()
{
#ifdef HAVE_O2HEADERS
  bool doGPU = GetRecoStepsGPU() & RecoStep::TRDTracking;
  GPUTRDTrackerGPU& Tracker = processors()->trdTracker;
  GPUTRDTrackerGPU& TrackerShadow = doGPU ? processorsShadow()->trdTracker : Tracker;

  const auto& threadContext = GetThreadContext();
  SetupGPUProcessor(&Tracker, false);
  TrackerShadow.OverrideGPUGeometry(reinterpret_cast<GPUTRDGeometry*>(mFlatObjectsDevice.mCalibObjects.trdGeometry));

  WriteToConstantMemory(RecoStep::TRDTracking, (char*)&processors()->trdTracker - (char*)processors(), &TrackerShadow, sizeof(TrackerShadow), 0);
  TransferMemoryResourcesToGPU(RecoStep::TRDTracking, &Tracker, 0);

  runKernel<GPUTRDTrackerKernels>(GetGridAuto(0), krnlRunRangeNone);
  TransferMemoryResourcesToHost(RecoStep::TRDTracking, &Tracker, 0);
  SynchronizeStream(0);

  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("GPU TRD tracker Finished");
  }
#endif
  return (0);
}

int GPUChainTracking::RunChain()
{
  const auto threadContext = GetThreadContext();
  if (GetProcessingSettings().runCompressionStatistics && mCompressionStatistics == nullptr) {
    mCompressionStatistics.reset(new GPUTPCClusterStatistics);
  }
  const bool needQA = GPUQA::QAAvailable() && (GetProcessingSettings().runQA || (GetProcessingSettings().eventDisplay && mIOPtrs.nMCInfosTPC));
  if (needQA && mQA->IsInitialized() == false) {
    if (mQA->InitQA()) {
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
#ifdef GPUCA_STANDALONE
  mRec->PrepareEvent();
#else
  try {
    mRec->PrepareEvent();
  } catch (const std::bad_alloc& e) {
    GPUError("Memory Allocation Error");
    return (1);
  }
#endif
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

  if (!GetProcessingSettings().doublePipeline) { // Synchronize with output copies running asynchronously
    SynchronizeStream(mRec->NStreams() - 2);
  }

  if (CheckErrorCodes()) {
    return 1;
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
