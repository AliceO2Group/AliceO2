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

#include "GPUChainTracking.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCSliceOutTrack.h"
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

#ifdef HAVE_O2HEADERS
#include "GPUTPCClusterStatistics.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#include "TPCFastTransform.h"

#include "utils/linux_helpers.h"

using namespace GPUCA_NAMESPACE::gpu;

#include "GPUO2DataTypes.h"

using namespace o2::tpc;
using namespace o2::trd;

static constexpr unsigned int DUMP_HEADER_SIZE = 4;
static constexpr char DUMP_HEADER[DUMP_HEADER_SIZE + 1] = "CAv1";

GPUChainTracking::GPUChainTracking(GPUReconstruction* rec, unsigned int maxTPCHits, unsigned int maxTRDTracklets) : GPUChain(rec), mIOPtrs(processors()->ioPtrs), mClusterNativeAccess(new ClusterNativeAccess), mMaxTPCHits(maxTPCHits), mMaxTRDTracklets(maxTRDTracklets)
{
  mFlatObjectsShadow.mChainTracking = this;
  mFlatObjectsDevice.mChainTracking = this;
}

GPUChainTracking::~GPUChainTracking() = default;

void GPUChainTracking::RegisterPermanentMemoryAndProcessors()
{
  mFlatObjectsShadow.InitGPUProcessor(mRec, GPUProcessor::PROCESSOR_TYPE_SLAVE);
  mFlatObjectsDevice.InitGPUProcessor(mRec, GPUProcessor::PROCESSOR_TYPE_DEVICE, &mFlatObjectsShadow);
  mFlatObjectsShadow.mMemoryResFlat = mRec->RegisterMemoryAllocation(&mFlatObjectsShadow, &GPUTrackingFlatObjects::SetPointersFlatObjects, GPUMemoryResource::MEMORY_PERMANENT, "Processors");

  if (GetRecoSteps() & RecoStep::TPCSliceTracking) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      mRec->RegisterGPUProcessor(&processors()->tpcTrackers[i], GetRecoStepsGPU() & RecoStep::TPCSliceTracking);
    }
  }
  if (GetRecoSteps() & RecoStep::TPCMerging) {
    processors()->tpcMerger.SetTrackingChain(this);
    mRec->RegisterGPUProcessor(&processors()->tpcMerger, GetRecoStepsGPU() & RecoStep::TPCMerging);
  }
  if (GetRecoSteps() & RecoStep::TRDTracking) {
    processors()->trdTracker.SetTrackingChain(this);
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
  for (unsigned int i = 0; i < NSLICES; i++) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->tpcClusterer[i], &processors()->tpcClusterer[i]);
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
      GPUError("Invalid Reconstruction Step Setting: Tracking requires TPC Conversion to be active");
      return false;
    }
    if (((GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) || (GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging)) && !(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion)) {
      GPUError("Invalid GPU Reconstruction Step Setting: Tracking requires TPC Conversion to be active");
      return false;
    }
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding) && !(GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCRaw)) {
    GPUError("Invalid inputy, TPC Clusterizer needs TPC raw input");
    return false;
  }
  bool tpcClustersAvail = (GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCClusters) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCClusterFinding);
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
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCRaw) || (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCClusters) || (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TRDTracklets)) {
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
  return true;
}
int GPUChainTracking::Init()
{
  const auto& threadContext = GetThreadContext();
  if (GetDeviceProcessingSettings().debugLevel >= 1) {
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
    GPUError("Invalid GPU Reconstruction Step / Input / Output configuration");
    return 1;
  }

  if (GPUQA::QAAvailable() && (GetDeviceProcessingSettings().runQA || GetDeviceProcessingSettings().eventDisplay)) {
    mQA.reset(new GPUQA(this));
  }
  if (GetDeviceProcessingSettings().eventDisplay) {
    mEventDisplay.reset(new GPUDisplay(GetDeviceProcessingSettings().eventDisplay, this, mQA.get()));
  }

  if (mRec->IsGPU()) {
    if (processors()->calibObjects.fastTransform) {
      memcpy((void*)mFlatObjectsShadow.mCalibObjects.fastTransform, (const void*)processors()->calibObjects.fastTransform, sizeof(*processors()->calibObjects.fastTransform));
      memcpy((void*)mFlatObjectsShadow.mTpcTransformBuffer, (const void*)processors()->calibObjects.fastTransform->getFlatBufferPtr(), processors()->calibObjects.fastTransform->getFlatBufferSize());
      mFlatObjectsShadow.mCalibObjects.fastTransform->clearInternalBufferPtr();
      mFlatObjectsShadow.mCalibObjects.fastTransform->setActualBufferAddress(mFlatObjectsShadow.mTpcTransformBuffer);
      mFlatObjectsShadow.mCalibObjects.fastTransform->setFutureBufferAddress(mFlatObjectsDevice.mTpcTransformBuffer);
    }
#ifdef HAVE_O2HEADERS
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
#endif
    TransferMemoryResourceLinkToGPU(RecoStep::NoRecoStep, mFlatObjectsShadow.mMemoryResFlat);
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->calibObjects - (char*)processors(), &mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects), -1);
  }

  if (GetDeviceProcessingSettings().debugLevel >= 4) {
    mDebugFile.open(mRec->IsGPU() ? "GPU.out" : "CPU.out");
  }

  return 0;
}

void GPUChainTracking::PrepareEventFromNative()
{
#ifdef HAVE_O2HEADERS
  ClusterNativeAccess* tmp = mClusterNativeAccess.get();
  if (tmp != mIOPtrs.clustersNative) {
    *tmp = *mIOPtrs.clustersNative;
  }
  mIOPtrs.clustersNative = tmp;
  processors()->tpcConverter.mClustersNative = mClusterNativeAccess.get();

  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    processors()->tpcTrackers[iSlice].Data().SetClusterData(nullptr, mIOPtrs.clustersNative->nClustersSector[iSlice], mIOPtrs.clustersNative->clusterOffset[iSlice][0]);
  }
  processors()->tpcCompressor.mMaxClusters = mIOPtrs.clustersNative->nClustersTotal;
  mRec->MemoryScalers()->nTPCHits = mIOPtrs.clustersNative->nClustersTotal;
  GPUInfo("Event has %d TPC Clusters, %d TRD Tracklets", tmp->nClustersTotal, mIOPtrs.nTRDTracklets);
#endif
}

int GPUChainTracking::PrepareEvent()
{
  mRec->MemoryScalers()->nTRDTracklets = mIOPtrs.nTRDTracklets;
  if (mIOPtrs.tpcPackedDigits) {
#ifdef HAVE_O2HEADERS
    mRec->MemoryScalers()->nTPCdigits = 0;
    size_t maxDigits = 0;
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      mRec->MemoryScalers()->nTPCdigits += mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice];
      if (mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice] > maxDigits) {
        maxDigits = mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice];
      }
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processors()->tpcTrackers[iSlice].Data().SetClusterData(nullptr, param().rec.fwdTPCDigitsAsClusters ? mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice] : mRec->MemoryScalers()->NTPCClusters(mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice]), 0); // TODO: fixme
      // Distribute maximum digits, so that we can reuse the memory easily
      processors()->tpcClusterer[iSlice].SetNMaxDigits(maxDigits);
    }
    mRec->MemoryScalers()->nTPCHits = param().rec.fwdTPCDigitsAsClusters ? mRec->MemoryScalers()->nTPCdigits : mRec->MemoryScalers()->NTPCClusters(mRec->MemoryScalers()->nTPCdigits);
    processors()->tpcCompressor.mMaxClusters = mRec->MemoryScalers()->nTPCHits;
    processors()->tpcConverter.mNClustersTotal = mRec->MemoryScalers()->nTPCHits;
    GPUInfo("Event has %lld TPC Digits", (long long int)mRec->MemoryScalers()->nTPCdigits);
#endif
  } else if (mIOPtrs.clustersNative) {
    PrepareEventFromNative();
  } else {
    int offset = 0;
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processors()->tpcTrackers[iSlice].Data().SetClusterData(mIOPtrs.clusterData[iSlice], mIOPtrs.nClusterData[iSlice], offset);
      offset += mIOPtrs.nClusterData[iSlice];
    }
#ifdef HAVE_O2HEADERS
    processors()->tpcCompressor.mMaxClusters = offset;
#endif
    mRec->MemoryScalers()->nTPCHits = offset;
    GPUInfo("Event has %d TPC Clusters (converted), %d TRD Tracklets", offset, mIOPtrs.nTRDTracklets);
  }

  if (mRec->IsGPU()) {
    memcpy((void*)processorsShadow(), (const void*)processors(), sizeof(*processors()));
    mRec->ResetDeviceProcessorTypes();
  }
  return 0;
}

int GPUChainTracking::ForceInitQA()
{
  return mQA->InitQA();
}

int GPUChainTracking::Finalize()
{
  if (GetDeviceProcessingSettings().runQA && mQA->IsInitialized()) {
    mQA->DrawQAHistograms();
  }
  if (GetDeviceProcessingSettings().debugLevel >= 4) {
    mDebugFile.close();
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
#ifdef HAVE_O2HEADERS
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
  std::memset((void*)mClusterNativeAccess.get(), 0, sizeof(*mClusterNativeAccess));
}

void GPUChainTracking::AllocateIOMemory()
{
  for (unsigned int i = 0; i < NSLICES; i++) {
    AllocateIOMemoryHelper(mIOPtrs.nClusterData[i], mIOPtrs.clusterData[i], mIOMem.clusterData[i]);
    AllocateIOMemoryHelper(mIOPtrs.nRawClusters[i], mIOPtrs.rawClusters[i], mIOMem.rawClusters[i]);
    AllocateIOMemoryHelper(mIOPtrs.nSliceOutTracks[i], mIOPtrs.sliceOutTracks[i], mIOMem.sliceOutTracks[i]);
    AllocateIOMemoryHelper(mIOPtrs.nSliceOutClusters[i], mIOPtrs.sliceOutClusters[i], mIOMem.sliceOutClusters[i]);
  }
  AllocateIOMemoryHelper(mClusterNativeAccess->nClustersTotal, mClusterNativeAccess->clustersLinear, mIOMem.clustersNative);
  mIOPtrs.clustersNative = mClusterNativeAccess.get();
  AllocateIOMemoryHelper(mIOPtrs.nMCLabelsTPC, mIOPtrs.mcLabelsTPC, mIOMem.mcLabelsTPC);
  AllocateIOMemoryHelper(mIOPtrs.nMCInfosTPC, mIOPtrs.mcInfosTPC, mIOMem.mcInfosTPC);
  AllocateIOMemoryHelper(mIOPtrs.nMergedTracks, mIOPtrs.mergedTracks, mIOMem.mergedTracks);
  AllocateIOMemoryHelper(mIOPtrs.nMergedTrackHits, mIOPtrs.mergedTrackHits, mIOMem.mergedTrackHits);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracks, mIOPtrs.trdTracks, mIOMem.trdTracks);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdTracklets, mIOMem.trdTracklets);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTrackletsMC, mIOPtrs.trdTrackletsMC, mIOMem.trdTrackletsMC);
}

GPUChainTracking::InOutMemory::InOutMemory() = default;
GPUChainTracking::InOutMemory::~InOutMemory() = default;
GPUChainTracking::InOutMemory::InOutMemory(GPUChainTracking::InOutMemory&&) = default;
GPUChainTracking::InOutMemory& GPUChainTracking::InOutMemory::operator=(GPUChainTracking::InOutMemory&&) = default;

void GPUChainTracking::DumpData(const char* filename)
{
  FILE* fp = fopen(filename, "w+b");
  if (fp == nullptr) {
    return;
  }
  fwrite(DUMP_HEADER, 1, DUMP_HEADER_SIZE, fp);
  fwrite(&GPUReconstruction::geometryType, sizeof(GPUReconstruction::geometryType), 1, fp);
  DumpData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, InOutPointerType::CLUSTER_DATA);
  DumpData(fp, mIOPtrs.rawClusters, mIOPtrs.nRawClusters, InOutPointerType::RAW_CLUSTERS);
  if (mIOPtrs.clustersNative) {
    DumpData(fp, &mIOPtrs.clustersNative->clustersLinear, &mIOPtrs.clustersNative->nClustersTotal, InOutPointerType::CLUSTERS_NATIVE);
    fwrite(&mIOPtrs.clustersNative->nClusters[0][0], sizeof(mIOPtrs.clustersNative->nClusters[0][0]), NSLICES * GPUCA_ROW_COUNT, fp);
  }
  if (mIOPtrs.tpcPackedDigits) {
    DumpData(fp, mIOPtrs.tpcPackedDigits->tpcDigits, mIOPtrs.tpcPackedDigits->nTPCDigits, InOutPointerType::TPC_DIGIT);
  }
  DumpData(fp, mIOPtrs.sliceOutTracks, mIOPtrs.nSliceOutTracks, InOutPointerType::SLICE_OUT_TRACK);
  DumpData(fp, mIOPtrs.sliceOutClusters, mIOPtrs.nSliceOutClusters, InOutPointerType::SLICE_OUT_CLUSTER);
  DumpData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, InOutPointerType::MC_LABEL_TPC);
  DumpData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, InOutPointerType::MC_INFO_TPC);
  DumpData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, InOutPointerType::MERGED_TRACK);
  DumpData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
  DumpData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, InOutPointerType::TRD_TRACK);
  DumpData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, InOutPointerType::TRD_TRACKLET);
  DumpData(fp, &mIOPtrs.trdTrackletsMC, &mIOPtrs.nTRDTrackletsMC, InOutPointerType::TRD_TRACKLET_MC);
  fclose(fp);
}

int GPUChainTracking::ReadData(const char* filename)
{
  ClearIOPointers();
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    return (1);
  }

  /*int nTotal = 0;
  int nRead;
  for (int i = 0;i < NSLICES;i++)
  {
    int nHits;
    nRead = fread(&nHits, sizeof(nHits), 1, fp);
    mIOPtrs.nClusterData[i] = nHits;
    AllocateIOMemoryHelper(nHits, mIOPtrs.clusterData[i], mIOMem.clusterData[i]);
    nRead = fread(mIOMem.clusterData[i].get(), sizeof(*mIOPtrs.clusterData[i]), nHits, fp);
    for (int j = 0;j < nHits;j++)
    {
      mIOMem.clusterData[i][j].fId = nTotal++;
    }
  }
  GPUInfo("Read %d hits", nTotal);
  mIOPtrs.nMCLabelsTPC = nTotal;
  AllocateIOMemoryHelper(nTotal, mIOPtrs.mcLabelsTPC, mIOMem.mcLabelsTPC);
  nRead = fread(mIOMem.mcLabelsTPC.get(), sizeof(*mIOPtrs.mcLabelsTPC), nTotal, fp);
  if (nRead != nTotal)
  {
    mIOPtrs.nMCLabelsTPC = 0;
  }
  else
  {
    GPUInfo("Read %d MC labels", nTotal);
    int nTracks;
    nRead = fread(&nTracks, sizeof(nTracks), 1, fp);
    if (nRead)
    {
      mIOPtrs.nMCInfosTPC = nTracks;
      AllocateIOMemoryHelper(nTracks, mIOPtrs.mcInfosTPC, mIOMem.mcInfosTPC);
      nRead = fread(mIOMem.mcInfosTPC.get(), sizeof(*mIOPtrs.mcInfosTPC), nTracks, fp);
      GPUInfo("Read %d MC Infos", nTracks);
    }
  }*/

  char buf[DUMP_HEADER_SIZE + 1] = "";
  size_t r = fread(buf, 1, DUMP_HEADER_SIZE, fp);
  if (strncmp(DUMP_HEADER, buf, DUMP_HEADER_SIZE)) {
    GPUError("Invalid file header");
    fclose(fp);
    return -1;
  }
  GeometryType geo;
  r = fread(&geo, sizeof(geo), 1, fp);
  if (geo != GPUReconstruction::geometryType) {
    GPUError("File has invalid geometry (%s v.s. %s)", GPUReconstruction::GEOMETRY_TYPE_NAMES[(int)geo], GPUReconstruction::GEOMETRY_TYPE_NAMES[(int)GPUReconstruction::geometryType]);
    fclose(fp);
    return 1;
  }
  ReadData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, mIOMem.clusterData, InOutPointerType::CLUSTER_DATA);
  int nClustersTotal = 0;
  ReadData(fp, mIOPtrs.rawClusters, mIOPtrs.nRawClusters, mIOMem.rawClusters, InOutPointerType::RAW_CLUSTERS);
#ifdef HAVE_O2HEADERS
  if (ReadData<ClusterNative>(fp, &mClusterNativeAccess->clustersLinear, &mClusterNativeAccess->nClustersTotal, &mIOMem.clustersNative, InOutPointerType::CLUSTERS_NATIVE)) {
    mIOPtrs.clustersNative = mClusterNativeAccess.get();
    r = fread(&mClusterNativeAccess->nClusters[0][0], sizeof(mClusterNativeAccess->nClusters[0][0]), NSLICES * GPUCA_ROW_COUNT, fp);
    mClusterNativeAccess->setOffsetPtrs();
  }
  mDigitMap.reset(new GPUTrackingInOutDigits);
  if (ReadData(fp, mDigitMap->tpcDigits, mDigitMap->nTPCDigits, mIOMem.tpcDigits, InOutPointerType::TPC_DIGIT)) {
    mIOPtrs.tpcPackedDigits = mDigitMap.get();
  }
#endif
  ReadData(fp, mIOPtrs.sliceOutTracks, mIOPtrs.nSliceOutTracks, mIOMem.sliceOutTracks, InOutPointerType::SLICE_OUT_TRACK);
  ReadData(fp, mIOPtrs.sliceOutClusters, mIOPtrs.nSliceOutClusters, mIOMem.sliceOutClusters, InOutPointerType::SLICE_OUT_CLUSTER);
  ReadData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, &mIOMem.mcLabelsTPC, InOutPointerType::MC_LABEL_TPC);
  ReadData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, &mIOMem.mcInfosTPC, InOutPointerType::MC_INFO_TPC);
  ReadData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, &mIOMem.mergedTracks, InOutPointerType::MERGED_TRACK);
  ReadData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, &mIOMem.mergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
  ReadData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, &mIOMem.trdTracks, InOutPointerType::TRD_TRACK);
  ReadData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, &mIOMem.trdTracklets, InOutPointerType::TRD_TRACKLET);
  ReadData(fp, &mIOPtrs.trdTrackletsMC, &mIOPtrs.nTRDTrackletsMC, &mIOMem.trdTrackletsMC, InOutPointerType::TRD_TRACKLET_MC);
  fclose(fp);
  (void)r;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < mIOPtrs.nClusterData[i]; j++) {
      mIOMem.clusterData[i][j].id = nClustersTotal++;
      if ((unsigned int)mIOMem.clusterData[i][j].amp >= 25 * 1024) {
        GPUError("Invalid cluster charge, truncating (%d >= %d)", (int)mIOMem.clusterData[i][j].amp, 25 * 1024);
        mIOMem.clusterData[i][j].amp = 25 * 1024 - 1;
      }
    }
    for (unsigned int j = 0; j < mIOPtrs.nRawClusters[i]; j++) {
      if ((unsigned int)mIOMem.rawClusters[i][j].GetCharge() >= 25 * 1024) {
        GPUError("Invalid raw cluster charge, truncating (%d >= %d)", (int)mIOMem.rawClusters[i][j].GetCharge(), 25 * 1024);
        mIOMem.rawClusters[i][j].SetCharge(25 * 1024 - 1);
      }
      if ((unsigned int)mIOMem.rawClusters[i][j].GetQMax() >= 1024) {
        GPUError("Invalid raw cluster charge max, truncating (%d >= %d)", (int)mIOMem.rawClusters[i][j].GetQMax(), 1024);
        mIOMem.rawClusters[i][j].SetQMax(1024 - 1);
      }
    }
  }

  return (0);
}

void GPUChainTracking::DumpSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "tpctransform.dump";
  if (processors()->calibObjects.fastTransform != nullptr) {
    DumpFlatObjectToFile(processors()->calibObjects.fastTransform, f.c_str());
  }

#ifdef HAVE_O2HEADERS
  f = dir;
  f += "matlut.dump";
  if (processors()->calibObjects.matLUT != nullptr) {
    DumpFlatObjectToFile(processors()->calibObjects.matLUT, f.c_str());
  }
  f = dir;
  f += "trdgeometry.dump";
  if (processors()->calibObjects.trdGeometry != nullptr) {
    DumpStructToFile(processors()->calibObjects.trdGeometry, f.c_str());
  }

#endif
}

void GPUChainTracking::ReadSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "tpctransform.dump";
  mTPCFastTransformU = ReadFlatObjectFromFile<TPCFastTransform>(f.c_str());
  processors()->calibObjects.fastTransform = mTPCFastTransformU.get();
#ifdef HAVE_O2HEADERS
  f = dir;
  f += "matlut.dump";
  mMatLUTU = ReadFlatObjectFromFile<o2::base::MatLayerCylSet>(f.c_str());
  processors()->calibObjects.matLUT = mMatLUTU.get();
  f = dir;
  f += "trdgeometry.dump";
  mTRDGeometryU = ReadStructFromFile<o2::trd::TRDGeometryFlat>(f.c_str());
  processors()->calibObjects.trdGeometry = mTRDGeometryU.get();
#endif
}

int GPUChainTracking::ConvertNativeToClusterData()
{
#ifdef HAVE_O2HEADERS
  const auto& threadContext = GetThreadContext();
  mRec->SetThreadCounts(RecoStep::TPCConversion);
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCConversion;
  GPUTPCConvert& convert = processors()->tpcConverter;
  GPUTPCConvert& convertShadow = doGPU ? processorsShadow()->tpcConverter : convert;

  ClusterNativeAccess* tmpExt = mClusterNativeAccess.get();
  convert.set(tmpExt, processors()->calibObjects.fastTransform);
  SetupGPUProcessor(&convert, false);
  if (doGPU) {
    convertShadow.set(convertShadow.mClustersNativeBuffer, mFlatObjectsDevice.mCalibObjects.fastTransform);
    processorsShadow()->ioPtrs.clustersNative = convertShadow.mClustersNativeBuffer;
    WriteToConstantMemory(RecoStep::TPCConversion, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    *convert.mClustersNativeBuffer = *mClusterNativeAccess.get();
    convert.mClustersNativeBuffer->clustersLinear = convertShadow.mInputClusters; // We overwrite the pointers of the host buffer, this will be moved to the GPU, should be cleaned up
    for (unsigned int i = 0; i < NSLICES; i++) {
      convert.mMemory->clusters[i] = convertShadow.mClusters + tmpExt->clusterOffset[i][0];
      for (unsigned int j = 0; j < Constants::MAXGLOBALPADROW; j++) {
        ClusterNative* ptr = convertShadow.mInputClusters + convert.mClustersNativeBuffer->clusterOffset[i][j];
        convert.mClustersNativeBuffer->clusters[i][j] = ptr; // We overwrite the pointers of the host buffer, this will be moved to the GPU, should be cleaned up
        GPUMemCpy(RecoStep::TPCConversion, ptr, mClusterNativeAccess->clusters[i][j], sizeof(mClusterNativeAccess->clusters[i][j][0]) * mClusterNativeAccess->nClusters[i][j], 0, true);
      }
    }
  } else {
    for (unsigned int i = 0; i < NSLICES; i++) {
      convert.mMemory->clusters[i] = convert.mClusters + tmpExt->clusterOffset[i][0];
    }
  }

  WriteToConstantMemory(RecoStep::TPCConversion, (char*)&processors()->tpcConverter - (char*)processors(), &convertShadow, sizeof(convertShadow), 0);
  TransferMemoryResourcesToGPU(RecoStep::TPCConversion, &convert, 0);
  runKernel<GPUTPCConvertKernel>({NSLICES * GPUCA_ROW_COUNT, ThreadCount(), 0}, krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(RecoStep::TPCConversion, &convert, 0);
  SynchronizeGPU();

  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.nClusterData[i] = (i == NSLICES - 1 ? tmpExt->nClustersTotal : tmpExt->clusterOffset[i + 1][0]) - tmpExt->clusterOffset[i][0];
    mIOPtrs.clusterData[i] = convert.mClusters + mClusterNativeAccess->clusterOffset[i][0];
    processors()->tpcTrackers[i].Data().SetClusterData(mIOPtrs.clusterData[i], mIOPtrs.nClusterData[i], mClusterNativeAccess->clusterOffset[i][0]);
  }
#endif
  return 0;
}

void GPUChainTracking::ConvertNativeToClusterDataLegacy()
{
  ClusterNativeAccess* tmp = mClusterNativeAccess.get();
  if (tmp != mIOPtrs.clustersNative) {
    *tmp = *mIOPtrs.clustersNative;
  }
  GPUReconstructionConvert::ConvertNativeToClusterData(mClusterNativeAccess.get(), mIOMem.clusterData, mIOPtrs.nClusterData, processors()->calibObjects.fastTransform, param().continuousMaxTimeBin);
  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.clusterData[i] = mIOMem.clusterData[i].get();
  }
  mIOPtrs.clustersNative = nullptr;
  mIOMem.clustersNative.reset(nullptr);
  memset((void*)mClusterNativeAccess.get(), 0, sizeof(*mClusterNativeAccess));
}

void GPUChainTracking::ConvertRun2RawToNative()
{
  GPUReconstructionConvert::ConvertRun2RawToNative(*mClusterNativeAccess, mIOMem.clustersNative, mIOPtrs.rawClusters, mIOPtrs.nRawClusters);
  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.rawClusters[i] = nullptr;
    mIOPtrs.nRawClusters[i] = 0;
    mIOMem.rawClusters[i].reset(nullptr);
    mIOPtrs.clusterData[i] = nullptr;
    mIOPtrs.nClusterData[i] = 0;
    mIOMem.clusterData[i].reset(nullptr);
  }
  mIOPtrs.clustersNative = mClusterNativeAccess.get();
}

void GPUChainTracking::LoadClusterErrors() { param().LoadClusterErrors(); }

void GPUChainTracking::SetTPCFastTransform(std::unique_ptr<TPCFastTransform>&& tpcFastTransform)
{
  mTPCFastTransformU = std::move(tpcFastTransform);
  processors()->calibObjects.fastTransform = mTPCFastTransformU.get();
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

int GPUChainTracking::ReadEvent(int iSlice, int threadId)
{
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("Running ReadEvent for slice %d on thread %d\n", iSlice, threadId);
  }
  HighResTimer& timer = getTimer<GPUTPCSliceData>("ReadEvent", threadId);
  timer.Start();
  if (processors()->tpcTrackers[iSlice].ReadEvent()) {
    return (1);
  }
  timer.Stop();
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("Finished ReadEvent for slice %d on thread %d\n", iSlice, threadId);
  }
  return (0);
}

void GPUChainTracking::WriteOutput(int iSlice, int threadId)
{
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("Running WriteOutput for slice %d on thread %d\n", iSlice, threadId);
  }
  HighResTimer& timer = getTimer<GPUTPCSliceOutput>("WriteOutput", threadId);
  timer.Start();
  if (GetDeviceProcessingSettings().nDeviceHelperThreads) {
    while (mLockAtomic.test_and_set(std::memory_order_acquire)) {
      ;
    }
  }
  processors()->tpcTrackers[iSlice].WriteOutputPrepare();
  if (GetDeviceProcessingSettings().nDeviceHelperThreads) {
    mLockAtomic.clear();
  }
  processors()->tpcTrackers[iSlice].WriteOutput();
  timer.Stop();
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("Finished WriteOutput for slice %d on thread %d\n", iSlice, threadId);
  }
}

void GPUChainTracking::ForwardTPCDigits()
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
      if (d.charge >= zsThreshold) {
        ClusterNative c;
        c.setTimeFlags(d.time, 0);
        c.setPad(d.pad);
        c.setSigmaTime(1);
        c.setSigmaPad(1);
        c.qTot = c.qMax = d.charge;
        tmp[i][d.row].emplace_back(c);
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
  printf("Forwarded %u TPC clusters\n", nTotal);
  PrepareEventFromNative();
#endif
}

int GPUChainTracking::GlobalTracking(int iSlice, int threadId)
{
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("GPU Tracker running Global Tracking for slice %d on thread %d\n", iSlice, threadId);
  }

  int sliceLeft = (iSlice + (NSLICES / 2 - 1)) % (NSLICES / 2);
  int sliceRight = (iSlice + 1) % (NSLICES / 2);
  if (iSlice >= (int)NSLICES / 2) {
    sliceLeft += NSLICES / 2;
    sliceRight += NSLICES / 2;
  }
  while (mSliceOutputReady < iSlice || mSliceOutputReady < sliceLeft || mSliceOutputReady < sliceRight) {
  }

  HighResTimer& timer = getTimer<GPUTPCGlobalTracking>("GlobalTracking", threadId);
  timer.Start();
  processors()->tpcTrackers[iSlice].PerformGlobalTracking(processors()->tpcTrackers[sliceLeft], processors()->tpcTrackers[sliceRight]);
  timer.Stop();

  mSliceLeftGlobalReady[sliceLeft] = 1;
  mSliceRightGlobalReady[sliceRight] = 1;
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("GPU Tracker finished Global Tracking for slice %d on thread %d\n", iSlice, threadId);
  }
  return (0);
}

void GPUChainTracking::RunTPCClusterizer_compactPeaks(GPUTPCClusterFinder& clusterer, GPUTPCClusterFinder& clustererShadow, int stage, bool doGPU, int lane)
{
#ifdef HAVE_O2HEADERS
  auto& in = stage ? clustererShadow.mPpeaks : clustererShadow.mPdigits;
  auto& out = stage ? clustererShadow.mPfilteredPeaks : clustererShadow.mPpeaks;
  if (doGPU) {
    const unsigned int iSlice = clusterer.mISlice;
    auto& count = stage ? clusterer.mPmemory->nPeaks : clusterer.mPmemory->nDigits;

    std::vector<size_t> counts;

    unsigned int nSteps = clusterer.getNSteps(count);
    if (nSteps > clusterer.mNBufs) {
      printf("Clusterer buffers exceeded (%d > %d)\n", nSteps, (int)clusterer.mNBufs);
      exit(1);
    }

    size_t tmpCount = count;
    if (nSteps > 1) {
      for (unsigned int i = 1; i < nSteps; i++) {
        counts.push_back(tmpCount);
        if (i == 1) {
          runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::nativeScanUpStart>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, stage);
        } else {
          runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::nativeScanUp>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, tmpCount);
        }
        tmpCount /= clusterer.mScanWorkGroupSize;
      }

      runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::nativeScanTop>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, nSteps, tmpCount);

      for (unsigned int i = nSteps - 1; i > 1; i--) {
        tmpCount = counts[i - 1];
        runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::nativeScanDown>(GetGrid(tmpCount - clusterer.mScanWorkGroupSize, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, clusterer.mScanWorkGroupSize, tmpCount);
      }
    }

    runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::compactDigit>(GetGrid(count, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, 1, stage, in, out);
  } else {
    auto& nOut = stage ? clusterer.mPmemory->nClusters : clusterer.mPmemory->nPeaks;
    auto& nIn = stage ? clusterer.mPmemory->nPeaks : clusterer.mPmemory->nDigits;
    size_t count = 0;
    for (size_t i = 0; i < nIn; i++) {
      if (clusterer.mPisPeak[i]) {
        out[count++] = in[i];
      }
    }
    nOut = count;
  }
#endif
}

int GPUChainTracking::RunTPCClusterizer()
{
#ifdef HAVE_O2HEADERS
  const auto& threadContext = GetThreadContext();
  mRec->SetThreadCounts(RecoStep::TPCClusterFinding);
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;

  if (doGPU) {
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)processors()->tpcClusterer - (char*)processors(), processorsShadow()->tpcClusterer, sizeof(GPUTPCClusterFinder) * NSLICES, mRec->NStreams() - 1, &mEvents->init);
  }
  SynchronizeGPU();

  static std::vector<o2::tpc::ClusterNative> clsMemory;
  size_t nClsTotal = 0;
  ClusterNativeAccess* tmp = mClusterNativeAccess.get();
  size_t pos = 0;
  clsMemory.reserve(mRec->MemoryScalers()->nTPCHits);
  for (unsigned int iSliceBase = 0; iSliceBase < NSLICES; iSliceBase += GetDeviceProcessingSettings().nTPCClustererLanes) {
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
      SetupGPUProcessor(&clusterer, false);
      memset((void*)clusterer.mPmemory, 0, sizeof(*clusterer.mPmemory));
      if (doGPU) {
        clusterer.mPmemory->nDigits = mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice];
        TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPdigits, mIOPtrs.tpcPackedDigits->tpcDigits[iSlice], sizeof(clustererShadow.mPdigits[0]) * clusterer.mPmemory->nDigits, lane, true);
        clusterer.mPdigits = (deprecated::PackedDigit*)mIOPtrs.tpcPackedDigits->tpcDigits[iSlice]; // TODO: Needs fixing (see below) + need pointer on CPU clusterizer for debug output
      } else {
        clusterer.mPdigits = (deprecated::PackedDigit*)mIOPtrs.tpcPackedDigits->tpcDigits[iSlice]; // TODO: Needs fixing, double-allocated and invalid const cast
        clusterer.mPmemory->nDigits = mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice];
      }
      if (clusterer.mPmemory->nDigits == 0) {
        continue;
      }
      if (GetDeviceProcessingSettings().debugLevel >= 6) {
        clusterer.DumpDigits(mDebugFile);
      }
      runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPchargeMap, TPC_NUM_OF_PADS * TPC_MAX_TIME_PADDED * sizeof(*clustererShadow.mPchargeMap));
      runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPpeakMap, TPC_NUM_OF_PADS * TPC_MAX_TIME_PADDED * sizeof(*clustererShadow.mPpeakMap));
      runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPclusterInRow, GPUCA_ROW_COUNT * sizeof(*clustererShadow.mPclusterInRow));
      runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPpeaks, clusterer.mNMaxDigits * sizeof(*clustererShadow.mPpeaks));                 // TODO: Do we really need to empty all of them?
      runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPfilteredPeaks, clusterer.mNMaxDigits * sizeof(*clustererShadow.mPfilteredPeaks)); // These 4 were not emptied originally (i think)
      runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPisPeak, clusterer.mNMaxDigits * sizeof(*clustererShadow.mPisPeak));               // but that yielded fake peaks after nDigits

      runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::fillChargeMap>(GetGrid(clusterer.mPmemory->nDigits, ClustererThreadCount(), lane), {iSlice}, {});
      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, -1, true);
        if (GetDeviceProcessingSettings().debugLevel >= 6) {
          clusterer.DumpChargeMap(mDebugFile, "Charges");
        }
      }
      runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::findPeaks>(GetGrid(clusterer.mPmemory->nDigits, ClustererThreadCount(), lane), {iSlice}, {});
      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, -1, true);
        if (GetDeviceProcessingSettings().debugLevel >= 6) {
          clusterer.DumpPeaks(mDebugFile);
        }
      }
      RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 0, doGPU, lane);
      TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, -1, true);
        if (GetDeviceProcessingSettings().debugLevel >= 6) {
          clusterer.DumpPeaksCompacted(mDebugFile);
        }
      }
    }
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
      SynchronizeStream(lane);
      if (clusterer.mPmemory->nPeaks == 0) {
        continue;
      }
      runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::noiseSuppression>(GetGrid(clusterer.mPmemory->nPeaks, ClustererThreadCount(), lane), {iSlice}, {});
      runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::updatePeaks>(GetGrid(clusterer.mPmemory->nPeaks, ClustererThreadCount(), lane), {iSlice}, {});
      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, -1, true);
        if (GetDeviceProcessingSettings().debugLevel >= 6) {
          clusterer.DumpSuppressedPeaks(mDebugFile);
        }
      }
      RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 1, doGPU, lane);
      TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, -1, true);
        if (GetDeviceProcessingSettings().debugLevel >= 6) {
          clusterer.DumpSuppressedPeaksCompacted(mDebugFile);
        }
      }
      SynchronizeStream(lane);
      RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 1, doGPU, lane);
      TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
    }
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      SynchronizeStream(lane);
      if (clusterer.mPmemory->nClusters == 0) {
        continue;
      }
      runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::countPeaks>(GetGrid(clusterer.mPmemory->nDigits, ClustererThreadCount(), lane), {iSlice}, {});
      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, -1, true);
        if (GetDeviceProcessingSettings().debugLevel >= 6) {
          clusterer.DumpCountedPeaks(mDebugFile);
          clusterer.DumpChargeMap(mDebugFile, "Split Charges");
        }
      }
      runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::computeClusters>(GetGrid(clusterer.mPmemory->nClusters, ClustererThreadCount(), lane), {iSlice}, {});
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        printf("Lane %d: Found clusters: digits %d peaks %d clusters %d\n", lane, (int)clusterer.mPmemory->nDigits, (int)clusterer.mPmemory->nPeaks, (int)clusterer.mPmemory->nClusters);
      }
      TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, lane);
      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, -1, true);
        if (GetDeviceProcessingSettings().debugLevel >= 6) {
          clusterer.DumpClusters(mDebugFile);
        }
      }
    }
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      SynchronizeStream(lane);
      nClsTotal += clusterer.mPmemory->nClusters;
      clsMemory.resize(nClsTotal);
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        memcpy((void*)&clsMemory[pos], (const void*)&clusterer.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], clusterer.mPclusterInRow[j] * sizeof(clsMemory[0]));
        tmp->nClusters[iSlice][j] = clusterer.mPclusterInRow[j];
        pos += clusterer.mPclusterInRow[j];
      }
      runKernel<GPUTPCClusterFinderKernels, GPUTPCClusterFinderKernels::resetMaps>(GetGrid(clusterer.mPmemory->nDigits, ClustererThreadCount(), lane), {iSlice}, {});
    }
  }

  tmp->clustersLinear = clsMemory.data();
  tmp->setOffsetPtrs();
  mIOPtrs.clustersNative = tmp;
  PrepareEventFromNative();
#endif
  return 0;
}

int GPUChainTracking::RunTPCTrackingSlices()
{
  if (!(GetRecoStepsGPU() & RecoStep::TPCSliceTracking) && mRec->OutputControl().OutputType != GPUOutputControl::AllocateInternal && GetDeviceProcessingSettings().nThreads > 1) {
    GPUError("mOutputPtr must not be used with multiple threads\n");
    return (1);
  }

  if (mRec->GPUStuck()) {
    GPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
    return (1);
  }

  const auto& threadContext = GetThreadContext();
  mRec->SetThreadCounts(RecoStep::TPCSliceTracking);

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
  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Slice Tracker");
  }
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCSliceTracking;

  bool streamInit[GPUCA_MAX_STREAMS] = {false};
  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    *processors()->tpcTrackers[iSlice].NTracklets() = 0;
    *processors()->tpcTrackers[iSlice].NTracks() = 0;
    *processors()->tpcTrackers[iSlice].NTrackHits() = 0;
    processors()->tpcTrackers[iSlice].CommonMemory()->kernelError = 0;
  }
  if (doGPU) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processorsShadow()->tpcTrackers[iSlice].GPUParametersConst()->gpumem = (char*)mRec->DeviceMemoryBase();
      // Initialize Startup Constants
      processors()->tpcTrackers[iSlice].GPUParameters()->nextTracklet = ((ConstructorBlockCount() + NSLICES - 1 - iSlice) / NSLICES) * ConstructorThreadCount();
      processorsShadow()->tpcTrackers[iSlice].SetGPUTextureBase(mRec->DeviceMemoryBase());
    }

    RunHelperThreads(&GPUChainTracking::HelperReadEvent, this, NSLICES);
    if (PrepareTextures()) {
      return (2);
    }

    // Copy Tracker Object to GPU Memory
    if (GetDeviceProcessingSettings().debugLevel >= 3) {
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
#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(doGPU ? 1 : GetDeviceProcessingSettings().nThreads)
#endif
  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    GPUTPCTracker& trk = processors()->tpcTrackers[iSlice];
    GPUTPCTracker& trkShadow = doGPU ? processorsShadow()->tpcTrackers[iSlice] : trk;

    if (GetDeviceProcessingSettings().debugLevel >= 3) {
      GPUInfo("Creating Slice Data (Slice %d)", iSlice);
    }
    if (!doGPU || iSlice % (GetDeviceProcessingSettings().nDeviceHelperThreads + 1) == 0) {
      if (ReadEvent(iSlice, 0)) {
        GPUError("Error reading event");
        error = 1;
        continue;
      }
    } else {
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Waiting for helper thread %d", iSlice % (GetDeviceProcessingSettings().nDeviceHelperThreads + 1) - 1);
      }
      while (HelperDone(iSlice % (GetDeviceProcessingSettings().nDeviceHelperThreads + 1) - 1) < (int)iSlice) {
        ;
      }
      if (HelperError(iSlice % (GetDeviceProcessingSettings().nDeviceHelperThreads + 1) - 1)) {
        error = 1;
        continue;
      }
    }
    if (!doGPU && trk.CheckEmptySlice()) {
      continue;
    }

    if (GetDeviceProcessingSettings().debugLevel >= 4) {
      if (!GetDeviceProcessingSettings().comparableDebutOutput) {
        mDebugFile << std::endl
                   << std::endl
                   << "Reconstruction: Slice " << iSlice << "/" << NSLICES << std::endl;
      }
      if (GetDeviceProcessingSettings().debugMask & 1) {
        trk.DumpSliceData(mDebugFile);
      }
    }

    int useStream = (iSlice % mRec->NStreams());
    // Initialize temporary memory where needed
    if (GetDeviceProcessingSettings().debugLevel >= 3) {
      GPUInfo("Copying Slice Data to GPU and initializing temporary memory");
    }
    if (GetDeviceProcessingSettings().keepAllMemory) {
      memset((void*)trk.Data().HitWeights(), 0, trkShadow.Data().NumberOfHitsPlusAlign() * sizeof(*trkShadow.Data().HitWeights()));
    } else {
      runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), useStream, RecoStep::TPCSliceTracking}, krnlRunRangeNone, {}, trkShadow.Data().HitWeights(), trkShadow.Data().NumberOfHitsPlusAlign() * sizeof(*trkShadow.Data().HitWeights()));
    }

    // Copy Data to GPU Global Memory
    HighResTimer& timer = getTimer<GPUTPCSliceData>("ReadEvent", useStream);
    timer.Start();
    TransferMemoryResourcesToGPU(RecoStep::TPCSliceTracking, &trk);
    if (GPUDebug("Initialization (3)", useStream)) {
      throw std::runtime_error("memcpy failure");
    }
    timer.Stop();

    runKernel<GPUTPCNeighboursFinder>({GPUCA_ROW_COUNT, FinderThreadCount(), useStream}, {iSlice}, {nullptr, streamInit[useStream] ? nullptr : &mEvents->init});
    streamInit[useStream] = true;

    if (GetDeviceProcessingSettings().keepAllMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &trk, -1, true);
      memcpy(trk.LinkTmpMemory(), mRec->Res(trk.MemoryResLinksScratch()).Ptr(), mRec->Res(trk.MemoryResLinksScratch()).Size());
      if (GetDeviceProcessingSettings().debugMask & 2) {
        trk.DumpLinks(mDebugFile);
      }
    }

    runKernel<GPUTPCNeighboursCleaner>({GPUCA_ROW_COUNT - 2, ThreadCount(), useStream}, {iSlice});

    if (GetDeviceProcessingSettings().debugLevel >= 4) {
      TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &trk, -1, true);
      if (GetDeviceProcessingSettings().debugMask & 4) {
        trk.DumpLinks(mDebugFile);
      }
    }

    runKernel<GPUTPCStartHitsFinder>({GPUCA_ROW_COUNT - 6, ThreadCount(), useStream}, {iSlice});

    if (doGPU) {
      runKernel<GPUTPCStartHitsSorter>({BlockCount(), ThreadCount(), useStream}, {iSlice});
    }

    if (GetDeviceProcessingSettings().debugLevel >= 2) {
      TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, trk.MemoryResCommon(), -1);
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *trk.NTracklets(), iSlice);
      }
    }

    if (GetDeviceProcessingSettings().debugLevel >= 4 && *trk.NTracklets()) {
      TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &trk, -1, true);
      if (GetDeviceProcessingSettings().debugMask & 32) {
        trk.DumpStartHits(mDebugFile);
      }
    }

    if (GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
      trk.UpdateMaxData();
      AllocateRegisteredMemory(trk.MemoryResTracklets());
      AllocateRegisteredMemory(trk.MemoryResOutput());
    }

    if (!doGPU || GetDeviceProcessingSettings().trackletConstructorInPipeline) {
      runKernel<GPUTPCTrackletConstructor>({ConstructorBlockCount(), ConstructorThreadCount(), useStream}, {iSlice});
      if (GetDeviceProcessingSettings().debugLevel >= 4) {
        TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &trk, -1, true);
      }
      if (GetDeviceProcessingSettings().debugMask & 128) {
        trk.DumpTrackletHits(mDebugFile);
      }
      if (GetDeviceProcessingSettings().debugMask & 256 && !GetDeviceProcessingSettings().comparableDebutOutput) {
        trk.DumpHitWeights(mDebugFile);
      }
    }

    if (!doGPU || GetDeviceProcessingSettings().trackletSelectorInPipeline) {
      runKernel<GPUTPCTrackletSelector>({SelectorBlockCount(), SelectorThreadCount(), useStream}, {iSlice});
      TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, trk.MemoryResCommon(), useStream, &mEvents->selector[iSlice]);
      streamMap[iSlice] = useStream;
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Slice %u, Number of tracks: %d", iSlice, *trk.NTracks());
      }
      if (GetDeviceProcessingSettings().debugLevel >= 4) {
        TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &trk, -1, true);
      }
      if (GetDeviceProcessingSettings().debugMask & 512) {
        trk.DumpTrackHits(mDebugFile);
      }
    }

    if (!doGPU) {
      trk.CommonMemory()->nLocalTracks = trk.CommonMemory()->nTracks;
      trk.CommonMemory()->nLocalTrackHits = trk.CommonMemory()->nTrackHits;
      if (!param().rec.GlobalTracking) {
        WriteOutput(iSlice, 0);
      }
    }
  }
  if (error) {
    return (3);
  }

  if (doGPU) {
    ReleaseEvent(&mEvents->init);
    WaitForHelperThreads();

    if (!GetDeviceProcessingSettings().trackletSelectorInPipeline) {
      if (GetDeviceProcessingSettings().trackletConstructorInPipeline) {
        SynchronizeGPU();
      } else {
        for (int i = 0; i < mRec->NStreams(); i++) {
          RecordMarker(&mEvents->stream[i], i);
        }
        runKernel<GPUTPCTrackletConstructor, 1>({ConstructorBlockCount(), ConstructorThreadCount(), 0}, krnlRunRangeNone, {&mEvents->constructor, mEvents->stream, mRec->NStreams()});
        for (int i = 0; i < mRec->NStreams(); i++) {
          ReleaseEvent(&mEvents->stream[i]);
        }
        SynchronizeEvents(&mEvents->constructor);
        ReleaseEvent(&mEvents->constructor);
      }

      if (GetDeviceProcessingSettings().debugLevel >= 4) {
        for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
          TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &processors()->tpcTrackers[iSlice], -1, true);
          GPUInfo("Obtained %d tracklets", *processors()->tpcTrackers[iSlice].NTracklets());
          if (GetDeviceProcessingSettings().debugMask & 128) {
            processors()->tpcTrackers[iSlice].DumpTrackletHits(mDebugFile);
          }
        }
      }

      unsigned int runSlices = 0;
      int useStream = 0;
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice += runSlices) {
        if (runSlices < GPUCA_TRACKLET_SELECTOR_SLICE_COUNT) {
          runSlices++;
        }
        runSlices = CAMath::Min(runSlices, NSLICES - iSlice);
        if (SelectorBlockCount() < runSlices) {
          runSlices = SelectorBlockCount();
        }

        if (GetDeviceProcessingSettings().debugLevel >= 3) {
          GPUInfo("Running HLT Tracklet selector (Stream %d, Slice %d to %d)", useStream, iSlice, iSlice + runSlices);
        }
        runKernel<GPUTPCTrackletSelector>({SelectorBlockCount(), SelectorThreadCount(), useStream}, {iSlice, (int)runSlices});
        for (unsigned int k = iSlice; k < iSlice + runSlices; k++) {
          TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, processors()->tpcTrackers[k].MemoryResCommon(), useStream, &mEvents->selector[k]);
          streamMap[k] = useStream;
        }
        useStream++;
        if (useStream >= mRec->NStreams()) {
          useStream = 0;
        }
      }
    }

    mSliceOutputReady = 0;

    if (param().rec.GlobalTracking) {
      memset((void*)mSliceLeftGlobalReady, 0, sizeof(mSliceLeftGlobalReady));
      memset((void*)mSliceRightGlobalReady, 0, sizeof(mSliceRightGlobalReady));
      mGlobalTrackingDone.fill(0);
      mWriteOutputDone.fill(0);
    }
    RunHelperThreads(&GPUChainTracking::HelperOutput, this, NSLICES);

    std::array<bool, NSLICES> transferRunning;
    transferRunning.fill(true);
    unsigned int tmpSlice = 0;
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Transfering Tracks from GPU to Host");
      }

      if (tmpSlice == iSlice) {
        SynchronizeEvents(&mEvents->selector[iSlice]);
      }
      while (tmpSlice < NSLICES && (tmpSlice == iSlice || IsEventDone(&mEvents->selector[tmpSlice]))) {
        ReleaseEvent(&mEvents->selector[tmpSlice]);
        if (*processors()->tpcTrackers[tmpSlice].NTracks() > 0) {
          TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, processors()->tpcTrackers[tmpSlice].MemoryResOutput(), streamMap[tmpSlice], &mEvents->selector[tmpSlice]);
        } else {
          transferRunning[tmpSlice] = false;
        }
        tmpSlice++;
      }

      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &processors()->tpcTrackers[iSlice], -1, true);
        if (!GetDeviceProcessingSettings().trackletConstructorInPipeline) {
          if (GetDeviceProcessingSettings().debugMask & 256 && !GetDeviceProcessingSettings().comparableDebutOutput) {
            processors()->tpcTrackers[iSlice].DumpHitWeights(mDebugFile);
          }
        }
        if (!GetDeviceProcessingSettings().trackletSelectorInPipeline) {
          if (GetDeviceProcessingSettings().debugMask & 512) {
            processors()->tpcTrackers[iSlice].DumpTrackHits(mDebugFile);
          }
        }
      }

      if (transferRunning[iSlice]) {
        SynchronizeEvents(&mEvents->selector[iSlice]);
      }
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Tracks Transfered: %d / %d", *processors()->tpcTrackers[iSlice].NTracks(), *processors()->tpcTrackers[iSlice].NTrackHits());
      }

      processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTracks = processors()->tpcTrackers[iSlice].CommonMemory()->nTracks;
      processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTrackHits = processors()->tpcTrackers[iSlice].CommonMemory()->nTrackHits;

      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Data ready for slice %d, helper thread %d", iSlice, iSlice % (GetDeviceProcessingSettings().nDeviceHelperThreads + 1));
      }
      mSliceOutputReady = iSlice;

      if (param().rec.GlobalTracking) {
        if (iSlice % (NSLICES / 2) == 2) {
          int tmpId = iSlice % (NSLICES / 2) - 1;
          if (iSlice >= NSLICES / 2) {
            tmpId += NSLICES / 2;
          }
          GlobalTracking(tmpId, 0);
          mGlobalTrackingDone[tmpId] = 1;
        }
        for (unsigned int tmpSlice3a = 0; tmpSlice3a < iSlice; tmpSlice3a += GetDeviceProcessingSettings().nDeviceHelperThreads + 1) {
          unsigned int tmpSlice3 = tmpSlice3a + 1;
          if (tmpSlice3 % (NSLICES / 2) < 1) {
            tmpSlice3 -= (NSLICES / 2);
          }
          if (tmpSlice3 >= iSlice) {
            break;
          }

          unsigned int sliceLeft = (tmpSlice3 + (NSLICES / 2 - 1)) % (NSLICES / 2);
          unsigned int sliceRight = (tmpSlice3 + 1) % (NSLICES / 2);
          if (tmpSlice3 >= (int)NSLICES / 2) {
            sliceLeft += NSLICES / 2;
            sliceRight += NSLICES / 2;
          }

          if (tmpSlice3 % (NSLICES / 2) != 1 && mGlobalTrackingDone[tmpSlice3] == 0 && sliceLeft < iSlice && sliceRight < iSlice) {
            GlobalTracking(tmpSlice3, 0);
            mGlobalTrackingDone[tmpSlice3] = 1;
          }

          if (mWriteOutputDone[tmpSlice3] == 0 && mSliceLeftGlobalReady[tmpSlice3] && mSliceRightGlobalReady[tmpSlice3]) {
            WriteOutput(tmpSlice3, 0);
            mWriteOutputDone[tmpSlice3] = 1;
          }
        }
      } else {
        if (iSlice % (GetDeviceProcessingSettings().nDeviceHelperThreads + 1) == 0) {
          WriteOutput(iSlice, 0);
        }
      }
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (transferRunning[iSlice]) {
        ReleaseEvent(&mEvents->selector[iSlice]);
      }
    }

    if (param().rec.GlobalTracking) {
      for (unsigned int tmpSlice3a = 0; tmpSlice3a < NSLICES; tmpSlice3a += GetDeviceProcessingSettings().nDeviceHelperThreads + 1) {
        unsigned int tmpSlice3 = (tmpSlice3a + 1);
        if (tmpSlice3 % (NSLICES / 2) < 1) {
          tmpSlice3 -= (NSLICES / 2);
        }
        if (mGlobalTrackingDone[tmpSlice3] == 0) {
          GlobalTracking(tmpSlice3, 0);
        }
      }
      for (unsigned int tmpSlice3a = 0; tmpSlice3a < NSLICES; tmpSlice3a += GetDeviceProcessingSettings().nDeviceHelperThreads + 1) {
        unsigned int tmpSlice3 = (tmpSlice3a + 1);
        if (tmpSlice3 % (NSLICES / 2) < 1) {
          tmpSlice3 -= (NSLICES / 2);
        }
        if (mWriteOutputDone[tmpSlice3] == 0) {
          while (mSliceLeftGlobalReady[tmpSlice3] == 0 || mSliceRightGlobalReady[tmpSlice3] == 0) {
            ;
          }
          WriteOutput(tmpSlice3, 0);
        }
      }
    }
    WaitForHelperThreads();
  } else {
    mSliceOutputReady = NSLICES;
#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(doGPU ? 1 : GetDeviceProcessingSettings().nThreads)
#endif
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (param().rec.GlobalTracking) {
        GlobalTracking(iSlice, 0);
      }
      WriteOutput(iSlice, 0);
    }
  }

  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    if (processors()->tpcTrackers[iSlice].CommonMemory()->kernelError != 0) {
      const char* errorMsgs[] = GPUCA_ERROR_STRINGS;
      const char* errorMsg = (unsigned)processors()->tpcTrackers[iSlice].CommonMemory()->kernelError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[processors()->tpcTrackers[iSlice].CommonMemory()->kernelError];
      GPUError("GPU Tracker returned Error Code %d (%s) in slice %d (Clusters %d)", processors()->tpcTrackers[iSlice].CommonMemory()->kernelError, errorMsg, iSlice, processors()->tpcTrackers[iSlice].Data().NumberOfHits());
      return (1);
    }
  }

  if (param().rec.GlobalTracking) {
    if (GetDeviceProcessingSettings().debugLevel >= 3) {
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
        GPUInfo("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d", iSlice, processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTracks, processors()->tpcTrackers[iSlice].CommonMemory()->nTracks, processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTrackHits,
                processors()->tpcTrackers[iSlice].CommonMemory()->nTrackHits);
      }
    }
  }

  if (GetDeviceProcessingSettings().debugMask & 1024 && !GetDeviceProcessingSettings().comparableDebutOutput) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      processors()->tpcTrackers[i].DumpOutput(mDebugFile);
    }
  }
  if (DoProfile()) {
    return (1);
  }
  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.nSliceOutTracks[i] = *processors()->tpcTrackers[i].NTracks();
    mIOPtrs.sliceOutTracks[i] = processors()->tpcTrackers[i].Tracks();
    mIOPtrs.nSliceOutClusters[i] = *processors()->tpcTrackers[i].NTrackHits();
    mIOPtrs.sliceOutClusters[i] = processors()->tpcTrackers[i].TrackHits();
  }
  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Slice Tracker finished");
  }
  return 0;
}

int GPUChainTracking::RunTPCTrackingMerger()
{
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCMerging;
  GPUTPCGMMerger& Merger = processors()->tpcMerger;
  GPUTPCGMMerger& MergerShadow = doGPU ? processorsShadow()->tpcMerger : Merger;
  if (Merger.CheckSlices()) {
    return 1;
  }
  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Merger");
  }
  const auto& threadContext = GetThreadContext();
  mRec->SetThreadCounts(RecoStep::TPCMerging);

  HighResTimer& timerUnpack = getTimer<GPUTPCGMMergerUnpack>("GMMergerUnpack");
  HighResTimer& timerMergeWithin = getTimer<GPUTPCGMMergerMergeWithin>("GMMergerMergeWithin");
  HighResTimer& timerMergeSlices = getTimer<GPUTPCGMMergerMergeSlices>("GMMergerMergeSlices");
  HighResTimer& timerMergeCE = getTimer<GPUTPCGMMergerMergeCE>("GMMergerMergeCE");
  HighResTimer& timerCollect = getTimer<GPUTPCGMMergerCollect>("GMMergerCollect");
  HighResTimer& timerClusters = getTimer<GPUTPCGMMergerClusters>("GMMergerClusters");
  HighResTimer& timerCopyToGPU = getTimer<GPUTPCGMMergerCopyToGPU>("GMMergerCopyToGPU");
  HighResTimer& timerCopyToHost = getTimer<GPUTPCGMMergerCopyToHost>("GMMergerCopyToHost");
  HighResTimer& timerFinalize = getTimer<GPUTPCGMMergerFinalize>("GMMergerFinalize");

  Merger.SetMatLUT(processors()->calibObjects.matLUT);
  SetupGPUProcessor(&Merger, true);

  timerUnpack.Start();
  Merger.UnpackSlices();
  timerUnpack.StopAndStart(timerMergeWithin);

  Merger.MergeWithingSlices();
  timerMergeWithin.StopAndStart(timerMergeSlices);

  Merger.MergeSlices();
  timerMergeSlices.StopAndStart(timerMergeCE);

  Merger.MergeCEInit();
  timerMergeCE.StopAndStart(timerCollect);

  Merger.CollectMergedTracks();
  timerCollect.StopAndStart(timerMergeCE);

  Merger.MergeCE();
  timerMergeCE.StopAndStart(timerClusters);

  Merger.PrepareClustersForFit();
  timerClusters.StopAndStart(timerCopyToGPU);

  if (doGPU) {
    SetupGPUProcessor(&Merger, false);
    MergerShadow.OverrideSliceTracker(processorsDevice()->tpcTrackers);
    MergerShadow.SetMatLUT(mFlatObjectsShadow.mCalibObjects.matLUT);
  }

  WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
  TransferMemoryResourceLinkToGPU(RecoStep::TPCMerging, Merger.MemoryResRefit());
  timerCopyToGPU.Stop();

  runKernel<GPUTPCGMMergerTrackFit>({BlockCount(), ThreadCount(), 0}, krnlRunRangeNone);
  SynchronizeGPU();

  timerCopyToHost.Start();
  TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResRefit());
  SynchronizeGPU();
  timerCopyToHost.StopAndStart(timerFinalize);

  Merger.Finalize();
  timerFinalize.StopAndStart(timerCopyToGPU);
  TransferMemoryResourceLinkToGPU(RecoStep::TPCMerging, Merger.MemoryResRefit()); // For compression
  timerCopyToGPU.Stop();

  mIOPtrs.mergedTracks = Merger.OutputTracks();
  mIOPtrs.nMergedTracks = Merger.NOutputTracks();
  mIOPtrs.mergedTrackHits = Merger.Clusters();
  mIOPtrs.nMergedTrackHits = Merger.NOutputTrackClusters();

  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Merger Finished (output clusters %d / input clusters %d)", Merger.NOutputTrackClusters(), Merger.NClusters());
  }
  return 0;
}

int GPUChainTracking::RunTPCCompression()
{
#ifdef HAVE_O2HEADERS
  RecoStep myStep = RecoStep::TPCCompression;
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCCompression;
  GPUTPCCompression& Compressor = processors()->tpcCompressor;
  GPUTPCCompression& CompressorShadow = doGPU ? processorsShadow()->tpcCompressor : Compressor;
  const auto& threadContext = GetThreadContext();
  mRec->SetThreadCounts(RecoStep::TPCCompression);

  Compressor.mMerger = &processors()->tpcMerger;
  Compressor.mNGPUBlocks = BlockCount();
  Compressor.mNMaxClusterSliceRow = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      if (mClusterNativeAccess->nClusters[i][j] > Compressor.mNMaxClusterSliceRow) {
        Compressor.mNMaxClusterSliceRow = mClusterNativeAccess->nClusters[i][j];
      }
    }
  }
  SetupGPUProcessor(&Compressor, true);
  new (Compressor.mMemory) GPUTPCCompression::memory;

  WriteToConstantMemory(myStep, (char*)&processors()->tpcCompressor - (char*)processors(), &CompressorShadow, sizeof(CompressorShadow), 0);
  TransferMemoryResourcesToGPU(myStep, &Compressor, 0);
  runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), 0, RecoStep::TPCCompression}, krnlRunRangeNone, krnlEventNone, CompressorShadow.mClusterStatus, Compressor.mMaxClusters * sizeof(CompressorShadow.mClusterStatus[0]));
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step0attached>({BlockCount(), ThreadCount(), 0}, krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step1unattached>({BlockCount(), ThreadCount(), 0}, krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(myStep, &Compressor, 0);
  SynchronizeGPU();
  memset((void*)&Compressor.mOutput, 0, sizeof(Compressor.mOutput));
  Compressor.mOutput.nTracks = Compressor.mMemory->nStoredTracks;
  Compressor.mOutput.nAttachedClusters = Compressor.mMemory->nStoredAttachedClusters;
  Compressor.mOutput.nUnattachedClusters = Compressor.mMemory->nStoredUnattachedClusters;
  Compressor.mOutput.nAttachedClustersReduced = Compressor.mOutput.nAttachedClusters - Compressor.mOutput.nTracks;
  Compressor.mOutput.nSliceRows = NSLICES * GPUCA_ROW_COUNT;
  Compressor.mOutput.nComppressionModes = param().rec.tpcCompressionModes;
  AllocateRegisteredMemory(Compressor.mMemoryResOutputHost);
  GPUMemCpyAlways(myStep, Compressor.mOutput.nSliceRowClusters, CompressorShadow.mPtrs.nSliceRowClusters, NSLICES * GPUCA_ROW_COUNT * sizeof(Compressor.mOutput.nSliceRowClusters[0]), 0, false);
  GPUMemCpyAlways(myStep, Compressor.mOutput.nTrackClusters, CompressorShadow.mPtrs.nTrackClusters, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.nTrackClusters[0]), 0, false);
  SynchronizeGPU();
  unsigned int offset = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      GPUMemCpyAlways(myStep, Compressor.mOutput.qTotU + offset, CompressorShadow.mPtrs.qTotU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.qTotU[0]), 0, false);
      GPUMemCpyAlways(myStep, Compressor.mOutput.qMaxU + offset, CompressorShadow.mPtrs.qMaxU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.qMaxU[0]), 0, false);
      GPUMemCpyAlways(myStep, Compressor.mOutput.flagsU + offset, CompressorShadow.mPtrs.flagsU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.flagsU[0]), 0, false);
      GPUMemCpyAlways(myStep, Compressor.mOutput.padDiffU + offset, CompressorShadow.mPtrs.padDiffU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.padDiffU[0]), 0, false);
      GPUMemCpyAlways(myStep, Compressor.mOutput.timeDiffU + offset, CompressorShadow.mPtrs.timeDiffU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.timeDiffU[0]), 0, false);
      GPUMemCpyAlways(myStep, Compressor.mOutput.sigmaPadU + offset, CompressorShadow.mPtrs.sigmaPadU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.sigmaPadU[0]), 0, false);
      GPUMemCpyAlways(myStep, Compressor.mOutput.sigmaTimeU + offset, CompressorShadow.mPtrs.sigmaTimeU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.sigmaTimeU[0]), 0, false);
      offset += Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
    }
  }
  offset = 0;
  for (unsigned int i = 0; i < Compressor.mOutput.nTracks; i++) {
    GPUMemCpyAlways(myStep, Compressor.mOutput.qTotA + offset, CompressorShadow.mPtrs.qTotA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.qTotA[0]), 0, false);
    GPUMemCpyAlways(myStep, Compressor.mOutput.qMaxA + offset, CompressorShadow.mPtrs.qMaxA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.qMaxA[0]), 0, false);
    GPUMemCpyAlways(myStep, Compressor.mOutput.flagsA + offset, CompressorShadow.mPtrs.flagsA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.flagsA[0]), 0, false);
    GPUMemCpyAlways(myStep, Compressor.mOutput.sigmaPadA + offset, CompressorShadow.mPtrs.sigmaPadA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.sigmaPadA[0]), 0, false);
    GPUMemCpyAlways(myStep, Compressor.mOutput.sigmaTimeA + offset, CompressorShadow.mPtrs.sigmaTimeA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.sigmaTimeA[0]), 0, false);

    // First index stored with track
    GPUMemCpyAlways(myStep, Compressor.mOutput.rowDiffA + offset - i, CompressorShadow.mPtrs.rowDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.rowDiffA[0]), 0, false);
    GPUMemCpyAlways(myStep, Compressor.mOutput.sliceLegDiffA + offset - i, CompressorShadow.mPtrs.sliceLegDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.sliceLegDiffA[0]), 0, false);
    GPUMemCpyAlways(myStep, Compressor.mOutput.padResA + offset - i, CompressorShadow.mPtrs.padResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.padResA[0]), 0, false);
    GPUMemCpyAlways(myStep, Compressor.mOutput.timeResA + offset - i, CompressorShadow.mPtrs.timeResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.timeResA[0]), 0, false);
    offset += Compressor.mOutput.nTrackClusters[i];
  }
  GPUMemCpyAlways(myStep, Compressor.mOutput.qPtA, CompressorShadow.mPtrs.qPtA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.qPtA[0]), 0, false);
  GPUMemCpyAlways(myStep, Compressor.mOutput.rowA, CompressorShadow.mPtrs.rowA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.rowA[0]), 0, false);
  GPUMemCpyAlways(myStep, Compressor.mOutput.sliceA, CompressorShadow.mPtrs.sliceA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.sliceA[0]), 0, false);
  GPUMemCpyAlways(myStep, Compressor.mOutput.timeA, CompressorShadow.mPtrs.timeA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.timeA[0]), 0, false);
  GPUMemCpyAlways(myStep, Compressor.mOutput.padA, CompressorShadow.mPtrs.padA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.padA[0]), 0, false);

  SynchronizeGPU();

  mIOPtrs.tpcCompressedClusters = &Compressor.mOutput;
#endif
  return 0;
}

int GPUChainTracking::RunTRDTracking()
{
  if (!processors()->trdTracker.IsInitialized()) {
    return 1;
  }
  std::vector<GPUTRDTrack> tracksTPC;
  std::vector<int> tracksTPCLab;
  GPUTRDTracker& Tracker = processors()->trdTracker;
  mRec->SetThreadCounts(RecoStep::TRDTracking);

  for (unsigned int i = 0; i < mIOPtrs.nMergedTracks; i++) {
    const GPUTPCGMMergedTrack& trk = mIOPtrs.mergedTracks[i];
    if (!trk.OK()) {
      continue;
    }
    if (trk.Looper()) {
      continue;
    }
    if (param().rec.NWaysOuter) {
      tracksTPC.emplace_back(trk.OuterParam());
    } else {
      tracksTPC.emplace_back(trk);
    }
    tracksTPC.back().SetTPCtrackId(i);
    tracksTPCLab.push_back(-1);
  }

  Tracker.Reset();

  Tracker.SetMaxData(processors()->ioPtrs);
  if (GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    AllocateRegisteredMemory(Tracker.MemoryTracks());
    AllocateRegisteredMemory(Tracker.MemoryTracklets()); // TODO: Is this needed?
  }

  for (unsigned int iTracklet = 0; iTracklet < mIOPtrs.nTRDTracklets; ++iTracklet) {
    if (Tracker.LoadTracklet(mIOPtrs.trdTracklets[iTracklet], mIOPtrs.trdTrackletsMC ? mIOPtrs.trdTrackletsMC[iTracklet].mLabel : nullptr)) {
      return 1;
    }
  }

  for (unsigned int iTrack = 0; iTrack < tracksTPC.size(); ++iTrack) {
    if (Tracker.LoadTrack(tracksTPC[iTrack], tracksTPCLab[iTrack])) {
      return 1;
    }
  }

  Tracker.DoTracking();

  mIOPtrs.nTRDTracks = Tracker.NTracks();
  mIOPtrs.trdTracks = Tracker.Tracks();

  return 0;
}

int GPUChainTracking::DoTRDGPUTracking()
{
#ifdef HAVE_O2HEADERS
  bool doGPU = GetRecoStepsGPU() & RecoStep::TRDTracking;
  GPUTRDTracker& Tracker = processors()->trdTracker;
  GPUTRDTracker& TrackerShadow = doGPU ? processorsShadow()->trdTracker : Tracker;

  const auto& threadContext = GetThreadContext();
  SetupGPUProcessor(&Tracker, false);
  TrackerShadow.SetGeometry(reinterpret_cast<GPUTRDGeometry*>(mFlatObjectsDevice.mCalibObjects.trdGeometry));

  WriteToConstantMemory(RecoStep::TRDTracking, (char*)&processors()->trdTracker - (char*)processors(), &TrackerShadow, sizeof(TrackerShadow), 0);
  TransferMemoryResourcesToGPU(RecoStep::TRDTracking, &Tracker);

  runKernel<GPUTRDTrackerGPU>({BlockCount(), TRDThreadCount(), 0}, krnlRunRangeNone);
  SynchronizeGPU();

  TransferMemoryResourcesToHost(RecoStep::TRDTracking, &Tracker);
  SynchronizeGPU();

  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("GPU TRD tracker Finished");
  }
#endif
  return (0);
}

int GPUChainTracking::RunChain()
{
  if (GetDeviceProcessingSettings().runCompressionStatistics && mCompressionStatistics == nullptr) {
    mCompressionStatistics.reset(new GPUTPCClusterStatistics);
  }
  const bool needQA = GPUQA::QAAvailable() && (GetDeviceProcessingSettings().runQA || (GetDeviceProcessingSettings().eventDisplay && mIOPtrs.nMCInfosTPC));
  if (needQA && mQA->IsInitialized() == false) {
    if (mQA->InitQA()) {
      return 1;
    }
  }
  static HighResTimer timerTracking, timerMerger, timerQA, timerTransform, timerCompression, timerClusterer;
  int nCount = mRec->getNEventsProcessed();

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

  PrepareDebugOutput();

  if (GetRecoSteps().isSet(RecoStep::TPCClusterFinding) && mIOPtrs.tpcPackedDigits) {
    timerClusterer.Start();
    if (param().rec.fwdTPCDigitsAsClusters) {
      ForwardTPCDigits();
    } else {
      RunTPCClusterizer();
    }
    timerClusterer.Stop();
  }

  if (GetRecoSteps().isSet(RecoStep::TPCConversion) && mIOPtrs.clustersNative) {
    timerTransform.Start();
    ConvertNativeToClusterData();
    timerTransform.Stop();
  }

  timerTracking.Start();
  if (GetRecoSteps().isSet(RecoStep::TPCSliceTracking) && RunTPCTrackingSlices()) {
    return 1;
  }
  timerTracking.Stop();

  timerMerger.Start();
  for (unsigned int i = 0; i < NSLICES; i++) {
    // GPUInfo("slice %d clusters %d tracks %d", i, mClusterData[i].NumberOfClusters(), processors()->tpcTrackers[i].Output()->NTracks());
    processors()->tpcMerger.SetSliceData(i, processors()->tpcTrackers[i].Output());
  }
  if (GetRecoSteps().isSet(RecoStep::TPCMerging) && RunTPCTrackingMerger()) {
    return 1;
  }
  timerMerger.Stop();

  if (GetRecoSteps().isSet(RecoStep::TPCCompression) && mIOPtrs.clustersNative) {
    timerCompression.Start();
    RunTPCCompression();
    if (GetDeviceProcessingSettings().runCompressionStatistics) {
      mCompressionStatistics->RunStatistics(mClusterNativeAccess.get(), &processors()->tpcCompressor.mOutput, param());
    }
    timerCompression.Stop();
  }

  if (needQA) {
    timerQA.Start();
    mQA->RunQA(!GetDeviceProcessingSettings().runQA);
    timerQA.Stop();
  }

  if (GetDeviceProcessingSettings().debugLevel >= 0) {
    char nAverageInfo[16] = "";
    if (nCount > 1) {
      sprintf(nAverageInfo, " (%d)", nCount);
    }
    printf("Tracking Time: %'d us%s\n", (int)(1000000 * timerTracking.GetElapsedTime() / nCount), nAverageInfo);
    printf("Merging and Refit Time: %'d us\n", (int)(1000000 * timerMerger.GetElapsedTime() / nCount));
    if (GetDeviceProcessingSettings().runQA) {
      printf("QA Time: %'d us\n", (int)(1000000 * timerQA.GetElapsedTime() / nCount));
    }
    if (mIOPtrs.tpcPackedDigits) {
      printf("TPC Clusterizer Time: %'d us\n", (int)(1000000 * timerClusterer.GetElapsedTime() / nCount));
    }
    if (mIOPtrs.clustersNative) {
      printf("TPC Transformation Time: %'d us\n", (int)(1000000 * timerTransform.GetElapsedTime() / nCount));
    }
    if (mIOPtrs.clustersNative) {
      printf("TPC Compression Time: %'d us\n", (int)(1000000 * timerCompression.GetElapsedTime() / nCount));
    }
  }

  if (GetRecoSteps().isSet(RecoStep::TRDTracking)) {
    if (mIOPtrs.nTRDTracklets) {
      HighResTimer timer;
      timer.Start();
      if (RunTRDTracking()) {
        return 1;
      }
      if (GetDeviceProcessingSettings().debugLevel >= 0) {
        printf("TRD tracking time: %'d us\n", (int)(1000000 * timer.GetCurrentElapsedTime()));
      }
    } else {
      processors()->trdTracker.Reset();
    }
  }

  if (GetDeviceProcessingSettings().resetTimers) {
    timerTracking.Reset();
    timerMerger.Reset();
    timerQA.Reset();
    timerTransform.Reset();
    timerCompression.Reset();
  }

  if (GetDeviceProcessingSettings().eventDisplay) {
    if (!mDisplayRunning) {
      if (mEventDisplay->StartDisplay()) {
        return (1);
      }
      mDisplayRunning = true;
    } else {
      mEventDisplay->ShowNextEvent();
    }

    if (GetDeviceProcessingSettings().eventDisplay->EnableSendKey()) {
      while (kbhit()) {
        getch();
      }
      GPUInfo("Press key for next event!");
    }

    int iKey;
    do {
      Sleep(10);
      if (GetDeviceProcessingSettings().eventDisplay->EnableSendKey()) {
        iKey = kbhit() ? getch() : 0;
        if (iKey == 'q') {
          GetDeviceProcessingSettings().eventDisplay->mDisplayControl = 2;
        } else if (iKey == 'n') {
          break;
        } else if (iKey) {
          while (GetDeviceProcessingSettings().eventDisplay->mSendKey != 0) {
            Sleep(1);
          }
          GetDeviceProcessingSettings().eventDisplay->mSendKey = iKey;
        }
      }
    } while (GetDeviceProcessingSettings().eventDisplay->mDisplayControl == 0);
    if (GetDeviceProcessingSettings().eventDisplay->mDisplayControl == 2) {
      mDisplayRunning = false;
      GetDeviceProcessingSettings().eventDisplay->DisplayExit();
      DeviceProcessingSettings().eventDisplay = nullptr;
      return (2);
    }
    GetDeviceProcessingSettings().eventDisplay->mDisplayControl = 0;
    GPUInfo("Loading next event");

    mEventDisplay->WaitForNextEvent();
  }

  PrintDebugOutput();

  //PrintMemoryRelations();
  return 0;
}

int GPUChainTracking::HelperReadEvent(int iSlice, int threadId, GPUReconstructionHelpers::helperParam* par) { return ReadEvent(iSlice, threadId); }

int GPUChainTracking::HelperOutput(int iSlice, int threadId, GPUReconstructionHelpers::helperParam* par)
{
  int mustRunSlice19 = 0;
  if (param().rec.GlobalTracking) {
    int realSlice = iSlice + 1;
    if (realSlice % (NSLICES / 2) < 1) {
      realSlice -= NSLICES / 2;
    }

    if (realSlice % (NSLICES / 2) != 1) {
      GlobalTracking(realSlice, threadId);
    }

    if (realSlice == 19) {
      mustRunSlice19 = 1;
    } else {
      while (mSliceLeftGlobalReady[realSlice] == 0 || mSliceRightGlobalReady[realSlice] == 0) {
        if (par->reset) {
          return 1;
        }
      }
      WriteOutput(realSlice, threadId);
    }
  } else {
    while (mSliceOutputReady < iSlice) {
      if (par->reset) {
        return 1;
      }
    }
    WriteOutput(iSlice, threadId);
  }
  if (iSlice >= par->count - (GetDeviceProcessingSettings().nDeviceHelperThreads + 1) && mustRunSlice19) {
    while (mSliceLeftGlobalReady[19] == 0 || mSliceRightGlobalReady[19] == 0) {
      if (par->reset) {
        return 1;
      }
    }
    WriteOutput(19, threadId);
  }
  return 0;
}
