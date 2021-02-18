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
#include "GPUTrackingInputProvider.h"

#ifdef HAVE_O2HEADERS
#include "Digit.h"
#include "GPUTPCClusterStatistics.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "GPUHostDataTypes.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#include "TPCFastTransform.h"

#include "utils/linux_helpers.h"
using namespace GPUCA_NAMESPACE::gpu;

#include "GPUO2DataTypes.h"

using namespace o2::tpc;
using namespace o2::trd;

GPUChainTracking::GPUChainTracking(GPUReconstruction* rec, unsigned int maxTPCHits, unsigned int maxTRDTracklets) : GPUChain(rec), mIOPtrs(processors()->ioPtrs), mInputsHost(new GPUTrackingInputProvider), mInputsShadow(new GPUTrackingInputProvider), mClusterNativeAccess(new ClusterNativeAccess), mMaxTPCHits(maxTPCHits), mMaxTRDTracklets(maxTRDTracklets)
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
  if (!ValidateSettings()) {
    GPUError("Invalid GPU Reconstruction Settings");
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
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->calibObjects - (char*)processors(), &mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects), -1); // First initialization, for users not using RunChain
  }

  if (GetDeviceProcessingSettings().debugLevel >= 6) {
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
    mIOPtrs.clustersNative = tmp;
  }
  processors()->tpcConverter.mClustersNative = tmp;

  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    processors()->tpcTrackers[iSlice].Data().SetClusterData(nullptr, mIOPtrs.clustersNative->nClustersSector[iSlice], mIOPtrs.clustersNative->clusterOffset[iSlice][0]);
    if (GetRecoStepsGPU() & RecoStep::TPCSliceTracking) {
      processorsShadow()->tpcTrackers[iSlice].Data().SetClusterData(nullptr, mIOPtrs.clustersNative->nClustersSector[iSlice], mIOPtrs.clustersNative->clusterOffset[iSlice][0]); // TODO: A bit of a hack, but we have to make sure this stays in sync
    }
  }
  processors()->tpcCompressor.mMaxClusters = mIOPtrs.clustersNative->nClustersTotal;
  mRec->MemoryScalers()->nTPCHits = mIOPtrs.clustersNative->nClustersTotal;
  GPUInfo("Event has %d TPC Clusters, %d TRD Tracklets", tmp->nClustersTotal, mIOPtrs.nTRDTracklets);
#endif
}

int GPUChainTracking::PrepareEvent()
{
  mRec->MemoryScalers()->nTRDTracklets = mIOPtrs.nTRDTracklets;
  if (mIOPtrs.tpcPackedDigits || mIOPtrs.tpcZS) {
#ifdef HAVE_O2HEADERS
    mRec->MemoryScalers()->nTPCdigits = 0;
    size_t maxDigits = 0;
    size_t maxPages = 0;
    size_t nPagesTotal = 0;
    unsigned int maxClusters[NSLICES] = {0};
    if (mIOPtrs.tpcZS) {
      if (param().rec.fwdTPCDigitsAsClusters) {
        throw std::runtime_error("Forwading zero-suppressed hits not supported");
      }
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
        size_t nPages = 0;
        for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
          for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
            nPages += mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
          }
        }
        processors()->tpcClusterer[iSlice].mPmemory->counters.nPages = nPages;
        if (nPages > maxPages) {
          maxPages = nPages;
        }
        nPagesTotal += nPages;
      }
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
        processors()->tpcClusterer[iSlice].SetNMaxDigits(0, maxPages);
        if (mRec->IsGPU()) {
          processorsShadow()->tpcClusterer[iSlice].SetNMaxDigits(0, maxPages);
        }
        AllocateRegisteredMemory(processors()->tpcClusterer[iSlice].mZSOffsetId);
      }
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      unsigned int nDigits = 0;
      if (mIOPtrs.tpcZS) {
        GPUTPCClusterFinder::ZSOffset* o = processors()->tpcClusterer[iSlice].mPzsOffsets;
        for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
          if (!(mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
            processors()->tpcClusterer[iSlice].mPzsOffsets[j] = GPUTPCClusterFinder::ZSOffset{nDigits, j, 0};
          }
          unsigned short num = 0;
          for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
            for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]; l++) {
              if ((mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
                *(o++) = GPUTPCClusterFinder::ZSOffset{nDigits, j, num++};
              }
              const unsigned char* const page = ((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE;
              const TPCZSHDR* const hdr = (const TPCZSHDR*)(page + sizeof(o2::header::RAWDataHeader));
              nDigits += hdr->nADCsamples;
            }
          }
        }
        processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits = nDigits;
      } else {
        nDigits = mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice];
      }
      mRec->MemoryScalers()->nTPCdigits += nDigits;
      if (nDigits > maxDigits) {
        maxDigits = nDigits;
      }
      maxClusters[iSlice] = param().rec.fwdTPCDigitsAsClusters ? nDigits : mRec->MemoryScalers()->NTPCClusters(nDigits);
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processors()->tpcTrackers[iSlice].Data().SetClusterData(nullptr, maxClusters[iSlice], 0); // TODO: fixme
      // Distribute maximum digits, so that we can reuse the memory easily
      processors()->tpcClusterer[iSlice].SetNMaxDigits(maxDigits, maxPages);
    }
    mRec->MemoryScalers()->nTPCHits = param().rec.fwdTPCDigitsAsClusters ? mRec->MemoryScalers()->nTPCdigits : mRec->MemoryScalers()->NTPCClusters(mRec->MemoryScalers()->nTPCdigits);
    processors()->tpcCompressor.mMaxClusters = mRec->MemoryScalers()->nTPCHits;
    processors()->tpcConverter.mNClustersTotal = mRec->MemoryScalers()->nTPCHits;
    if (mIOPtrs.tpcZS) {
      GPUInfo("Event has %lld 8kb TPC ZS pages, %lld digits", (long long int)nPagesTotal, (long long int)mRec->MemoryScalers()->nTPCdigits);
    } else {
      GPUInfo("Event has %lld TPC Digits", (long long int)mRec->MemoryScalers()->nTPCdigits);
    }
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

  UpdateShadowProcessors();
  return 0;
}

void GPUChainTracking::UpdateShadowProcessors()
{
  if (mRec->IsGPU()) {
    memcpy((void*)processorsShadow(), (const void*)processors(), sizeof(*processors()));
    mRec->ResetDeviceProcessorTypes();
  }
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
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
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
  mIOPtrs.clustersNative = mClusterNativeAccess->nClustersTotal ? mClusterNativeAccess.get() : nullptr;
  AllocateIOMemoryHelper(mIOPtrs.nMCLabelsTPC, mIOPtrs.mcLabelsTPC, mIOMem.mcLabelsTPC);
  AllocateIOMemoryHelper(mIOPtrs.nMCInfosTPC, mIOPtrs.mcInfosTPC, mIOMem.mcInfosTPC);
  AllocateIOMemoryHelper(mIOPtrs.nMergedTracks, mIOPtrs.mergedTracks, mIOMem.mergedTracks);
  AllocateIOMemoryHelper(mIOPtrs.nMergedTrackHits, mIOPtrs.mergedTrackHits, mIOMem.mergedTrackHits);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracks, mIOPtrs.trdTracks, mIOMem.trdTracks);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdTracklets, mIOMem.trdTracklets);
  AllocateIOMemoryHelper(mIOPtrs.nTRDTrackletsMC, mIOPtrs.trdTrackletsMC, mIOMem.trdTrackletsMC);
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
  if (tmpExt->nClustersTotal > convert.mNClustersTotal) {
    GPUError("Too many input clusters in conversion (expected <= %ld)\n", convert.mNClustersTotal);
    for (unsigned int i = 0; i < NSLICES; i++) {
      mIOPtrs.nClusterData[i] = 0;
      processors()->tpcTrackers[i].Data().SetClusterData(nullptr, 0, 0);
    }
    return 1;
  }
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
  runKernel<GPUTPCConvertKernel>({NSLICES * GPUCA_ROW_COUNT, ConverterThreadCount(), 0}, krnlRunRangeNone, krnlEventNone);
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
    if (GetDeviceProcessingSettings().registerStandaloneInputMemory) {
      mRec->registerMemoryForGPU(mIOMem.clusterData[i].get(), mIOPtrs.nClusterData[i] * sizeof(*mIOPtrs.clusterData[i]));
    }
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
  if (GetDeviceProcessingSettings().registerStandaloneInputMemory) {
    mRec->registerMemoryForGPU(mIOMem.clustersNative.get(), mClusterNativeAccess->nClustersTotal * sizeof(*mClusterNativeAccess->clustersLinear));
  }
}

void GPUChainTracking::ConvertZSEncoder(bool zs12bit)
{
#ifdef HAVE_O2HEADERS
  mTPCZSSizes.reset(new unsigned int[NSLICES * GPUTrackingInOutZS::NENDPOINTS]);
  mTPCZSPtrs.reset(new void*[NSLICES * GPUTrackingInOutZS::NENDPOINTS]);
  mTPCZS.reset(new GPUTrackingInOutZS);
  GPUReconstructionConvert::RunZSEncoder<deprecated::PackedDigit>(*mIOPtrs.tpcPackedDigits, &mTPCZSBuffer, mTPCZSSizes.get(), nullptr, nullptr, param(), zs12bit, true);
  GPUReconstructionConvert::RunZSEncoderCreateMeta(mTPCZSBuffer.get(), mTPCZSSizes.get(), mTPCZSPtrs.get(), mTPCZS.get());
  mIOPtrs.tpcZS = mTPCZS.get();
  if (GetDeviceProcessingSettings().registerStandaloneInputMemory) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[i].count[j]; k++) {
          mRec->registerMemoryForGPU(mIOPtrs.tpcZS->slice[i].zsPtr[j][k], mIOPtrs.tpcZS->slice[i].nZSPtr[j][k] * TPCZSHDR::TPC_ZS_PAGE_SIZE);
        }
      }
    }
  }
#endif
}

void GPUChainTracking::ConvertZSFilter(bool zs12bit)
{
  GPUReconstructionConvert::RunZSFilter(mIOMem.tpcDigits, mIOPtrs.tpcPackedDigits->tpcDigits, mDigitMap->nTPCDigits, mIOPtrs.tpcPackedDigits->nTPCDigits, param(), zs12bit);
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

  HighResTimer& timer = getTimer<GPUTPCGlobalTracking>("GlobalTracking", threadId);
  timer.Start();
  processors()->tpcTrackers[iSlice].PerformGlobalTracking(processors()->tpcTrackers[sliceLeft], processors()->tpcTrackers[sliceRight]);
  timer.Stop();

  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("GPU Tracker finished Global Tracking for slice %d on thread %d\n", iSlice, threadId);
  }
  return (0);
}

void GPUChainTracking::RunTPCClusterizer_compactPeaks(GPUTPCClusterFinder& clusterer, GPUTPCClusterFinder& clustererShadow, int stage, bool doGPU, int lane)
{
#ifdef HAVE_O2HEADERS
  auto& in = stage ? clustererShadow.mPpeakPositions : clustererShadow.mPpositions;
  auto& out = stage ? clustererShadow.mPfilteredPeakPositions : clustererShadow.mPpeakPositions;
  if (doGPU) {
    const unsigned int iSlice = clusterer.mISlice;
    auto& count = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nDigits;

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
          runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::nativeScanUpStart>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, stage);
        } else {
          runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::nativeScanUp>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, tmpCount);
        }
        tmpCount = (tmpCount + clusterer.mScanWorkGroupSize - 1) / clusterer.mScanWorkGroupSize;
      }

      runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::nativeScanTop>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, nSteps, tmpCount);

      for (unsigned int i = nSteps - 1; i > 1; i--) {
        tmpCount = counts[i - 1];
        runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::nativeScanDown>(GetGrid(tmpCount - clusterer.mScanWorkGroupSize, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, clusterer.mScanWorkGroupSize, tmpCount);
      }
    }

    runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::compact>(GetGrid(count, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, 1, stage, in, out);
  } else {
    auto& nOut = stage ? clusterer.mPmemory->counters.nClusters : clusterer.mPmemory->counters.nPeaks;
    auto& nIn = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nDigits;
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
    if (mIOPtrs.tpcZS) {
      processorsShadow()->ioPtrs.tpcZS = mInputsShadow->mPzsMeta;
      WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    }
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)processors()->tpcClusterer - (char*)processors(), processorsShadow()->tpcClusterer, sizeof(GPUTPCClusterFinder) * NSLICES, mRec->NStreams() - 1, &mEvents->init);
  }
  SynchronizeGPU();

  static std::vector<o2::tpc::ClusterNative> clsMemory; // TODO: remove static temporary, data should remain on the GPU anyway
  size_t nClsTotal = 0;
  ClusterNativeAccess* tmp = mClusterNativeAccess.get();
  size_t pos = 0;
  clsMemory.reserve(mRec->MemoryScalers()->nTPCHits);

  // setup MC Labels
  bool propagateMCLabels = !doGPU && GetDeviceProcessingSettings().runMC && processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC != nullptr;

  auto* digitsMC = propagateMCLabels ? processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC : nullptr;

  GPUTPCLinearLabels mcLinearLabels;
  if (propagateMCLabels) {
    mcLinearLabels.header.reserve(mRec->MemoryScalers()->nTPCHits);
    mcLinearLabels.data.reserve(mRec->MemoryScalers()->nTPCHits * 16); // Assumption: cluster will have less than 16 labels on average
  }

  for (unsigned int iSliceBase = 0; iSliceBase < NSLICES; iSliceBase += GetDeviceProcessingSettings().nTPCClustererLanes) {
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
      SetupGPUProcessor(&clusterer, false);
      clusterer.mPmemory->counters.nPeaks = clusterer.mPmemory->counters.nClusters = 0;

      if (propagateMCLabels) {
        clusterer.PrepareMC();
        clusterer.mPinputLabels = digitsMC->v[iSlice];
        // TODO: Why is the number of header entries in truth container
        // sometimes larger than the number of digits?
        assert(clusterer.mPinputLabels->getIndexedSize() >= mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice]);
      }

      unsigned int nPagesTotal = 0;
      if (mIOPtrs.tpcZS) {
        for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
          for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
            nPagesTotal += mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
          }
        }
        if (doGPU) {
          unsigned int nPagesSector = 0;
          for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
            unsigned int nPages = 0;
            mInputsHost->mPzsMeta->slice[iSlice].zsPtr[j] = &mInputsShadow->mPzsPtrs[iSlice * GPUTrackingInOutZS::NENDPOINTS + j];
            mInputsHost->mPzsPtrs[iSlice * GPUTrackingInOutZS::NENDPOINTS + j] = clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE;
            for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
              GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE, mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k], mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k] * TPCZSHDR::TPC_ZS_PAGE_SIZE, lane, true);
              nPages += mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
            }
            mInputsHost->mPzsMeta->slice[iSlice].nZSPtr[j] = &mInputsShadow->mPzsSizes[iSlice * GPUTrackingInOutZS::NENDPOINTS + j];
            mInputsHost->mPzsSizes[iSlice * GPUTrackingInOutZS::NENDPOINTS + j] = nPages;
            mInputsHost->mPzsMeta->slice[iSlice].count[j] = 1;
            nPagesSector += nPages;
          }
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceZS, lane);
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mZSOffsetId, lane);
        }
      } else {
        auto* inDigits = mIOPtrs.tpcPackedDigits;
        size_t numDigits = inDigits->nTPCDigits[iSlice];
        clusterer.mPmemory->counters.nDigits = numDigits;
        if (doGPU) {
          GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPdigits, inDigits->tpcDigits[iSlice], sizeof(clustererShadow.mPdigits[0]) * numDigits, lane, true);
        } else {
          clustererShadow.mPdigits = const_cast<deprecated::Digit*>(inDigits->tpcDigits[iSlice]); // TODO: Needs fixing (invalid const cast)
        }
      }
      TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);

      // These buffers only have to be cleared once entirely. The 'resetMaps' kernel
      // takes care of subsequent clean ups.
      if (iSliceBase == 0) {
        using ChargeMapType = decltype(*clustererShadow.mPchargeMap);
        using PeakMapType = decltype(*clustererShadow.mPpeakMap);
        runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPchargeMap, TPCMapMemoryLayout<ChargeMapType>::items() * sizeof(ChargeMapType));
        runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPpeakMap, TPCMapMemoryLayout<PeakMapType>::items() * sizeof(PeakMapType));
      }
      runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), lane, RecoStep::TPCClusterFinding}, krnlRunRangeNone, {}, clustererShadow.mPclusterInRow, GPUCA_ROW_COUNT * sizeof(*clustererShadow.mPclusterInRow));
      DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, mDebugFile, "Zeroed Charges");
      if (mIOPtrs.tpcZS ? (nPagesTotal == 0) : (clusterer.mPmemory->counters.nDigits == 0)) {
        continue;
      }

      if (mIOPtrs.tpcZS) {
        int firstHBF = mIOPtrs.tpcZS->slice[0].count[0] ? o2::raw::RDHUtils::getHeartBeatOrbit((const o2::header::RAWDataHeader*)mIOPtrs.tpcZS->slice[0].zsPtr[0][0]) : 0;
        runKernel<GPUTPCCFDecodeZS, GPUTPCCFDecodeZS::decodeZS>({doGPU ? clusterer.mPmemory->counters.nPages : GPUTrackingInOutZS::NENDPOINTS, CFDecodeThreadCount(), lane}, {iSlice}, {}, firstHBF);
        TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        SynchronizeStream(lane);
      } else {
        runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillFromDigits>(GetGrid(clusterer.mPmemory->counters.nDigits, ClustererThreadCount(), lane), {iSlice}, {});
      }
      if (DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpDigits, mDebugFile)) {
        clusterer.DumpChargeMap(mDebugFile, "Charges");
      }

      if (propagateMCLabels) {
        runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillIndexMap>(GetGrid(clusterer.mPmemory->counters.nDigits, ClustererThreadCount(), lane), {iSlice}, {});
      }

      runKernel<GPUTPCCFPeakFinder, GPUTPCCFPeakFinder::findPeaks>(GetGrid(clusterer.mPmemory->counters.nDigits, ClustererThreadCount(), lane), {iSlice}, {});
      DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpPeaks, mDebugFile);

      RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 0, doGPU, lane);
      TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
      DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpPeaksCompacted, mDebugFile);
    }
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
      SynchronizeStream(lane);
      if (clusterer.mPmemory->counters.nPeaks == 0) {
        continue;
      }
      runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::noiseSuppression>(GetGrid(clusterer.mPmemory->counters.nPeaks, ClustererThreadCount(), lane), {iSlice}, {});
      runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::updatePeaks>(GetGrid(clusterer.mPmemory->counters.nPeaks, ClustererThreadCount(), lane), {iSlice}, {});
      DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaks, mDebugFile);

      RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 1, doGPU, lane);
      TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
      DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaksCompacted, mDebugFile);

      SynchronizeStream(lane);
      RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 1, doGPU, lane);
      TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
    }
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      SynchronizeStream(lane);
      if (clusterer.mPmemory->counters.nClusters == 0) {
        continue;
      }

      runKernel<GPUTPCCFDeconvolution, GPUTPCCFDeconvolution::countPeaks>(GetGrid(clusterer.mPmemory->counters.nDigits, ClustererThreadCount(), lane), {iSlice}, {});
      DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, mDebugFile, "Split Charges");

      runKernel<GPUTPCCFClusterizer, GPUTPCCFClusterizer::computeClusters>(GetGrid(clusterer.mPmemory->counters.nClusters, ClustererThreadCount(), lane), {iSlice}, {});
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        printf("Lane %d: Found clusters: digits %d peaks %d clusters %d\n", lane, (int)clusterer.mPmemory->counters.nDigits, (int)clusterer.mPmemory->counters.nPeaks, (int)clusterer.mPmemory->counters.nClusters);
      }
      TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, lane);
      DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpCountedPeaks, mDebugFile);
      DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpClusters, mDebugFile);
    }
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      SynchronizeStream(lane);
      if (clusterer.mPmemory->counters.nDigits) {
        runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::resetMaps>(GetGrid(clusterer.mPmemory->counters.nDigits, ClustererThreadCount(), lane), {iSlice}, {});
      }
      if (clusterer.mPmemory->counters.nClusters == 0) {
        for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
          tmp->nClusters[iSlice][j] = 0;
        }
        continue;
      }
      nClsTotal += clusterer.mPmemory->counters.nClusters;
      clsMemory.resize(nClsTotal);
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        memcpy((void*)&clsMemory[pos], (const void*)&clusterer.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], clusterer.mPclusterInRow[j] * sizeof(clsMemory[0]));
        tmp->nClusters[iSlice][j] = clusterer.mPclusterInRow[j];
        if (GetDeviceProcessingSettings().debugLevel >= 4) {
          std::sort(&clsMemory[pos], &clsMemory[pos + clusterer.mPclusterInRow[j]]);
        }
        pos += clusterer.mPclusterInRow[j];
      }

      if (not propagateMCLabels) {
        continue;
      }

      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::setRowOffsets>(GetGrid(GPUCA_ROW_COUNT, ClustererThreadCount(), lane), {iSlice}, {});
      GPUTPCCFMCLabelFlattener::setGlobalOffsetsAndAllocate(clusterer, mcLinearLabels);
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::flatten>(GetGrid(clusterer.mPclusterInRow[j], ClustererThreadCount(), lane), {iSlice}, {}, j, &mcLinearLabels);
      }
    }
  }

  static o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcLabels;

  assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);

  mcLabels.setFrom(mcLinearLabels.header, mcLinearLabels.data);

  tmp->clustersLinear = clsMemory.data();
  tmp->clustersMCTruth = propagateMCLabels ? &mcLabels : nullptr;
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
    processors()->tpcTrackers[iSlice].SetupCommonMemory();
  }
  if (doGPU) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processorsShadow()->tpcTrackers[iSlice].GPUParametersConst()->gpumem = (char*)mRec->DeviceMemoryBase();
      // Initialize Startup Constants
      processors()->tpcTrackers[iSlice].GPUParameters()->nextStartHit = ((ConstructorBlockCount() + NSLICES - 1 - iSlice) / NSLICES) * ConstructorThreadCount();
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
    if (!doGPU && trk.CheckEmptySlice() && GetDeviceProcessingSettings().debugLevel == 0) {
      continue;
    }

    if (GetDeviceProcessingSettings().debugLevel >= 6) {
      mDebugFile << "\n\nReconstruction: Slice " << iSlice << "/" << NSLICES << std::endl;
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
    DoDebugAndDump(RecoStep::TPCSliceTracking, 4, trk, &GPUTPCTracker::DumpLinks, mDebugFile);

    runKernel<GPUTPCStartHitsFinder>({GPUCA_ROW_COUNT - 6, HitsFinderThreadCount(), useStream}, {iSlice});
#ifdef GPUCA_SORT_STARTHITS_GPU
    if (doGPU) {
      runKernel<GPUTPCStartHitsSorter>({HitsSorterBlockCount(), HitsSorterThreadCount(), useStream}, {iSlice});
    }
#endif
    DoDebugAndDump(RecoStep::TPCSliceTracking, 32, trk, &GPUTPCTracker::DumpStartHits, mDebugFile);

    if (GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
      trk.UpdateMaxData();
      AllocateRegisteredMemory(trk.MemoryResTracklets());
      AllocateRegisteredMemory(trk.MemoryResOutput());
    }

    if (!(doGPU || GetDeviceProcessingSettings().debugLevel >= 1) || GetDeviceProcessingSettings().trackletConstructorInPipeline) {
      runKernel<GPUTPCTrackletConstructor>({ConstructorBlockCount(), ConstructorThreadCount(), useStream}, {iSlice});
      DoDebugAndDump(RecoStep::TPCSliceTracking, 128, trk, &GPUTPCTracker::DumpTrackletHits, mDebugFile);
      if (GetDeviceProcessingSettings().debugMask & 256 && !GetDeviceProcessingSettings().comparableDebutOutput) {
        trk.DumpHitWeights(mDebugFile);
      }
    }

    if (!(doGPU || GetDeviceProcessingSettings().debugLevel >= 1) || GetDeviceProcessingSettings().trackletSelectorInPipeline) {
      runKernel<GPUTPCTrackletSelector>({SelectorBlockCount(), SelectorThreadCount(), useStream}, {iSlice});
      TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, trk.MemoryResCommon(), useStream, &mEvents->selector[iSlice]);
      streamMap[iSlice] = useStream;
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Slice %u, Number of tracks: %d", iSlice, *trk.NTracks());
      }
      DoDebugAndDump(RecoStep::TPCSliceTracking, 512, trk, &GPUTPCTracker::DumpTrackHits, mDebugFile);
    }

    if (!(doGPU || GetDeviceProcessingSettings().debugLevel >= 1)) {
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

  if (doGPU || GetDeviceProcessingSettings().debugLevel >= 1) {
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
          DoDebugAndDump(RecoStep::TPCSliceTracking, 128, processors()->tpcTrackers[iSlice], &GPUTPCTracker::DumpTrackletHits, mDebugFile);
        }
      }

      unsigned int runSlices = 0;
      int useStream = 0;
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice += runSlices) {
        if (runSlices < (unsigned int)GetDeviceProcessingSettings().trackletSelectorSlices) {
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

    mSliceSelectorReady = 0;

    if (param().rec.GlobalTracking) {
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
      mSliceSelectorReady = iSlice;

      if (param().rec.GlobalTracking) {
        for (unsigned int tmpSlice2a = 0; tmpSlice2a <= iSlice; tmpSlice2a += GetDeviceProcessingSettings().nDeviceHelperThreads + 1) {
          unsigned int tmpSlice2 = GPUTPCTracker::GlobalTrackingSliceOrder(tmpSlice2a);
          unsigned int sliceLeft = (tmpSlice2 + (NSLICES / 2 - 1)) % (NSLICES / 2);
          unsigned int sliceRight = (tmpSlice2 + 1) % (NSLICES / 2);
          if (tmpSlice2 >= (int)NSLICES / 2) {
            sliceLeft += NSLICES / 2;
            sliceRight += NSLICES / 2;
          }

          if (tmpSlice2 <= iSlice && sliceLeft <= iSlice && sliceRight <= iSlice && mWriteOutputDone[tmpSlice2] == 0) {
            GlobalTracking(tmpSlice2, 0);
            WriteOutput(tmpSlice2, 0);
            mWriteOutputDone[tmpSlice2] = 1;
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

    WaitForHelperThreads();
  } else {
    mSliceSelectorReady = NSLICES;
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
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    Merger.DumpSliceTracks(mDebugFile);
  }
  timerUnpack.StopAndStart(timerMergeWithin);

  Merger.MergeWithingSlices();
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    Merger.DumpMergedWithinSlices(mDebugFile);
  }
  timerMergeWithin.StopAndStart(timerMergeSlices);

  Merger.MergeSlices();
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    Merger.DumpMergedBetweenSlices(mDebugFile);
  }
  timerMergeSlices.StopAndStart(timerMergeCE);

  Merger.MergeCEInit();
  timerMergeCE.StopAndStart(timerCollect);

  Merger.CollectMergedTracks();
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    Merger.DumpCollected(mDebugFile);
  }
  timerCollect.StopAndStart(timerMergeCE);

  Merger.MergeCE();
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    Merger.DumpMergeCE(mDebugFile);
  }
  timerMergeCE.StopAndStart(timerClusters);

  Merger.PrepareClustersForFit();
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    Merger.DumpFitPrepare(mDebugFile);
  }
  timerClusters.StopAndStart(timerCopyToGPU);

  if (doGPU) {
    SetupGPUProcessor(&Merger, false);
    MergerShadow.OverrideSliceTracker(processorsDevice()->tpcTrackers);
    MergerShadow.SetMatLUT(mFlatObjectsShadow.mCalibObjects.matLUT);
  }

  WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
  TransferMemoryResourceLinkToGPU(RecoStep::TPCMerging, Merger.MemoryResRefit());
  timerCopyToGPU.Stop();

  if (GetDeviceProcessingSettings().mergerSortTracks) {
    runKernel<GPUTPCGMMergerTrackFit>(GetGrid(Merger.NSlowTracks(), WarpSize(), 0), krnlRunRangeNone, krnlEventNone, -1);
    runKernel<GPUTPCGMMergerTrackFit>(GetGrid(Merger.NOutputTracks() - Merger.NSlowTracks(), FitThreadCount(), 0), krnlRunRangeNone, krnlEventNone, 1);
  } else {
    runKernel<GPUTPCGMMergerTrackFit>(GetGrid(Merger.NOutputTracks(), FitThreadCount(), 0), krnlRunRangeNone, krnlEventNone, 0);
  }
  SynchronizeGPU();

  timerCopyToHost.Start();
  TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResRefit());
  SynchronizeGPU();
  DoDebugAndDump(RecoStep::TPCMerging, 0, Merger, &GPUTPCGMMerger::DumpRefit, mDebugFile);
  timerCopyToHost.StopAndStart(timerFinalize);

  Merger.Finalize();
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    Merger.DumpFinal(mDebugFile);
  }
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
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step0attached>({BlockCount(), Compression1ThreadCount(), 0}, krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step1unattached>({BlockCount(), Compression2ThreadCount(), 0}, krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(myStep, &Compressor, 0);
  SynchronizeGPU();
  memset((void*)&Compressor.mOutput, 0, sizeof(Compressor.mOutput));
  Compressor.mOutput.nTracks = Compressor.mMemory->nStoredTracks;
  Compressor.mOutput.nAttachedClusters = Compressor.mMemory->nStoredAttachedClusters;
  Compressor.mOutput.nUnattachedClusters = Compressor.mMemory->nStoredUnattachedClusters;
  Compressor.mOutput.nAttachedClustersReduced = Compressor.mOutput.nAttachedClusters - Compressor.mOutput.nTracks;
  Compressor.mOutput.nSliceRows = NSLICES * GPUCA_ROW_COUNT;
  Compressor.mOutput.nComppressionModes = param().rec.tpcCompressionModes;
  AllocateRegisteredMemory(Compressor.mMemoryResOutputHost, &mRec->OutputControl());
  const GPUTPCCompression* outputSrc = nullptr;
  char direction = 0;
  if (DeviceProcessingSettings().tpcCompressionGatherMode == 0) {
    outputSrc = &CompressorShadow;
  } else if (DeviceProcessingSettings().tpcCompressionGatherMode == 1) {
    outputSrc = &Compressor;
    direction = -1;
  }
  GPUMemCpyAlways(myStep, Compressor.mOutput.nSliceRowClusters, outputSrc->mPtrs.nSliceRowClusters, NSLICES * GPUCA_ROW_COUNT * sizeof(Compressor.mOutput.nSliceRowClusters[0]), 0, direction);
  GPUMemCpyAlways(myStep, Compressor.mOutput.nTrackClusters, outputSrc->mPtrs.nTrackClusters, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.nTrackClusters[0]), 0, direction);
  SynchronizeGPU();
  unsigned int offset = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      GPUMemCpyAlways(myStep, Compressor.mOutput.qTotU + offset, outputSrc->mPtrs.qTotU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.qTotU[0]), 0, direction);
      GPUMemCpyAlways(myStep, Compressor.mOutput.qMaxU + offset, outputSrc->mPtrs.qMaxU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.qMaxU[0]), 0, direction);
      GPUMemCpyAlways(myStep, Compressor.mOutput.flagsU + offset, outputSrc->mPtrs.flagsU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.flagsU[0]), 0, direction);
      GPUMemCpyAlways(myStep, Compressor.mOutput.padDiffU + offset, outputSrc->mPtrs.padDiffU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.padDiffU[0]), 0, direction);
      GPUMemCpyAlways(myStep, Compressor.mOutput.timeDiffU + offset, outputSrc->mPtrs.timeDiffU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.timeDiffU[0]), 0, direction);
      GPUMemCpyAlways(myStep, Compressor.mOutput.sigmaPadU + offset, outputSrc->mPtrs.sigmaPadU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.sigmaPadU[0]), 0, direction);
      GPUMemCpyAlways(myStep, Compressor.mOutput.sigmaTimeU + offset, outputSrc->mPtrs.sigmaTimeU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.sigmaTimeU[0]), 0, direction);
      offset += Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
    }
  }
  offset = 0;
  for (unsigned int i = 0; i < Compressor.mOutput.nTracks; i++) {
    GPUMemCpyAlways(myStep, Compressor.mOutput.qTotA + offset, outputSrc->mPtrs.qTotA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.qTotA[0]), 0, direction);
    GPUMemCpyAlways(myStep, Compressor.mOutput.qMaxA + offset, outputSrc->mPtrs.qMaxA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.qMaxA[0]), 0, direction);
    GPUMemCpyAlways(myStep, Compressor.mOutput.flagsA + offset, outputSrc->mPtrs.flagsA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.flagsA[0]), 0, direction);
    GPUMemCpyAlways(myStep, Compressor.mOutput.sigmaPadA + offset, outputSrc->mPtrs.sigmaPadA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.sigmaPadA[0]), 0, direction);
    GPUMemCpyAlways(myStep, Compressor.mOutput.sigmaTimeA + offset, outputSrc->mPtrs.sigmaTimeA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.sigmaTimeA[0]), 0, direction);

    // First index stored with track
    GPUMemCpyAlways(myStep, Compressor.mOutput.rowDiffA + offset - i, outputSrc->mPtrs.rowDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.rowDiffA[0]), 0, direction);
    GPUMemCpyAlways(myStep, Compressor.mOutput.sliceLegDiffA + offset - i, outputSrc->mPtrs.sliceLegDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.sliceLegDiffA[0]), 0, direction);
    GPUMemCpyAlways(myStep, Compressor.mOutput.padResA + offset - i, outputSrc->mPtrs.padResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.padResA[0]), 0, direction);
    GPUMemCpyAlways(myStep, Compressor.mOutput.timeResA + offset - i, outputSrc->mPtrs.timeResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.timeResA[0]), 0, direction);
    offset += Compressor.mOutput.nTrackClusters[i];
  }
  GPUMemCpyAlways(myStep, Compressor.mOutput.qPtA, outputSrc->mPtrs.qPtA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.qPtA[0]), 0, direction);
  GPUMemCpyAlways(myStep, Compressor.mOutput.rowA, outputSrc->mPtrs.rowA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.rowA[0]), 0, direction);
  GPUMemCpyAlways(myStep, Compressor.mOutput.sliceA, outputSrc->mPtrs.sliceA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.sliceA[0]), 0, direction);
  GPUMemCpyAlways(myStep, Compressor.mOutput.timeA, outputSrc->mPtrs.timeA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.timeA[0]), 0, direction);
  GPUMemCpyAlways(myStep, Compressor.mOutput.padA, outputSrc->mPtrs.padA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.padA[0]), 0, direction);

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
  std::vector<GPUTRDTrackGPU> tracksTPC;
  std::vector<int> tracksTPCLab;
  GPUTRDTrackerGPU& Tracker = processors()->trdTracker;
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
  GPUTRDTrackerGPU& Tracker = processors()->trdTracker;
  GPUTRDTrackerGPU& TrackerShadow = doGPU ? processorsShadow()->trdTracker : Tracker;

  const auto& threadContext = GetThreadContext();
  SetupGPUProcessor(&Tracker, false);
  TrackerShadow.SetGeometry(reinterpret_cast<GPUTRDGeometry*>(mFlatObjectsDevice.mCalibObjects.trdGeometry));

  WriteToConstantMemory(RecoStep::TRDTracking, (char*)&processors()->trdTracker - (char*)processors(), &TrackerShadow, sizeof(TrackerShadow), 0);
  TransferMemoryResourcesToGPU(RecoStep::TRDTracking, &Tracker);

  runKernel<GPUTRDTrackerKernels>({BlockCount(), TRDThreadCount(), 0}, krnlRunRangeNone);
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
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    mDebugFile << "\n\nProcessing event " << nCount << std::endl;
  }
  if (mRec->slavesExist() && mRec->IsGPU()) {
    const auto threadContext = GetThreadContext();
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->calibObjects - (char*)processors(), &mFlatObjectsDevice.mCalibObjects, sizeof(mFlatObjectsDevice.mCalibObjects), -1); // Reinitialize
  }

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

  if (GetRecoSteps().isSet(RecoStep::TPCClusterFinding) && (mIOPtrs.tpcPackedDigits || mIOPtrs.tpcZS)) {
    timerClusterer.Start();
    if (param().rec.fwdTPCDigitsAsClusters) {
      ForwardTPCDigits();
    } else {
      if (RunTPCClusterizer()) {
        return 1;
      }
    }
    timerClusterer.Stop();
  }

  if (GetRecoSteps().isSet(RecoStep::TPCConversion) && mIOPtrs.clustersNative) {
    timerTransform.Start();
    if (ConvertNativeToClusterData()) {
      return 1;
    }
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
    if (RunTPCCompression()) {
      return 1;
    }
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
    if (mIOPtrs.clustersNative && GetRecoSteps() & RecoStep::TPCCompression) {
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
  if (param().rec.GlobalTracking) {
    int tmpSlice = GPUTPCTracker::GlobalTrackingSliceOrder(iSlice);
    int sliceLeft = (tmpSlice + (NSLICES / 2 - 1)) % (NSLICES / 2);
    int sliceRight = (tmpSlice + 1) % (NSLICES / 2);
    if (tmpSlice >= (int)NSLICES / 2) {
      sliceLeft += NSLICES / 2;
      sliceRight += NSLICES / 2;
    }

    while (mSliceSelectorReady < tmpSlice || mSliceSelectorReady < sliceLeft || mSliceSelectorReady < sliceRight) {
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
