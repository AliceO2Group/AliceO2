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
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "GPUHostDataTypes.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCdEdxCalibrationSplines.h"
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
  if (((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) || !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCConversion)) && param().rec.mergerReadFromTrackerDirectly) {
    GPUError("Invalid input / output / step, mergerReadFromTrackerDirectly cannot read/store sectors tracks and needs TPC conversion");
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
  if (mRec->IsGPU() && GetDeviceProcessingSettings().nTPCClustererLanes >= mRec->NStreams()) {
    GPUError("NStreams must be > nTPCClustererLanes");
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
    return 1;
  }
  if (!ValidateSettings()) {
    return 1;
  }

  if (GPUQA::QAAvailable() && (GetDeviceProcessingSettings().runQA || GetDeviceProcessingSettings().eventDisplay)) {
    mQA.reset(new GPUQA(this));
  }
  if (GetDeviceProcessingSettings().eventDisplay) {
    mEventDisplay.reset(new GPUDisplay(GetDeviceProcessingSettings().eventDisplay, this, mQA.get()));
  }

  if (mOutputCompressedClusters == nullptr) {
    mOutputCompressedClusters = &mRec->OutputControl();
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

unsigned int GPUChainTracking::TPCClusterizerDecodeZSCount(unsigned int iSlice, unsigned int minTime, unsigned int maxTime)
{
  unsigned int nDigits = 0;
#ifdef HAVE_O2HEADERS
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;

  processors()->tpcClusterer[iSlice].mPmemory->counters.nPagesSubslice = 0;
  GPUTPCClusterFinder::ZSOffset* o = processors()->tpcClusterer[iSlice].mPzsOffsets;
  int firstHBF = mIOPtrs.tpcZS->slice[iSlice].count[0] ? o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0]) : 0;
  for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    if (!doGPU) {
      processors()->tpcClusterer[iSlice].mPzsOffsets[j] = GPUTPCClusterFinder::ZSOffset{nDigits, j, 0};
    }
    processors()->tpcClusterer[iSlice].mMinMaxCN[j].maxC = mIOPtrs.tpcZS->slice[iSlice].count[j];
    processors()->tpcClusterer[iSlice].mMinMaxCN[j].minC = processors()->tpcClusterer[iSlice].mMinMaxCN[j].maxC;
    processors()->tpcClusterer[iSlice].mMinMaxCN[j].maxN = mIOPtrs.tpcZS->slice[iSlice].count[j] ? mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][mIOPtrs.tpcZS->slice[iSlice].count[j] - 1] : 0;
    processors()->tpcClusterer[iSlice].mMinMaxCN[j].minN = processors()->tpcClusterer[iSlice].mMinMaxCN[j].minN;

    bool firstFound = false;
    unsigned short num = 0;
    for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
      bool breakEndpoint = false;
      for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]; l++) {
        const unsigned char* const page = ((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)page;
        if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
          if (firstFound) {
            processors()->tpcClusterer[iSlice].mPmemory->counters.nPagesSubslice++;
          }
          continue;
        }
        const TPCZSHDR* const hdr = (const TPCZSHDR*)(page + sizeof(o2::header::RAWDataHeader));
        unsigned int timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / Constants::LHCBCPERTIMEBIN;
        if (timeBin + hdr->nTimeBins < minTime) {
          continue;
        }
        if (timeBin > maxTime) {
          processors()->tpcClusterer[iSlice].mMinMaxCN[j].maxC = k + 1;
          processors()->tpcClusterer[iSlice].mMinMaxCN[j].maxN = l;
          breakEndpoint = true;
          break;
        }
        if (!firstFound) {
          processors()->tpcClusterer[iSlice].mMinMaxCN[j].minC = k;
          processors()->tpcClusterer[iSlice].mMinMaxCN[j].minN = l;
          firstFound = true;
        }
        if (timeBin + hdr->nTimeBins > mTPCMaxTimeBin) {
          mTPCMaxTimeBin = timeBin + hdr->nTimeBins;
        }

        if (doGPU) {
          *(o++) = GPUTPCClusterFinder::ZSOffset{nDigits, j, num++};
        }
        processors()->tpcClusterer[iSlice].mPmemory->counters.nPagesSubslice++;
        nDigits += hdr->nADCsamples;
      }
      if (breakEndpoint) {
        break;
      }
    }
  }
#endif
  return nDigits;
}

int GPUChainTracking::PrepareEvent()
{
  mRec->MemoryScalers()->nTRDTracklets = mIOPtrs.nTRDTracklets;
  if (mIOPtrs.tpcPackedDigits || mIOPtrs.tpcZS) {
#ifdef GPUCA_TPC_GEOMETRY_O2
    if (mIOPtrs.tpcZS && param().rec.fwdTPCDigitsAsClusters) {
      throw std::runtime_error("Forwading zero-suppressed hits not supported");
    }
    mRec->MemoryScalers()->nTPCdigits = 0;
    size_t nPagesTotal = 0;
    mTPCMaxTimeBin = mIOPtrs.tpcZS ? 0 : std::max<int>(param().continuousMaxTimeBin, TPC_MAX_FRAGMENT_LEN);
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      unsigned int nDigits = 0;
      unsigned int nPages = 0;
      if (mIOPtrs.tpcZS) {
        for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
          for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
            nPages += mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
          }
        }
        processors()->tpcClusterer[iSlice].mPmemory->counters.nPages = nPages;
        processors()->tpcClusterer[iSlice].SetNMaxDigits(0, nPages);
        if (mRec->IsGPU()) {
          processorsShadow()->tpcClusterer[iSlice].SetNMaxDigits(0, nPages);
        }
        AllocateRegisteredMemory(processors()->tpcClusterer[iSlice].mZSOffsetId);
        nPagesTotal += nPages;
        nDigits = TPCClusterizerDecodeZSCount(iSlice, 0, -1);
        processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits = nDigits;
      } else {
        nDigits = mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice];
      }
      mRec->MemoryScalers()->nTPCdigits += nDigits;
      unsigned int maxClusters = param().rec.fwdTPCDigitsAsClusters ? nDigits : mRec->MemoryScalers()->NTPCClusters(nDigits);
      processors()->tpcTrackers[iSlice].Data().SetClusterData(nullptr, maxClusters, 0); // TODO: fixme
      // Distribute maximum digits, so that we can reuse the memory easily
      processors()->tpcClusterer[iSlice].SetNMaxDigits(nDigits, nPages);
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
  std::memset((void*)mClusterNativeAccess.get(), 0, sizeof(*mClusterNativeAccess));
}

void GPUChainTracking::AllocateIOMemory()
{
  for (unsigned int i = 0; i < NSLICES; i++) {
    AllocateIOMemoryHelper(mIOPtrs.nClusterData[i], mIOPtrs.clusterData[i], mIOMem.clusterData[i]);
    AllocateIOMemoryHelper(mIOPtrs.nRawClusters[i], mIOPtrs.rawClusters[i], mIOMem.rawClusters[i]);
    AllocateIOMemoryHelper(mIOPtrs.nSliceTracks[i], mIOPtrs.sliceTracks[i], mIOMem.sliceTracks[i]);
    AllocateIOMemoryHelper(mIOPtrs.nSliceClusters[i], mIOPtrs.sliceClusters[i], mIOMem.sliceClusters[i]);
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
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCConversion;
  GPUTPCConvert& convert = processors()->tpcConverter;
  GPUTPCConvert& convertShadow = doGPU ? processorsShadow()->tpcConverter : convert;

  if (mIOPtrs.clustersNative->nClustersTotal > convert.mNClustersTotal) {
    GPUError("Too many input clusters in conversion (expected <= %ld)\n", convert.mNClustersTotal);
    for (unsigned int i = 0; i < NSLICES; i++) {
      mIOPtrs.nClusterData[i] = 0;
      processors()->tpcTrackers[i].Data().SetClusterData(nullptr, 0, 0);
    }
    return 1;
  }
  SetupGPUProcessor(&convert, false);
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
    if (GetDeviceProcessingSettings().debugLevel >= 3) {
      GPUInfo("Early transform inactive, skipping TPC Early transformation kernel, transformed on the fly during slice data creation / refit");
    }
    for (unsigned int i = 0; i < NSLICES; i++) {
      processors()->tpcTrackers[i].Data().SetClusterData(nullptr, mIOPtrs.clustersNative->nClustersSector[i], mIOPtrs.clustersNative->clusterOffset[i][0]);
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
  SynchronizeGPU();

  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.nClusterData[i] = (i == NSLICES - 1 ? mIOPtrs.clustersNative->nClustersTotal : mIOPtrs.clustersNative->clusterOffset[i + 1][0]) - mIOPtrs.clustersNative->clusterOffset[i][0];
    mIOPtrs.clusterData[i] = convert.mClusters + mIOPtrs.clustersNative->clusterOffset[i][0];
    processors()->tpcTrackers[i].Data().SetClusterData(mIOPtrs.clusterData[i], mIOPtrs.nClusterData[i], mIOPtrs.clustersNative->clusterOffset[i][0]);
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
  GPUReconstructionConvert::RunZSEncoder<o2::tpc::Digit>(*mIOPtrs.tpcPackedDigits, &mTPCZSBuffer, mTPCZSSizes.get(), nullptr, nullptr, param(), zs12bit, true);
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
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("Running ReadEvent for slice %d on thread %d\n", iSlice, threadId);
  }
  runKernel<GPUTPCCreateSliceData>({GetGridBlk(1, 0, GPUReconstruction::krnlDeviceType::CPU)}, {iSlice});
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
  PrepareEventFromNative();
#endif
}

int GPUChainTracking::GlobalTracking(unsigned int iSlice, int threadId, bool synchronizeOutput)
{
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("GPU Tracker running Global Tracking for slice %u on thread %d\n", iSlice, threadId);
  }

  GPUReconstruction::krnlDeviceType deviceType = GetDeviceProcessingSettings().fullMergerOnGPU ? GPUReconstruction::krnlDeviceType::Auto : GPUReconstruction::krnlDeviceType::CPU;
  runKernel<GPUTPCGlobalTracking>(GetGridBlk(256, iSlice % mRec->NStreams(), deviceType), {iSlice});
  if (GetDeviceProcessingSettings().fullMergerOnGPU) {
    TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, processors()->tpcTrackers[iSlice].MemoryResCommon(), iSlice % mRec->NStreams());
  }
  if (synchronizeOutput) {
    SynchronizeStream(iSlice % mRec->NStreams());
  }

  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("GPU Tracker finished Global Tracking for slice %u on thread %d\n", iSlice, threadId);
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
    auto& nIn = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;
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

int GPUChainTracking::RunTPCClusterizer(bool synchronizeOutput)
{
#ifdef GPUCA_TPC_GEOMETRY_O2
  const auto& threadContext = GetThreadContext();
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;

  if (doGPU) {
    if (mIOPtrs.tpcZS) {
      processorsShadow()->ioPtrs.tpcZS = mInputsShadow->mPzsMeta;
      WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    }
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)processors()->tpcClusterer - (char*)processors(), processorsShadow()->tpcClusterer, sizeof(GPUTPCClusterFinder) * NSLICES, mRec->NStreams() - 1, &mEvents->init);
  }
  SynchronizeGPU();

  size_t nClsTotal = 0;
  ClusterNativeAccess* tmp = mClusterNativeAccess.get();
  std::fill(&tmp->nClusters[0][0], &tmp->nClusters[tpc::Constants::MAXSECTOR][tpc::Constants::MAXGLOBALPADROW], 0);

  // setup MC Labels
  bool propagateMCLabels = !doGPU && GetDeviceProcessingSettings().runMC && processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC != nullptr;

  auto* digitsMC = propagateMCLabels ? processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC : nullptr;

  GPUTPCLinearLabels mcLinearLabels;
  if (propagateMCLabels) {
    mcLinearLabels.header.reserve(mRec->MemoryScalers()->nTPCHits);
    mcLinearLabels.data.reserve(mRec->MemoryScalers()->nTPCHits * 16); // Assumption: cluster will have less than 16 labels on average
  }

  if (param().continuousMaxTimeBin > 0 && mTPCMaxTimeBin >= (unsigned int)std::max(param().continuousMaxTimeBin + 1, TPC_MAX_FRAGMENT_LEN)) {
    GPUError("Input data has invalid time bin\n");
    return 1;
  }
  tpccf::TPCTime timeSliceLen = std::max<int>(mTPCMaxTimeBin + 1, TPC_MAX_FRAGMENT_LEN);
  tpccf::TPCFragmentTime fragmentLen = TPC_MAX_FRAGMENT_LEN;
  bool buildNativeGPU = false, buildNativeHost = false;
  mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mRec->MemoryScalers()->nTPCHits;
  if ((mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression)) {
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
    buildNativeGPU = true;
  }
  if (mRec->GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCClusters) { // TODO: Should do this also when clusters are needed for later steps on the host but not requested as output
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput);
    buildNativeHost = true;
  }

  char transferRunning[NSLICES] = {0};
  for (unsigned int iSliceBase = 0; iSliceBase < NSLICES; iSliceBase += GetDeviceProcessingSettings().nTPCClustererLanes) {
    std::vector<bool> laneHasData(GetDeviceProcessingSettings().nTPCClustererLanes, false);
    for (CfFragment fragment{timeSliceLen, fragmentLen}; !fragment.isEnd(); fragment = fragment.next(timeSliceLen, fragmentLen)) {
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Processing time bins [%d, %d)", fragment.start, fragment.last());
      }
      for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        SetupGPUProcessor(&clusterer, false);
        clusterer.mPmemory->counters.nPeaks = clusterer.mPmemory->counters.nClusters = 0;
        clusterer.mPmemory->fragment = fragment;

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
        }
        if (not mIOPtrs.tpcZS || propagateMCLabels) {
          auto* inDigits = mIOPtrs.tpcPackedDigits;
          size_t numDigits = inDigits->nTPCDigits[iSlice];
          clusterer.mPmemory->counters.nDigits = numDigits;
          if (doGPU) {
            GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPdigits, inDigits->tpcDigits[iSlice], sizeof(clustererShadow.mPdigits[0]) * numDigits, lane, true);
          } else {
            clustererShadow.mPdigits = const_cast<o2::tpc::Digit*>(inDigits->tpcDigits[iSlice]); // TODO: Needs fixing (invalid const cast)
          }
        }
        TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);

        if (mIOPtrs.tpcZS && nPagesTotal) {
          clusterer.mPmemory->counters.nPositions = TPCClusterizerDecodeZSCount(iSlice, fragment.start, fragment.last() - 1);
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
          if (doGPU) {
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
                  size_t size = o2::raw::RDHUtils::getMemorySize(*(const o2::header::RAWDataHeader*)src);
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
            TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceZS, lane);
            TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mZSOffsetId, lane);
          }
        }

        // These buffers only have to be cleared once entirely. The 'resetMaps' kernel
        // takes care of subsequent clean ups.
        if (iSliceBase == 0) {
          using ChargeMapType = decltype(*clustererShadow.mPchargeMap);
          using PeakMapType = decltype(*clustererShadow.mPpeakMap);
          runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPchargeMap, TPCMapMemoryLayout<ChargeMapType>::items() * sizeof(ChargeMapType));
          runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPpeakMap, TPCMapMemoryLayout<PeakMapType>::items() * sizeof(PeakMapType));
        }
        if (fragment.index == 0) {
          runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPclusterInRow, GPUCA_ROW_COUNT * sizeof(*clustererShadow.mPclusterInRow));
        }
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, mDebugFile, "Zeroed Charges");
        if (mIOPtrs.tpcZS && nPagesTotal == 0) {
          continue;
        }

        if (propagateMCLabels || not mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane), {iSlice}, {});
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }

        if (mIOPtrs.tpcZS) {
          int firstHBF = mIOPtrs.tpcZS->slice[iSlice].count[0] ? o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0]) : 0;
          runKernel<GPUTPCCFDecodeZS, GPUTPCCFDecodeZS::decodeZS>(GetGridBlk(doGPU ? clusterer.mPmemory->counters.nPagesSubslice : GPUTrackingInOutZS::NENDPOINTS, lane), {iSlice}, {}, firstHBF);
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }
      }
      for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        SynchronizeStream(lane);
        if (propagateMCLabels || not mIOPtrs.tpcZS) {
          if (clusterer.mPmemory->counters.nPositions == 0) {
            continue;
          }
        }
        if (!mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillFromDigits>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        }
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpDigits, mDebugFile)) {
          clusterer.DumpChargeMap(mDebugFile, "Charges");
        }

        if (propagateMCLabels) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillIndexMap>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        }

        runKernel<GPUTPCCFPeakFinder, GPUTPCCFPeakFinder::findPeaks>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
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
        runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::noiseSuppression>(GetGrid(clusterer.mPmemory->counters.nPeaks, lane), {iSlice}, {});
        runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::updatePeaks>(GetGrid(clusterer.mPmemory->counters.nPeaks, lane), {iSlice}, {});
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaks, mDebugFile);

        RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 1, doGPU, lane);
        TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaksCompacted, mDebugFile);
      }
      for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        SynchronizeStream(lane);
        if (clusterer.mPmemory->counters.nClusters == 0) {
          continue;
        }

        runKernel<GPUTPCCFDeconvolution, GPUTPCCFDeconvolution::countPeaks>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, mDebugFile, "Split Charges");

        runKernel<GPUTPCCFClusterizer, GPUTPCCFClusterizer::computeClusters>(GetGrid(clusterer.mPmemory->counters.nClusters, lane), {iSlice}, {nullptr, transferRunning[lane] == 1 ? &mEvents->slice[lane] : nullptr});
        transferRunning[lane] = 2;
        if (GetDeviceProcessingSettings().debugLevel >= 3) {
          GPUInfo("Lane %d: Found clusters: digits %u peaks %u clusters %u", lane, (int)clusterer.mPmemory->counters.nPositions, (int)clusterer.mPmemory->counters.nPeaks, (int)clusterer.mPmemory->counters.nClusters);
        }
        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, lane);
        laneHasData[lane] = true;
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpCountedPeaks, mDebugFile);
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpClusters, mDebugFile);
      }
      for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        if (clusterer.mPmemory->counters.nPositions) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::resetMaps>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        }
      }
    }
    size_t nClsFirst = nClsTotal;
    bool anyLaneHasData = false;
    for (int lane = 0; lane < GetDeviceProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      SynchronizeStream(lane);
      unsigned int iSlice = iSliceBase + lane;
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
      if (laneHasData[lane]) {
        anyLaneHasData = true;
        for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
          if (buildNativeGPU) {
            GPUMemCpyAlways(RecoStep::TPCClusterFinding, (void*)&mInputsShadow->mPclusterNativeBuffer[nClsTotal], (const void*)&clustererShadow.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * clusterer.mPclusterInRow[j], mRec->NStreams() - 1, -2);
          } else if (buildNativeHost) {
            GPUMemCpyAlways(RecoStep::TPCClusterFinding, (void*)&mInputsHost->mPclusterNativeOutput[nClsTotal], (const void*)&clustererShadow.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * clusterer.mPclusterInRow[j], mRec->NStreams() - 1, false);
          }
          tmp->nClusters[iSlice][j] += clusterer.mPclusterInRow[j];
          nClsTotal += clusterer.mPclusterInRow[j];
        }
        if (transferRunning[lane]) {
          ReleaseEvent(&mEvents->slice[lane]);
        }
        RecordMarker(&mEvents->slice[lane], mRec->NStreams() - 1);
        transferRunning[lane] = 1;
      }

      if (not propagateMCLabels) {
        continue;
      }

      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::setRowOffsets>(GetGrid(GPUCA_ROW_COUNT, lane), {iSlice}, {});
      GPUTPCCFMCLabelFlattener::setGlobalOffsetsAndAllocate(clusterer, mcLinearLabels);
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        if (clusterer.mPclusterInRow[j] == 0) {
          continue;
        }
        runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::flatten>(GetGrid(clusterer.mPclusterInRow[j], lane), {iSlice}, {}, j, &mcLinearLabels);
      }
    }
    if (buildNativeHost && buildNativeGPU && anyLaneHasData) {
      GPUMemCpy(RecoStep::TPCClusterFinding, (void*)&mInputsHost->mPclusterNativeOutput[nClsFirst], (void*)&mInputsShadow->mPclusterNativeBuffer[nClsFirst], (nClsTotal - nClsFirst) * sizeof(mInputsHost->mPclusterNativeOutput[nClsFirst]), mRec->NStreams() - 1, false);
    }
  }
  for (int i = 0; i < GetDeviceProcessingSettings().nTPCClustererLanes; i++) {
    if (transferRunning[i]) {
      ReleaseEvent(&mEvents->slice[i]);
    }
  }

  static o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcLabels;

  assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);

  mcLabels.setFrom(mcLinearLabels.header, mcLinearLabels.data);

  if (buildNativeHost) {
    tmp->clustersLinear = mInputsHost->mPclusterNativeOutput;
    tmp->clustersMCTruth = propagateMCLabels ? &mcLabels : nullptr;
    tmp->setOffsetPtrs();
    mIOPtrs.clustersNative = tmp;
  }
  if (buildNativeGPU) {
    processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
    mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
    mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
    TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceClusterNativeAccess, 0);
  }
  PrepareEventFromNative();
  if (synchronizeOutput) {
    SynchronizeStream(mRec->NStreams() - 1);
  }
  if (buildNativeHost && GetDeviceProcessingSettings().debugLevel >= 4) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        std::sort(&mInputsHost->mPclusterNativeOutput[tmp->clusterOffset[i][j]], &mInputsHost->mPclusterNativeOutput[tmp->clusterOffset[i][j] + tmp->nClusters[i][j]]);
      }
    }
  }

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
  bool doSliceDataOnGPU = processors()->tpcTrackers[0].SliceDataOnGPU();

  bool streamInit[GPUCA_MAX_STREAMS] = {false};
  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    processors()->tpcTrackers[iSlice].SetupCommonMemory();
  }
  if (doGPU) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processorsShadow()->tpcTrackers[iSlice].GPUParametersConst()->gpumem = (char*)mRec->DeviceMemoryBase();
      // Initialize Startup Constants
      processors()->tpcTrackers[iSlice].GPUParameters()->nextStartHit = (((getKernelProperties<GPUTPCTrackletConstructor, GPUTPCTrackletConstructor::allSlices>().minBlocks * BlockCount()) + NSLICES - 1 - iSlice) / NSLICES) * getKernelProperties<GPUTPCTrackletConstructor, GPUTPCTrackletConstructor::allSlices>().nThreads;
      processorsShadow()->tpcTrackers[iSlice].SetGPUTextureBase(mRec->DeviceMemoryBase());
#ifdef HAVE_O2HEADERS
      if (GetRecoSteps().isSet(RecoStep::TPCConversion)) {
        processorsShadow()->tpcTrackers[iSlice].Data().SetClusterData(processorsShadow()->tpcConverter.mClusters + processors()->tpcTrackers[iSlice].Data().ClusterIdOffset(), processors()->tpcTrackers[iSlice].NHitsTotal(), processors()->tpcTrackers[iSlice].Data().ClusterIdOffset());
      }
#endif
    }

    if (!doSliceDataOnGPU) {
      RunHelperThreads(&GPUChainTracking::HelperReadEvent, this, NSLICES);
    }
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
    int useStream = (iSlice % mRec->NStreams());

    if (GetDeviceProcessingSettings().debugLevel >= 3) {
      GPUInfo("Creating Slice Data (Slice %d)", iSlice);
    }
    if (doSliceDataOnGPU) {
      TransferMemoryResourcesToGPU(RecoStep::TPCSliceTracking, &trk, useStream);
      runKernel<GPUTPCCreateSliceData>(GetGridBlk(GPUCA_ROW_COUNT, useStream), {iSlice}, {nullptr, streamInit[useStream] ? nullptr : &mEvents->init});
      streamInit[useStream] = true;
    } else if (!doGPU || iSlice % (GetDeviceProcessingSettings().nDeviceHelperThreads + 1) == 0) {
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

    // Initialize temporary memory where needed
    if (GetDeviceProcessingSettings().debugLevel >= 3) {
      GPUInfo("Copying Slice Data to GPU and initializing temporary memory");
    }
    if (GetDeviceProcessingSettings().keepAllMemory) {
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

    if (GetDeviceProcessingSettings().keepDisplayMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &trk, -1, true);
      memcpy(trk.LinkTmpMemory(), mRec->Res(trk.MemoryResLinks()).Ptr(), mRec->Res(trk.MemoryResLinks()).Size());
      if (GetDeviceProcessingSettings().debugMask & 2) {
        trk.DumpLinks(mDebugFile);
      }
    }

    runKernel<GPUTPCNeighboursCleaner>(GetGridBlk(GPUCA_ROW_COUNT - 2, useStream), {iSlice});
    DoDebugAndDump(RecoStep::TPCSliceTracking, 4, trk, &GPUTPCTracker::DumpLinks, mDebugFile);

    runKernel<GPUTPCStartHitsFinder>(GetGridBlk(GPUCA_ROW_COUNT - 6, useStream), {iSlice});
#ifdef GPUCA_SORT_STARTHITS_GPU
    if (doGPU) {
      runKernel<GPUTPCStartHitsSorter>(GetGridAuto(useStream), {iSlice});
    }
#endif
    DoDebugAndDump(RecoStep::TPCSliceTracking, 32, trk, &GPUTPCTracker::DumpStartHits, mDebugFile);

    if (GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
      trk.UpdateMaxData();
      AllocateRegisteredMemory(trk.MemoryResTracklets());
      AllocateRegisteredMemory(trk.MemoryResOutput());
    }

    if (!(doGPU || GetDeviceProcessingSettings().debugLevel >= 1) || GetDeviceProcessingSettings().trackletConstructorInPipeline) {
      runKernel<GPUTPCTrackletConstructor>(GetGridAuto(useStream), {iSlice});
      DoDebugAndDump(RecoStep::TPCSliceTracking, 128, trk, &GPUTPCTracker::DumpTrackletHits, mDebugFile);
      if (GetDeviceProcessingSettings().debugMask & 256 && !GetDeviceProcessingSettings().comparableDebutOutput) {
        trk.DumpHitWeights(mDebugFile);
      }
    }

    if (!(doGPU || GetDeviceProcessingSettings().debugLevel >= 1) || GetDeviceProcessingSettings().trackletSelectorInPipeline) {
      runKernel<GPUTPCTrackletSelector>(GetGridAuto(useStream), {iSlice});
      runKernel<GPUTPCGlobalTrackingCopyNumbers>({1, -ThreadCount(), useStream}, {iSlice}, {}, 1);
      TransferMemoryResourceLinkToHost(RecoStep::TPCSliceTracking, trk.MemoryResCommon(), useStream, &mEvents->slice[iSlice]);
      streamMap[iSlice] = useStream;
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Slice %u, Number of tracks: %d", iSlice, *trk.NTracks());
      }
      DoDebugAndDump(RecoStep::TPCSliceTracking, 512, trk, &GPUTPCTracker::DumpTrackHits, mDebugFile);
    }
  }
  if (error) {
    return (3);
  }

  if (doGPU || GetDeviceProcessingSettings().debugLevel >= 1) {
    ReleaseEvent(&mEvents->init);
    if (!doSliceDataOnGPU) {
      WaitForHelperThreads();
    }

    if (!GetDeviceProcessingSettings().trackletSelectorInPipeline) {
      if (GetDeviceProcessingSettings().trackletConstructorInPipeline) {
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

      if (GetDeviceProcessingSettings().debugLevel >= 4) {
        for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
          DoDebugAndDump(RecoStep::TPCSliceTracking, 128, processors()->tpcTrackers[iSlice], &GPUTPCTracker::DumpTrackletHits, mDebugFile);
        }
      }

      int runSlices = 0;
      int useStream = 0;
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice += runSlices) {
        if (runSlices < GetDeviceProcessingSettings().trackletSelectorSlices) {
          runSlices++;
        }
        runSlices = CAMath::Min<int>(runSlices, NSLICES - iSlice);
        if (getKernelProperties<GPUTPCTrackletSelector>().minBlocks * BlockCount() < (unsigned int)runSlices) {
          runSlices = getKernelProperties<GPUTPCTrackletSelector>().minBlocks * BlockCount();
        }

        if (GetDeviceProcessingSettings().debugLevel >= 3) {
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
        if (GetDeviceProcessingSettings().debugLevel >= 3) {
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
          SynchronizeEvents(&mEvents->slice[iSlice]);
        }
        if (GetDeviceProcessingSettings().debugLevel >= 3) {
          GPUInfo("Tracks Transfered: %d / %d", *processors()->tpcTrackers[iSlice].NTracks(), *processors()->tpcTrackers[iSlice].NTrackHits());
        }

        if (GetDeviceProcessingSettings().debugLevel >= 3) {
          GPUInfo("Data ready for slice %d, helper thread %d", iSlice, iSlice % (GetDeviceProcessingSettings().nDeviceHelperThreads + 1));
        }
        mSliceSelectorReady = iSlice;

        if (param().rec.GlobalTracking) {
          for (unsigned int tmpSlice2a = 0; tmpSlice2a <= iSlice; tmpSlice2a += GetDeviceProcessingSettings().nDeviceHelperThreads + 1) {
            unsigned int tmpSlice2 = GPUTPCGlobalTracking::GlobalTrackingSliceOrder(tmpSlice2a);
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
      WaitForHelperThreads();
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (transferRunning[iSlice]) {
        ReleaseEvent(&mEvents->slice[iSlice]);
      }
      if (!(GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) && param().rec.GlobalTracking) {
        GlobalTracking(iSlice, 0, !GetDeviceProcessingSettings().fullMergerOnGPU);
      }
    }
  } else {
    mSliceSelectorReady = NSLICES;
#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(doGPU ? 1 : GetDeviceProcessingSettings().nThreads)
#endif
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (param().rec.GlobalTracking) {
        GlobalTracking(iSlice, 0);
      }
      if (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) {
        WriteOutput(iSlice, 0);
      }
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

  if (param().rec.GlobalTracking && GetDeviceProcessingSettings().debugLevel >= 3) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      GPUInfo("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d", iSlice,
              processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTracks, processors()->tpcTrackers[iSlice].CommonMemory()->nTracks, processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTrackHits, processors()->tpcTrackers[iSlice].CommonMemory()->nTrackHits);
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
    mIOPtrs.nSliceTracks[i] = *processors()->tpcTrackers[i].NTracks();
    mIOPtrs.sliceTracks[i] = processors()->tpcTrackers[i].Tracks();
    mIOPtrs.nSliceClusters[i] = *processors()->tpcTrackers[i].NTrackHits();
    mIOPtrs.sliceClusters[i] = processors()->tpcTrackers[i].TrackHits();
    if (GetDeviceProcessingSettings().keepDisplayMemory && !GetDeviceProcessingSettings().keepAllMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSliceTracking, &processors()->tpcTrackers[i], -1, true);
    }
  }
  SynchronizeGPU(); // Todo: why needed? processing small data fails otherwise
  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Slice Tracker finished");
  }
  return 0;
}

void GPUChainTracking::RunTPCTrackingMerger_MergeBorderTracks(char withinSlice, char mergeMode, GPUReconstruction::krnlDeviceType deviceType)
{
  unsigned int n = withinSlice == -1 ? NSLICES / 2 : NSLICES;
  if (GetDeviceProcessingSettings().alternateBorderSort) {
    bool doGPUall = GetRecoStepsGPU() & RecoStep::TPCMerging && GetDeviceProcessingSettings().fullMergerOnGPU;
    GPUTPCGMMerger& Merger = processors()->tpcMerger;
    GPUTPCGMMerger& MergerShadow = doGPUall ? processorsShadow()->tpcMerger : Merger;
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, nullptr, &mEvents->init);
    RecordMarker(&mEvents->single, 0);
    for (unsigned int i = 0; i < n; i++) {
      int stream = i % mRec->NStreams();
      runKernel<GPUTPCGMMergerMergeBorders, 0>(GetGridAuto(stream, deviceType), krnlRunRangeNone, {nullptr, stream ? &mEvents->single : nullptr}, i, withinSlice, mergeMode);
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
      GPUTPCGMBorderTrack::Range* range1 = MergerShadow.BorderRange(i);
      GPUTPCGMBorderTrack::Range* range2 = MergerShadow.BorderRange(jSlice) + *processors()->tpcTrackers[jSlice].NTracks();
      runKernel<GPUTPCGMMergerMergeBorders, 3>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, range1, n1, 0);
      runKernel<GPUTPCGMMergerMergeBorders, 3>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, range2, n2, 1);
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
}

int GPUChainTracking::RunTPCTrackingMerger(bool synchronizeOutput)
{
  if (GetDeviceProcessingSettings().debugLevel >= 6 && GetDeviceProcessingSettings().comparableDebutOutput && param().rec.mergerReadFromTrackerDirectly) {
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
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCMerging;
  bool doGPUall = doGPU && GetDeviceProcessingSettings().fullMergerOnGPU;
  GPUReconstruction::krnlDeviceType deviceType = GetDeviceProcessingSettings().fullMergerOnGPU ? GPUReconstruction::krnlDeviceType::Auto : GPUReconstruction::krnlDeviceType::CPU;
  unsigned int numBlocks = doGPUall ? BlockCount() : 1;

  GPUTPCGMMerger& Merger = processors()->tpcMerger;
  GPUTPCGMMerger& MergerShadow = doGPU ? processorsShadow()->tpcMerger : Merger;
  GPUTPCGMMerger& MergerShadowAll = doGPUall ? processorsShadow()->tpcMerger : Merger;
  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Merger");
  }
  const auto& threadContext = GetThreadContext();

  SetupGPUProcessor(&Merger, true);

  if (Merger.CheckSlices()) {
    return 1;
  }

  memset(Merger.Memory(), 0, sizeof(*Merger.Memory()));
  WriteToConstantMemory(RecoStep::TPCMerging, (char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
  if (doGPUall) {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger, 0);
  }

  for (unsigned int i = 0; i < NSLICES; i++) {
    runKernel<GPUTPCGMMergerUnpackResetIds>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, i);
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, i);
    runKernel<GPUTPCGMMergerSliceRefit>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, i);
  }
  for (unsigned int i = 0; i < NSLICES; i++) {
    runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, NSLICES + i);
    runKernel<GPUTPCGMMergerUnpackGlobal>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, i);
  }
  runKernel<GPUTPCGMMergerUnpackSaveNumber>({1, -WarpSize(), 0, deviceType}, krnlRunRangeNone, krnlEventNone, 2 * NSLICES);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpSliceTracks, mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, 0);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeWithinPrepare>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);

  RunTPCTrackingMerger_MergeBorderTracks(1, 0, deviceType);
  runKernel<GPUTPCGMMergerResolve>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 1);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergedWithinSlices, mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, 0);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 2, 3, 0);
  RunTPCTrackingMerger_MergeBorderTracks(0, 0, deviceType);
  runKernel<GPUTPCGMMergerResolve>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 0);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 1, 0);
  RunTPCTrackingMerger_MergeBorderTracks(0, 0, deviceType);
  runKernel<GPUTPCGMMergerResolve>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 0);
  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));
  runKernel<GPUTPCGMMergerMergeSlicesPrepare>(GetGridBlk(std::max(2u, numBlocks), 0, deviceType), krnlRunRangeNone, krnlEventNone, 0, 1, 1);
  RunTPCTrackingMerger_MergeBorderTracks(0, -1, deviceType);
  runKernel<GPUTPCGMMergerResolve>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, 1, 0);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergedBetweenSlices, mDebugFile);

  runKernel<GPUMemClean16>({1, -WarpSize(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.TmpCounter(), 2 * NSLICES * sizeof(*MergerShadowAll.TmpCounter()));

  runKernel<GPUTPCGMMergerLinkGlobalTracks>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCGMMergerCollect>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpCollected, mDebugFile);

  runKernel<GPUTPCGMMergerClearLinks>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone, 1);
  RunTPCTrackingMerger_MergeBorderTracks(-1, 1, deviceType);
  RunTPCTrackingMerger_MergeBorderTracks(-1, 2, deviceType);
  runKernel<GPUTPCGMMergerMergeCE>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpMergeCE, mDebugFile);
  int waitForTransfer = 0;
  if (doGPUall) {
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResMemory(), 0, nullptr, &mEvents->single);
    waitForTransfer = 1;
  }

  if (GetDeviceProcessingSettings().mergerSortTracks) {
    runKernel<GPUTPCGMMergerSortTracksPrepare>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
    CondWaitEvent(waitForTransfer, &mEvents->single);
    runKernel<GPUTPCGMMergerSortTracks>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
  }

  unsigned int maxId = param().rec.NonConsecutiveIDs ? Merger.Memory()->nOutputTrackClusters : Merger.NMaxClusters();
  if (maxId > Merger.NMaxClusters()) {
    throw std::runtime_error("mNMaxClusters too small");
  }
  if (!param().rec.NonConsecutiveIDs) {
    unsigned int* sharedCount = (unsigned int*)MergerShadowAll.TmpMem() + CAMath::nextMultipleOf<4>(Merger.Memory()->nOutputTracks);
    runKernel<GPUMemClean16>({numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, sharedCount, maxId * sizeof(*sharedCount));
    runKernel<GPUMemClean16>({numBlocks, -ThreadCount(), 0, deviceType, RecoStep::TPCMerging}, krnlRunRangeNone, {}, MergerShadowAll.ClusterAttachment(), maxId * sizeof(*MergerShadowAll.ClusterAttachment()));
    runKernel<GPUTPCGMMergerPrepareClusters, 0>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
    CondWaitEvent(waitForTransfer, &mEvents->single);
    runKernel<GPUTPCGMMergerSortTracksQPt>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerPrepareClusters, 1>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerPrepareClusters, 2>(GetGridBlk(numBlocks, 0, deviceType), krnlRunRangeNone, krnlEventNone);
  }

  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpFitPrepare, mDebugFile);

  if (doGPUall) {
    CondWaitEvent(waitForTransfer, &mEvents->single);
    if (waitForTransfer) {
      ReleaseEvent(&mEvents->single);
    }
  } else if (doGPU) {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger);
  }
  if (GetDeviceProcessingSettings().fitSlowTracksInOtherPass && GetDeviceProcessingSettings().mergerSortTracks) {
    runKernel<GPUTPCGMMergerTrackFit>(GetGrid(Merger.NSlowTracks(), -WarpSize(), 0), krnlRunRangeNone, krnlEventNone, -1);
    runKernel<GPUTPCGMMergerTrackFit>(GetGridBlk(Merger.NOutputTracks() - Merger.NSlowTracks(), 0), krnlRunRangeNone, krnlEventNone, 1);
  } else {
    runKernel<GPUTPCGMMergerTrackFit>(GetGridBlk(Merger.NOutputTracks(), 0), krnlRunRangeNone, krnlEventNone, 0);
  }
  if (param().rec.retryRefit == 1) {
    runKernel<GPUTPCGMMergerTrackFit>(GetGridBlk(Merger.NOutputTracks(), 0), krnlRunRangeNone, krnlEventNone, -2);
  }
  if (param().rec.loopInterpolationInExtraPass) {
    runKernel<GPUTPCGMMergerFollowLoopers>(GetGridBlk(Merger.NOutputTracks(), 0), krnlRunRangeNone, krnlEventNone);
  }
  if (doGPU && !doGPUall) {
    TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger);
    SynchronizeGPU();
  }

  DoDebugAndDump(RecoStep::TPCMerging, 0, Merger, &GPUTPCGMMerger::DumpRefit, mDebugFile);
  runKernel<GPUTPCGMMergerFinalize, 0>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  if (!param().rec.NonConsecutiveIDs) {
    runKernel<GPUTPCGMMergerFinalize, 1>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
    runKernel<GPUTPCGMMergerFinalize, 2>(GetGridAuto(0, deviceType), krnlRunRangeNone, krnlEventNone);
  }
  DoDebugAndDump(RecoStep::TPCMerging, 0, doGPUall, Merger, &GPUTPCGMMerger::DumpFinal, mDebugFile);

  if (doGPUall) {
    RecordMarker(&mEvents->single, 0);
    TransferMemoryResourceLinkToHost(RecoStep::TPCMerging, Merger.MemoryResOutput(), mRec->NStreams() - 1, nullptr, &mEvents->single);
    ReleaseEvent(&mEvents->single);
    if (synchronizeOutput) {
      SynchronizeStream(mRec->NStreams() - 1);
    }
  } else {
    TransferMemoryResourcesToGPU(RecoStep::TPCMerging, &Merger);
  }
  if (GetDeviceProcessingSettings().keepDisplayMemory && !GetDeviceProcessingSettings().keepAllMemory) {
    TransferMemoryResourcesToHost(RecoStep::TPCMerging, &Merger, -1, true);
  }

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
  runKernel<GPUMemClean16>(GetGridAutoStep(0, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone, CompressorShadow.mClusterStatus, Compressor.mMaxClusters * sizeof(CompressorShadow.mClusterStatus[0]));
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step0attached>(GetGridAuto(0), krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step1unattached>(GetGridAuto(0), krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(myStep, &Compressor, 0);
  SynchronizeGPU();
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
  if (DeviceProcessingSettings().tpcCompressionGatherMode == 2) {
    TransferMemoryResourcesToGPU(myStep, &Compressor, 0);
    unsigned int nBlocks = 2;
    runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step2gather>(GetGridBlk(nBlocks, 0), krnlRunRangeNone, krnlEventNone);
    getKernelTimer<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step2gather>(RecoStep::TPCCompression, 0, outputSize);
  } else {
    char direction = 0;
    if (DeviceProcessingSettings().tpcCompressionGatherMode == 0) {
      P = &CompressorShadow.mPtrs;
    } else if (DeviceProcessingSettings().tpcCompressionGatherMode == 1) {
      P = &Compressor.mPtrs;
      direction = -1;
      gatherTimer = &getTimer<GPUTPCCompressionKernels>("GPUTPCCompression_GatherOnCPU", 0);
      gatherTimer->Start();
    }
    GPUMemCpyAlways(myStep, O->nSliceRowClusters, P->nSliceRowClusters, NSLICES * GPUCA_ROW_COUNT * sizeof(O->nSliceRowClusters[0]), 0, direction);
    GPUMemCpyAlways(myStep, O->nTrackClusters, P->nTrackClusters, O->nTracks * sizeof(O->nTrackClusters[0]), 0, direction);
    SynchronizeGPU();
    unsigned int offset = 0;
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        GPUMemCpyAlways(myStep, O->qTotU + offset, P->qTotU + mClusterNativeAccess->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->qTotU[0]), 0, direction);
        GPUMemCpyAlways(myStep, O->qMaxU + offset, P->qMaxU + mClusterNativeAccess->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->qMaxU[0]), 0, direction);
        GPUMemCpyAlways(myStep, O->flagsU + offset, P->flagsU + mClusterNativeAccess->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->flagsU[0]), 0, direction);
        GPUMemCpyAlways(myStep, O->padDiffU + offset, P->padDiffU + mClusterNativeAccess->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->padDiffU[0]), 0, direction);
        GPUMemCpyAlways(myStep, O->timeDiffU + offset, P->timeDiffU + mClusterNativeAccess->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->timeDiffU[0]), 0, direction);
        GPUMemCpyAlways(myStep, O->sigmaPadU + offset, P->sigmaPadU + mClusterNativeAccess->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->sigmaPadU[0]), 0, direction);
        GPUMemCpyAlways(myStep, O->sigmaTimeU + offset, P->sigmaTimeU + mClusterNativeAccess->clusterOffset[i][j], O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(O->sigmaTimeU[0]), 0, direction);
        offset += O->nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
      }
    }
    offset = 0;
    for (unsigned int i = 0; i < O->nTracks; i++) {
      GPUMemCpyAlways(myStep, O->qTotA + offset, P->qTotA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->qTotA[0]), 0, direction);
      GPUMemCpyAlways(myStep, O->qMaxA + offset, P->qMaxA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->qMaxA[0]), 0, direction);
      GPUMemCpyAlways(myStep, O->flagsA + offset, P->flagsA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->flagsA[0]), 0, direction);
      GPUMemCpyAlways(myStep, O->sigmaPadA + offset, P->sigmaPadA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->sigmaPadA[0]), 0, direction);
      GPUMemCpyAlways(myStep, O->sigmaTimeA + offset, P->sigmaTimeA + Compressor.mAttachedClusterFirstIndex[i], O->nTrackClusters[i] * sizeof(O->sigmaTimeA[0]), 0, direction);

      // First index stored with track
      GPUMemCpyAlways(myStep, O->rowDiffA + offset - i, P->rowDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (O->nTrackClusters[i] - 1) * sizeof(O->rowDiffA[0]), 0, direction);
      GPUMemCpyAlways(myStep, O->sliceLegDiffA + offset - i, P->sliceLegDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (O->nTrackClusters[i] - 1) * sizeof(O->sliceLegDiffA[0]), 0, direction);
      GPUMemCpyAlways(myStep, O->padResA + offset - i, P->padResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (O->nTrackClusters[i] - 1) * sizeof(O->padResA[0]), 0, direction);
      GPUMemCpyAlways(myStep, O->timeResA + offset - i, P->timeResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (O->nTrackClusters[i] - 1) * sizeof(O->timeResA[0]), 0, direction);
      offset += O->nTrackClusters[i];
    }
    GPUMemCpyAlways(myStep, O->qPtA, P->qPtA, O->nTracks * sizeof(O->qPtA[0]), 0, direction);
    GPUMemCpyAlways(myStep, O->rowA, P->rowA, O->nTracks * sizeof(O->rowA[0]), 0, direction);
    GPUMemCpyAlways(myStep, O->sliceA, P->sliceA, O->nTracks * sizeof(O->sliceA[0]), 0, direction);
    GPUMemCpyAlways(myStep, O->timeA, P->timeA, O->nTracks * sizeof(O->timeA[0]), 0, direction);
    GPUMemCpyAlways(myStep, O->padA, P->padA, O->nTracks * sizeof(O->padA[0]), 0, direction);
  }
  SynchronizeGPU();
  if (DeviceProcessingSettings().tpcCompressionGatherMode == 1) {
    gatherTimer->Stop();
  }

  mIOPtrs.tpcCompressedClusters = Compressor.mOutputFlat;
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

  Tracker.DoTracking(this);

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

  runKernel<GPUTRDTrackerKernels>(GetGridAuto(0), krnlRunRangeNone);
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
  if (GetDeviceProcessingSettings().debugLevel >= 6) {
    mDebugFile << "\n\nProcessing event " << mRec->getNEventsProcessed() << std::endl;
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
      if (RunTPCClusterizer(false)) {
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
    processors()->tpcMerger.SetSliceData(i, param().rec.mergerReadFromTrackerDirectly ? nullptr : processors()->tpcTrackers[i].Output());
  }
  if (GetRecoSteps().isSet(RecoStep::TPCMerging) && RunTPCTrackingMerger(false)) {
    return 1;
  }
  timerMerger.Stop();

  if (GetRecoSteps().isSet(RecoStep::TPCCompression) && mIOPtrs.clustersNative) {
    timerCompression.Start();
    if (RunTPCCompression()) {
      return 1;
    }
    if (GetDeviceProcessingSettings().runCompressionStatistics) {
      mCompressionStatistics->RunStatistics(mClusterNativeAccess.get(), processors()->tpcCompressor.mOutput, param());
    }
    timerCompression.Stop();
  }

  {
    const auto& threadContext = GetThreadContext();
    SynchronizeStream(mRec->NStreams() - 1);
  }

  if (needQA) {
    timerQA.Start();
    mQA->RunQA(!GetDeviceProcessingSettings().runQA);
    timerQA.Stop();
  }

  if (GetDeviceProcessingSettings().debugLevel >= 0) {
    int nCount = mRec->getNEventsProcessedInStat();
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
    int tmpSlice = GPUTPCGlobalTracking::GlobalTrackingSliceOrder(iSlice);
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
