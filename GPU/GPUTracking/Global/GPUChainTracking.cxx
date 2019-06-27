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
#include "GPUTPCConvert.h"
#include "GPUTPCCompression.h"
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
#include "ClusterNativeAccessExt.h"
#include "GPUTRDTrackletLabels.h"
#include "GPUDisplay.h"
#include "GPUQA.h"
#include "GPUTPCClusterStatistics.h"
#include "GPULogging.h"
#include "GPUReconstructionConvert.h"

#include "TPCFastTransform.h"

#include "utils/linux_helpers.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifdef HAVE_O2HEADERS
#include "TRDBase/TRDGeometryFlat.h"
#include "DetectorsBase/MatLayerCylSet.h"
#else
namespace o2
{
namespace trd
{
class TRDGeometryFlat
{
};
} // namespace trd
namespace base
{
class MatLayerCylSet
{
};
} // namespace base
} // namespace o2
#endif
using namespace o2::trd;

static constexpr unsigned int DUMP_HEADER_SIZE = 4;
static constexpr char DUMP_HEADER[DUMP_HEADER_SIZE + 1] = "CAv1";

GPUChainTracking::GPUChainTracking(GPUReconstruction* rec) : GPUChain(rec), mClusterNativeAccess(new ClusterNativeAccessExt)
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

  for (unsigned int i = 0; i < NSLICES; i++) {
    mRec->RegisterGPUProcessor(&processors()->tpcTrackers[i], GetRecoStepsGPU() & RecoStep::TPCSliceTracking);
  }
  processors()->tpcMerger.SetTrackingChain(this);
  mRec->RegisterGPUProcessor(&processors()->tpcMerger, GetRecoStepsGPU() & RecoStep::TPCMerging);
  processors()->trdTracker.SetTrackingChain(this);
  mRec->RegisterGPUProcessor(&processors()->trdTracker, GetRecoStepsGPU() & RecoStep::TRDTracking);
#ifdef HAVE_O2HEADERS
  mRec->RegisterGPUProcessor(&processors()->tpcConverter, GetRecoStepsGPU() & RecoStep::TPCConversion);
  mRec->RegisterGPUProcessor(&processors()->tpcCompressor, GetRecoStepsGPU() & RecoStep::TPCCompression);
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
    printf("Invalid Reconstruction Step Setting: dEdx requires TPC Merger to be active\n");
    return false;
  }
  if ((GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCdEdx) && !(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging)) {
    printf("Invalid GPU Reconstruction Step Setting: dEdx requires TPC Merger to be active\n");
    return false;
  }
#ifdef GPUCA_O2_LIB
  if (param().continuousMaxTimeBin) {
    if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCConversion)) {
      printf("Invalid Reconstruction Step Setting: Tracking requires TPC Conversion to be active\n");
      return false;
    }
    if ((GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) && !(GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion)) {
      printf("Invalid GPU Reconstruction Step Setting: Tracking requires TPC Conversion to be active\n");
      return false;
    }
  }
#endif
#ifndef GPUCA_ALIROOT_LIB
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging) && !(GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCClusters)) {
    printf("Invalid Inputs, TPC Clusters required\n");
    return false;
  }
#endif
  if (((GetRecoSteps() & GPUDataTypes::RecoStep::TPCConversion) ||
       (GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking) ||
       (GetRecoSteps() & GPUDataTypes::RecoStep::TPCCompression) ||
       (GetRecoSteps() & GPUDataTypes::RecoStep::TPCdEdx)) &&
      !(GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCClusters)) {
    printf("Invalid Inputs, TPC Clusters required\n");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging) && !((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking))) {
    printf("Input for TPC merger missing\n");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TPCCompression) && !((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCMergedTracks) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging))) {
    printf("Input for TPC compressor missing\n");
    return false;
  }
  if ((GetRecoSteps() & GPUDataTypes::RecoStep::TRDTracking) && (!((GetRecoStepsInputs() & GPUDataTypes::InOutType::TPCMergedTracks) || (GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging)) || !(GetRecoStepsInputs() & GPUDataTypes::InOutType::TRDTracklets))) {
    printf("Input for TRD Tracker missing\n");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCClusters) || (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TRDTracklets)) {
    printf("TPC Clusters / TRD Tracklets cannot be output\n");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking)) {
    printf("No TPC Slice Tracker Output available\n");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCMergedTracks) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCMerging)) {
    printf("No TPC Merged Track Output available\n");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCCompressedClusters) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TPCCompression)) {
    printf("No TPC Compression Output available\n");
    return false;
  }
  if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TRDTracks) && !(GetRecoSteps() & GPUDataTypes::RecoStep::TRDTracking)) {
    printf("No TRD Tracker Output available\n");
    return false;
  }
  return true;
}
int GPUChainTracking::Init()
{
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
    printf("Invalid GPU Reconstruction Step / Input / Output configuration\n");
    return 1;
  }

  if (GPUQA::QAAvailable() && (GetDeviceProcessingSettings().runQA || GetDeviceProcessingSettings().eventDisplay)) {
    mQA.reset(new GPUQA(this));
  }
  if (GetDeviceProcessingSettings().eventDisplay) {
    mEventDisplay.reset(new GPUDisplay(GetDeviceProcessingSettings().eventDisplay, this, mQA.get()));
  }

  if (mRec->IsGPU()) {
    if (mTPCFastTransform) {
      memcpy((void*)mFlatObjectsShadow.mTpcTransform, (const void*)mTPCFastTransform, sizeof(*mTPCFastTransform));
      memcpy((void*)mFlatObjectsShadow.mTpcTransformBuffer, (const void*)mTPCFastTransform->getFlatBufferPtr(), mTPCFastTransform->getFlatBufferSize());
      mFlatObjectsShadow.mTpcTransform->clearInternalBufferPtr();
      mFlatObjectsShadow.mTpcTransform->setActualBufferAddress(mFlatObjectsShadow.mTpcTransformBuffer);
      mFlatObjectsShadow.mTpcTransform->setFutureBufferAddress(mFlatObjectsDevice.mTpcTransformBuffer);
    }
#ifdef HAVE_O2HEADERS
    if (mMatLUT) {
      memcpy((void*)mFlatObjectsShadow.mMatLUT, (const void*)mMatLUT, sizeof(*mMatLUT));
      memcpy((void*)mFlatObjectsShadow.mMatLUTBuffer, (const void*)mMatLUT->getFlatBufferPtr(), mMatLUT->getFlatBufferSize());
      mFlatObjectsShadow.mMatLUT->clearInternalBufferPtr();
      mFlatObjectsShadow.mMatLUT->setActualBufferAddress(mFlatObjectsShadow.mMatLUTBuffer);
      mFlatObjectsShadow.mMatLUT->setFutureBufferAddress(mFlatObjectsDevice.mMatLUTBuffer);
    }
    if (mTRDGeometry) {
      memcpy((void*)mFlatObjectsShadow.mTrdGeometry, (const void*)mTRDGeometry, sizeof(*mTRDGeometry));
      mFlatObjectsShadow.mTrdGeometry->clearInternalBufferPtr();
    }
#endif
    TransferMemoryResourceLinkToGPU(mFlatObjectsShadow.mMemoryResFlat);
  }

  if (GetDeviceProcessingSettings().debugLevel >= 4) {
    mDebugFile.open(mRec->IsGPU() ? "GPU.out" : "CPU.out");
  }

  for (unsigned int i = 0; i < NSLICES; i++) {
    processors()->tpcTrackers[i].SetSlice(i);
  }

  return 0;
}

int GPUChainTracking::PrepareEvent()
{
  int offset = 0;
  if (mIOPtrs.clustersNative) {
    o2::tpc::ClusterNativeAccessFullTPC* tmp = mClusterNativeAccess.get();
    if (tmp != mIOPtrs.clustersNative) {
      *tmp = *mIOPtrs.clustersNative;
    }
    mIOPtrs.clustersNative = tmp;
    processors()->tpcConverter.mClustersNative = mClusterNativeAccess.get();

    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      int nCl = 0;
      for (unsigned int k = 0; k < o2::tpc::Constants::MAXGLOBALPADROW; k++) {
        nCl += mIOPtrs.clustersNative->nClusters[iSlice][k];
      }
      processors()->tpcTrackers[iSlice].Data().SetClusterData(nullptr, nCl, offset);
      offset += nCl;
    }
  } else {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processors()->tpcTrackers[iSlice].Data().SetClusterData(mIOPtrs.clusterData[iSlice], mIOPtrs.nClusterData[iSlice], offset);
      offset += mIOPtrs.nClusterData[iSlice];
    }
  }
  printf("Event has %d TPC Clusters\n", offset);
  processors()->tpcCompressor.mMaxClusters = offset;

  if (mRec->IsGPU()) {
    memcpy((void*)processorsShadow(), (const void*)processors(), sizeof(*processors()));
    mRec->ResetDeviceProcessorTypes();
  }
  return 0;
}

int GPUChainTracking::Finalize()
{
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
    computePointerWithAlignment(mem, mTpcTransform, 1);
    computePointerWithAlignment(mem, mTpcTransformBuffer, mChainTracking->GetTPCTransform()->getFlatBufferSize());
  }
#ifdef HAVE_O2HEADERS
  if (mChainTracking->GetMatLUT()) {
    computePointerWithAlignment(mem, mMatLUT, 1);
    computePointerWithAlignment(mem, mMatLUTBuffer, mChainTracking->GetMatLUT()->getFlatBufferSize());
  }
  if (mChainTracking->GetTRDGeometry()) {
    computePointerWithAlignment(mem, mTrdGeometry, 1);
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
  for (unsigned int i = 0; i < NSLICES * GPUCA_ROW_COUNT; i++) {
    AllocateIOMemoryHelper((&mClusterNativeAccess->nClusters[0][0])[i], (&mClusterNativeAccess->clusters[0][0])[i], mIOMem.clustersNative[i]);
  }
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
    DumpData(fp, &mIOPtrs.clustersNative->clusters[0][0], &mIOPtrs.clustersNative->nClusters[0][0], InOutPointerType::CLUSTERS_NATIVE);
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
  printf("Read %d hits\n", nTotal);
  mIOPtrs.nMCLabelsTPC = nTotal;
  AllocateIOMemoryHelper(nTotal, mIOPtrs.mcLabelsTPC, mIOMem.mcLabelsTPC);
  nRead = fread(mIOMem.mcLabelsTPC.get(), sizeof(*mIOPtrs.mcLabelsTPC), nTotal, fp);
  if (nRead != nTotal)
  {
    mIOPtrs.nMCLabelsTPC = 0;
  }
  else
  {
    printf("Read %d MC labels\n", nTotal);
    int nTracks;
    nRead = fread(&nTracks, sizeof(nTracks), 1, fp);
    if (nRead)
    {
      mIOPtrs.nMCInfosTPC = nTracks;
      AllocateIOMemoryHelper(nTracks, mIOPtrs.mcInfosTPC, mIOMem.mcInfosTPC);
      nRead = fread(mIOMem.mcInfosTPC.get(), sizeof(*mIOPtrs.mcInfosTPC), nTracks, fp);
      printf("Read %d MC Infos\n", nTracks);
    }
  }*/

  char buf[DUMP_HEADER_SIZE + 1] = "";
  size_t r = fread(buf, 1, DUMP_HEADER_SIZE, fp);
  if (strncmp(DUMP_HEADER, buf, DUMP_HEADER_SIZE)) {
    printf("Invalid file header\n");
    fclose(fp);
    return -1;
  }
  GeometryType geo;
  r = fread(&geo, sizeof(geo), 1, fp);
  if (geo != GPUReconstruction::geometryType) {
    printf("File has invalid geometry (%s v.s. %s)\n", GPUReconstruction::GEOMETRY_TYPE_NAMES[(int)geo], GPUReconstruction::GEOMETRY_TYPE_NAMES[(int)GPUReconstruction::geometryType]);
    fclose(fp);
    return 1;
  }
  (void)r;
  ReadData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, mIOMem.clusterData, InOutPointerType::CLUSTER_DATA);
  int nClustersTotal = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < mIOPtrs.nClusterData[i]; j++) {
      mIOMem.clusterData[i][j].id = nClustersTotal++;
    }
  }
  ReadData(fp, mIOPtrs.rawClusters, mIOPtrs.nRawClusters, mIOMem.rawClusters, InOutPointerType::RAW_CLUSTERS);
  mIOPtrs.clustersNative = ReadData<o2::tpc::ClusterNative>(fp, (const o2::tpc::ClusterNative**)&mClusterNativeAccess->clusters[0][0], &mClusterNativeAccess->nClusters[0][0], mIOMem.clustersNative, InOutPointerType::CLUSTERS_NATIVE) ? mClusterNativeAccess.get() : nullptr;
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

  return (0);
}

void GPUChainTracking::DumpSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "tpctransform.dump";
  if (mTPCFastTransform != nullptr) {
    DumpFlatObjectToFile(mTPCFastTransform, f.c_str());
  }

#ifdef HAVE_O2HEADERS
  f = dir;
  f += "matlut.dump";
  if (mMatLUT != nullptr) {
    DumpFlatObjectToFile(mMatLUT, f.c_str());
  }
  f = dir;
  f += "trdgeometry.dump";
  if (mTRDGeometry != nullptr) {
    DumpStructToFile(mTRDGeometry, f.c_str());
  }

#endif
}

void GPUChainTracking::ReadSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "tpctransform.dump";
  mTPCFastTransformU = ReadFlatObjectFromFile<TPCFastTransform>(f.c_str());
  mTPCFastTransform = mTPCFastTransformU.get();
#ifdef HAVE_O2HEADERS
  f = dir;
  f += "matlut.dump";
  mMatLUTU = ReadFlatObjectFromFile<o2::base::MatLayerCylSet>(f.c_str());
  mMatLUT = mMatLUTU.get();
  f = dir;
  f += "trdgeometry.dump";
  mTRDGeometryU = ReadStructFromFile<o2::trd::TRDGeometryFlat>(f.c_str());
  mTRDGeometry = mTRDGeometryU.get();
#endif
}

int GPUChainTracking::ConvertNativeToClusterData()
{
#ifdef HAVE_O2HEADERS
  ActivateThreadContext();
  mRec->SetThreadCounts(RecoStep::TPCConversion);
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCConversion;
  GPUTPCConvert& convert = processors()->tpcConverter;
  GPUTPCConvert& convertShadow = doGPU ? processorsShadow()->tpcConverter : convert;

  ClusterNativeAccessExt* tmpExt = mClusterNativeAccess.get();
  convert.set(tmpExt, mTPCFastTransform);
  SetupGPUProcessor(&convert, false);
  if (GetRecoStepsGPU() & RecoStep::TPCConversion) {
    convertShadow.set(convertShadow.mClustersNativeBuffer, mFlatObjectsDevice.mTpcTransform);
    *convert.mClustersNativeBuffer = *mClusterNativeAccess.get();
    for (unsigned int i = 0; i < NSLICES; i++) {
      convert.mMemory->clusters[i] = convertShadow.mClusters + tmpExt->clusterOffset[i][0];
      for (unsigned int j = 0; j < o2::tpc::Constants::MAXGLOBALPADROW; j++) {
        o2::tpc::ClusterNative* ptr = convertShadow.mInputClusters + convert.mClustersNativeBuffer->clusterOffset[i][j];
        convert.mClustersNativeBuffer->clusters[i][j] = ptr;
        mRec->GPUMemCpy(ptr, mClusterNativeAccess->clusters[i][j], sizeof(mClusterNativeAccess->clusters[i][j][0]) * mClusterNativeAccess->nClusters[i][j], 0, true);
      }
    }
  } else {
    for (unsigned int i = 0; i < NSLICES; i++) {
      convert.mMemory->clusters[i] = convert.mClusters + tmpExt->clusterOffset[i][0];
    }
  }

  WriteToConstantMemory((char*)&processors()->tpcConverter - (char*)processors(), &convertShadow, sizeof(convertShadow), 0);
  TransferMemoryResourcesToGPU(&convert, 0);
  runKernel<GPUTPCConvertKernel>({ NSLICES * GPUCA_ROW_COUNT, ThreadCount(), 0 }, nullptr, krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(&convert, 0);
  SynchronizeGPU();

  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.nClusterData[i] = (i == NSLICES - 1 ? convert.mNClustersTotal : tmpExt->clusterOffset[i + 1][0]) - tmpExt->clusterOffset[i][0];
    mIOPtrs.clusterData[i] = convert.mClusters + mClusterNativeAccess->clusterOffset[i][0];
    processors()->tpcTrackers[i].Data().SetClusterData(mIOPtrs.clusterData[i], mIOPtrs.nClusterData[i], mClusterNativeAccess->clusterOffset[i][0]);
  }

  ReleaseThreadContext();
#endif
  return 0;
}

void GPUChainTracking::ConvertNativeToClusterDataLegacy()
{
  o2::tpc::ClusterNativeAccessFullTPC* tmp = mClusterNativeAccess.get();
  if (tmp != mIOPtrs.clustersNative) {
    *tmp = *mIOPtrs.clustersNative;
  }
  GPUReconstructionConvert::ConvertNativeToClusterData(mClusterNativeAccess.get(), mIOMem.clusterData, mIOPtrs.nClusterData, mTPCFastTransform, param().continuousMaxTimeBin);
  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.clusterData[i] = mIOMem.clusterData[i].get();
  }
  mIOPtrs.clustersNative = nullptr;
  for (unsigned int i = 0; i < NSLICES * GPUCA_ROW_COUNT; i++) {
    mIOMem.clustersNative[i].reset(nullptr);
  }
  memset((void*)mClusterNativeAccess.get(), 0, sizeof(*mClusterNativeAccess));
}

void GPUChainTracking::ConvertRun2RawToNative()
{
  GPUReconstructionConvert::ConvertRun2RawToNative(mClusterNativeAccess.get(), mIOMem.clustersNative, mIOPtrs.rawClusters, mIOPtrs.nRawClusters);
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
  mTPCFastTransform = mTPCFastTransformU.get();
}

void GPUChainTracking::SetMatLUT(std::unique_ptr<o2::base::MatLayerCylSet>&& lut)
{
  mMatLUTU = std::move(lut);
  mMatLUT = mMatLUTU.get();
}

void GPUChainTracking::SetTRDGeometry(std::unique_ptr<o2::trd::TRDGeometryFlat>&& geo)
{
  mTRDGeometryU = std::move(geo);
  mTRDGeometry = mTRDGeometryU.get();
}

int GPUChainTracking::ReadEvent(int iSlice, int threadId)
{
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("Running ReadEvent for slice %d on thread %d\n", iSlice, threadId);
  }
  timerTPCtracking[iSlice][0].Start();
  if (processors()->tpcTrackers[iSlice].ReadEvent()) {
    return (1);
  }
  timerTPCtracking[iSlice][0].Stop();
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
  timerTPCtracking[iSlice][9].Start();
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
  timerTPCtracking[iSlice][9].Stop();
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("Finished WriteOutput for slice %d on thread %d\n", iSlice, threadId);
  }
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

  timerTPCtracking[iSlice][8].Start();
  processors()->tpcTrackers[iSlice].PerformGlobalTracking(processors()->tpcTrackers[sliceLeft], processors()->tpcTrackers[sliceRight]);
  timerTPCtracking[iSlice][8].Stop();

  mSliceLeftGlobalReady[sliceLeft] = 1;
  mSliceRightGlobalReady[sliceRight] = 1;
  if (GetDeviceProcessingSettings().debugLevel >= 5) {
    GPUInfo("GPU Tracker finished Global Tracking for slice %d on thread %d\n", iSlice, threadId);
  }
  return (0);
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

  ActivateThreadContext();
  mRec->SetThreadCounts(RecoStep::TPCSliceTracking);

  int retVal = RunTPCTrackingSlices_internal();
  if (retVal) {
    SynchronizeGPU();
  }
  if (retVal >= 2) {
    ResetHelperThreads(retVal >= 3);
  }
  ReleaseThreadContext();
  return (retVal != 0);
}

int GPUChainTracking::RunTPCTrackingSlices_internal()
{
  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Slice Tracker");
  }
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCSliceTracking;

  bool streamInit[GPUCA_MAX_STREAMS] = { false };
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

    WriteToConstantMemory((char*)&processors()->param - (char*)processors(), &param(), sizeof(GPUParam), mRec->NStreams() - 1);
    WriteToConstantMemory((char*)processors()->tpcTrackers - (char*)processors(), processorsShadow()->tpcTrackers, sizeof(GPUTPCTracker) * NSLICES, mRec->NStreams() - 1, &mEvents->init);

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
#ifdef GPUCA_HAVE_OPENMP
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
    runKernel<GPUMemClean16>({ BlockCount(), ThreadCount(), useStream }, &timerTPCtracking[iSlice][5], krnlRunRangeNone, {}, trkShadow.Data().HitWeights(), trkShadow.Data().NumberOfHitsPlusAlign() * sizeof(*trkShadow.Data().HitWeights()));

    // Copy Data to GPU Global Memory
    timerTPCtracking[iSlice][0].Start();
    TransferMemoryResourcesToGPU(&trk);
    if (GPUDebug("Initialization (3)", useStream)) {
      throw std::runtime_error("memcpy failure");
    }
    timerTPCtracking[iSlice][0].Stop();

    runKernel<GPUTPCNeighboursFinder>({ GPUCA_ROW_COUNT, FinderThreadCount(), useStream }, &timerTPCtracking[iSlice][1], { iSlice }, { nullptr, streamInit[useStream] ? nullptr : &mEvents->init });
    streamInit[useStream] = true;

    if (GetDeviceProcessingSettings().keepAllMemory) {
      TransferMemoryResourcesToHost(&trk, -1, true);
      memcpy(trk.LinkTmpMemory(), mRec->Res(trk.MemoryResLinksScratch()).Ptr(), mRec->Res(trk.MemoryResLinksScratch()).Size());
      if (GetDeviceProcessingSettings().debugMask & 2) {
        trk.DumpLinks(mDebugFile);
      }
    }

    runKernel<GPUTPCNeighboursCleaner>({ GPUCA_ROW_COUNT - 2, ThreadCount(), useStream }, &timerTPCtracking[iSlice][2], { iSlice });

    if (GetDeviceProcessingSettings().debugLevel >= 4) {
      TransferMemoryResourcesToHost(&trk, -1, true);
      if (GetDeviceProcessingSettings().debugMask & 4) {
        trk.DumpLinks(mDebugFile);
      }
    }

    runKernel<GPUTPCStartHitsFinder>({ GPUCA_ROW_COUNT - 6, ThreadCount(), useStream }, &timerTPCtracking[iSlice][3], { iSlice });

    if (doGPU) {
      runKernel<GPUTPCStartHitsSorter>({ BlockCount(), ThreadCount(), useStream }, &timerTPCtracking[iSlice][4], { iSlice });
    }

    if (GetDeviceProcessingSettings().debugLevel >= 2) {
      TransferMemoryResourceLinkToHost(trk.MemoryResCommon(), -1);
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        GPUInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *trk.NTracklets(), iSlice);
      }
    }

    if (GetDeviceProcessingSettings().debugLevel >= 4 && *trk.NTracklets()) {
      TransferMemoryResourcesToHost(&trk, -1, true);
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
      runKernel<GPUTPCTrackletConstructor>({ ConstructorBlockCount(), ConstructorThreadCount(), useStream }, &timerTPCtracking[iSlice][6], { iSlice });
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        printf("Slice %u, Number of tracklets: %d\n", iSlice, *trk.NTracklets());
      }
      if (GetDeviceProcessingSettings().debugMask & 128) {
        trk.DumpTrackletHits(mDebugFile);
      }
      if (GetDeviceProcessingSettings().debugMask & 256 && !GetDeviceProcessingSettings().comparableDebutOutput) {
        trk.DumpHitWeights(mDebugFile);
      }
    }

    if (!doGPU || GetDeviceProcessingSettings().trackletSelectorInPipeline) {
      runKernel<GPUTPCTrackletSelector>({ SelectorBlockCount(), SelectorThreadCount(), useStream }, &timerTPCtracking[iSlice][7], { iSlice });
      TransferMemoryResourceLinkToHost(trk.MemoryResCommon(), useStream, &mEvents->selector[iSlice]);
      streamMap[iSlice] = useStream;
      if (GetDeviceProcessingSettings().debugLevel >= 3) {
        printf("Slice %u, Number of tracks: %d\n", iSlice, *trk.NTracks());
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
        runKernel<GPUTPCTrackletConstructor, 1>({ ConstructorBlockCount(), ConstructorThreadCount(), 0 }, &timerTPCtracking[0][6], krnlRunRangeNone, { &mEvents->constructor, mEvents->stream, mRec->NStreams() });
        for (int i = 0; i < mRec->NStreams(); i++) {
          ReleaseEvent(&mEvents->stream[i]);
        }
        SynchronizeEvents(&mEvents->constructor);
        ReleaseEvent(&mEvents->constructor);
      }

      if (GetDeviceProcessingSettings().debugLevel >= 4) {
        for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
          TransferMemoryResourcesToHost(&processors()->tpcTrackers[iSlice], -1, true);
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
        runKernel<GPUTPCTrackletSelector>({ SelectorBlockCount(), SelectorThreadCount(), useStream }, &timerTPCtracking[iSlice][7], { iSlice, (int)runSlices });
        for (unsigned int k = iSlice; k < iSlice + runSlices; k++) {
          TransferMemoryResourceLinkToHost(processors()->tpcTrackers[k].MemoryResCommon(), useStream, &mEvents->selector[k]);
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
          TransferMemoryResourceLinkToHost(processors()->tpcTrackers[tmpSlice].MemoryResOutput(), streamMap[tmpSlice], &mEvents->selector[tmpSlice]);
        } else {
          transferRunning[tmpSlice] = false;
        }
        tmpSlice++;
      }

      if (GetDeviceProcessingSettings().keepAllMemory) {
        TransferMemoryResourcesToHost(&processors()->tpcTrackers[iSlice], -1, true);
        if (GetDeviceProcessingSettings().debugMask & 256 && !GetDeviceProcessingSettings().comparableDebutOutput) {
          processors()->tpcTrackers[iSlice].DumpHitWeights(mDebugFile);
        }
        if (GetDeviceProcessingSettings().debugMask & 512) {
          processors()->tpcTrackers[iSlice].DumpTrackHits(mDebugFile);
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
    if (param().rec.GlobalTracking) {
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
        GlobalTracking(iSlice, 0);
      }
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
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

  if (param().rec.GlobalTracking) {
    if (GetDeviceProcessingSettings().debugLevel >= 3) {
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
        GPUInfo("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d", iSlice, processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTracks, processors()->tpcTrackers[iSlice].CommonMemory()->nTracks, processors()->tpcTrackers[iSlice].CommonMemory()->nLocalTrackHits,
                processors()->tpcTrackers[iSlice].CommonMemory()->nTrackHits);
      }
    }
  }

  if (GetDeviceProcessingSettings().debugMask & 1024) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      processors()->tpcTrackers[i].DumpOutput(stdout);
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
  ActivateThreadContext();
  mRec->SetThreadCounts(RecoStep::TPCMerging);

  HighResTimer timer;
  static double times[10] = {};
  static int nCount = 0;
  if (GetDeviceProcessingSettings().resetTimers || !GPUCA_TIMING_SUM) {
    for (unsigned int k = 0; k < sizeof(times) / sizeof(times[0]); k++) {
      times[k] = 0;
    }
    nCount = 0;
  }

  Merger.SetMatLUT(mMatLUT);
  SetupGPUProcessor(&Merger, true);

  timer.ResetStart();
  Merger.UnpackSlices();
  times[0] += timer.GetCurrentElapsedTime(true);
  Merger.MergeWithingSlices();

  times[1] += timer.GetCurrentElapsedTime(true);
  Merger.MergeSlices();

  times[2] += timer.GetCurrentElapsedTime(true);
  Merger.MergeCEInit();

  times[3] += timer.GetCurrentElapsedTime(true);
  Merger.CollectMergedTracks();

  times[4] += timer.GetCurrentElapsedTime(true);
  Merger.MergeCE();

  times[3] += timer.GetCurrentElapsedTime(true);
  Merger.PrepareClustersForFit();

  times[5] += timer.GetCurrentElapsedTime(true);

  if (doGPU) {
    SetupGPUProcessor(&Merger, false);
    MergerShadow.OverrideSliceTracker(processorsDevice()->tpcTrackers);
    MergerShadow.SetMatLUT(mFlatObjectsShadow.mMatLUT);
  }

  WriteToConstantMemory((char*)&processors()->tpcMerger - (char*)processors(), &MergerShadow, sizeof(MergerShadow), 0);
  TransferMemoryResourceLinkToGPU(Merger.MemoryResRefit());
  times[6] += timer.GetCurrentElapsedTime(true);

  runKernel<GPUTPCGMMergerTrackFit>({ BlockCount(), ThreadCount(), 0 }, nullptr, krnlRunRangeNone);
  SynchronizeGPU();
  times[7] += timer.GetCurrentElapsedTime(true);

  TransferMemoryResourceLinkToHost(Merger.MemoryResRefit());
  SynchronizeGPU();
  times[8] += timer.GetCurrentElapsedTime(true);

  Merger.Finalize();
  times[9] += timer.GetCurrentElapsedTime(true);

  nCount++;
  if (GetDeviceProcessingSettings().debugLevel > 0) {
    int copysize = 4 * Merger.NOutputTrackClusters() * sizeof(float) + Merger.NOutputTrackClusters() * sizeof(unsigned int) + Merger.NOutputTracks() * sizeof(GPUTPCGMMergedTrack) + 6 * sizeof(float) + sizeof(GPUParam);
    printf("Merge Time:\tUnpack Slices:\t%'7d us\n", (int)(times[0] * 1000000 / nCount));
    printf("\t\tMerge Within:\t%'7d us\n", (int)(times[1] * 1000000 / nCount));
    printf("\t\tMerge Slices:\t%'7d us\n", (int)(times[2] * 1000000 / nCount));
    printf("\t\tMerge CE:\t%'7d us\n", (int)(times[3] * 1000000 / nCount));
    printf("\t\tCollect:\t%'7d us\n", (int)(times[4] * 1000000 / nCount));
    printf("\t\tClusters:\t%'7d us\n", (int)(times[5] * 1000000 / nCount));
    double speed = (double)copysize / std::max(1., times[6]) * nCount / 1e9;
    if (doGPU) {
      printf("\t\tCopy From:\t%'7d us (%6.3f GB/s)\n", (int)(times[6] * 1000000 / nCount), speed);
    }
    printf("\t\tRefit:\t\t%'7d us\n", (int)(times[7] * 1000000 / nCount));
    speed = (double)copysize / std::max(1., times[8]) * nCount / 1e9;
    if (doGPU) {
      printf("\t\tCopy To:\t%'7d us (%6.3f GB/s)\n", (int)(times[8] * 1000000 / nCount), speed);
    }
    printf("\t\tFinalize:\t%'7d us\n", (int)(times[9] * 1000000 / nCount));
  }

  mIOPtrs.mergedTracks = Merger.OutputTracks();
  mIOPtrs.nMergedTracks = Merger.NOutputTracks();
  mIOPtrs.mergedTrackHits = Merger.Clusters();
  mIOPtrs.nMergedTrackHits = Merger.NOutputTrackClusters();

  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Merger Finished (output clusters %d / input clusters %d)", Merger.NOutputTrackClusters(), Merger.NClusters());
  }
  ReleaseThreadContext();
  return 0;
}

int GPUChainTracking::RunTPCCompression()
{
#ifdef HAVE_O2HEADERS
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCCompression;
  GPUTPCCompression& Compressor = processors()->tpcCompressor;
  GPUTPCCompression& CompressorShadow = doGPU ? processorsShadow()->tpcCompressor : Compressor;
  ActivateThreadContext();
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

  WriteToConstantMemory((char*)&processors()->tpcCompressor - (char*)processors(), &CompressorShadow, sizeof(CompressorShadow), 0);
  TransferMemoryResourcesToGPU(&Compressor, 0);
  runKernel<GPUMemClean16>({ BlockCount(), ThreadCount(), 0 }, nullptr, krnlRunRangeNone, krnlEventNone, CompressorShadow.mClusterStatus, Compressor.mMaxClusters * sizeof(CompressorShadow.mClusterStatus[0]));
  runKernel<GPUTPCCompressionKernels, 0>({ BlockCount(), ThreadCount(), 0 }, nullptr, krnlRunRangeNone, krnlEventNone);
  runKernel<GPUTPCCompressionKernels, 1>({ BlockCount(), ThreadCount(), 0 }, nullptr, krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(&Compressor, 0);
  SynchronizeGPU();
  memset((void*)&Compressor.mOutput, 0, sizeof(Compressor.mOutput));
  Compressor.mOutput.nTracks = Compressor.mMemory->nStoredTracks;
  Compressor.mOutput.nAttachedClusters = Compressor.mMemory->nStoredAttachedClusters;
  Compressor.mOutput.nUnattachedClusters = Compressor.mMemory->nStoredUnattachedClusters;
  Compressor.mOutput.nAttachedClustersReduced = Compressor.mOutput.nAttachedClusters - Compressor.mOutput.nTracks;
  Compressor.mOutput.nSliceRows = NSLICES * GPUCA_ROW_COUNT;
  Compressor.mOutput.nComppressionModes = param().rec.tpcCompressionModes;
  AllocateRegisteredMemory(Compressor.mMemoryResOutputHost);
  mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.nSliceRowClusters, CompressorShadow.mPtrs.nSliceRowClusters, NSLICES * GPUCA_ROW_COUNT * sizeof(Compressor.mOutput.nSliceRowClusters[0]), 0, false);
  mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.nTrackClusters, CompressorShadow.mPtrs.nTrackClusters, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.nTrackClusters[0]), 0, false);
  SynchronizeGPU();
  unsigned int offset = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.qTotU + offset, CompressorShadow.mPtrs.qTotU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.qTotU[0]), 0, false);
      mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.qMaxU + offset, CompressorShadow.mPtrs.qMaxU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.qMaxU[0]), 0, false);
      mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.flagsU + offset, CompressorShadow.mPtrs.flagsU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.flagsU[0]), 0, false);
      mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.padDiffU + offset, CompressorShadow.mPtrs.padDiffU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.padDiffU[0]), 0, false);
      mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.timeDiffU + offset, CompressorShadow.mPtrs.timeDiffU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.timeDiffU[0]), 0, false);
      mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.sigmaPadU + offset, CompressorShadow.mPtrs.sigmaPadU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.sigmaPadU[0]), 0, false);
      mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.sigmaTimeU + offset, CompressorShadow.mPtrs.sigmaTimeU + mClusterNativeAccess->clusterOffset[i][j], Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j] * sizeof(Compressor.mOutput.sigmaTimeU[0]), 0, false);
      offset += Compressor.mOutput.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
    }
  }
  offset = 0;
  for (unsigned int i = 0; i < Compressor.mOutput.nTracks; i++) {
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.qTotA + offset, CompressorShadow.mPtrs.qTotA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.qTotA[0]), 0, false);
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.qMaxA + offset, CompressorShadow.mPtrs.qMaxA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.qMaxA[0]), 0, false);
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.flagsA + offset, CompressorShadow.mPtrs.flagsA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.flagsA[0]), 0, false);
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.sigmaPadA + offset, CompressorShadow.mPtrs.sigmaPadA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.sigmaPadA[0]), 0, false);
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.sigmaTimeA + offset, CompressorShadow.mPtrs.sigmaTimeA + Compressor.mAttachedClusterFirstIndex[i], Compressor.mOutput.nTrackClusters[i] * sizeof(Compressor.mOutput.sigmaTimeA[0]), 0, false);

    // First index stored with track
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.rowDiffA + offset - i, CompressorShadow.mPtrs.rowDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.rowDiffA[0]), 0, false);
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.sliceLegDiffA + offset - i, CompressorShadow.mPtrs.sliceLegDiffA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.sliceLegDiffA[0]), 0, false);
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.padResA + offset - i, CompressorShadow.mPtrs.padResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.padResA[0]), 0, false);
    mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.timeResA + offset - i, CompressorShadow.mPtrs.timeResA + Compressor.mAttachedClusterFirstIndex[i] + 1, (Compressor.mOutput.nTrackClusters[i] - 1) * sizeof(Compressor.mOutput.timeResA[0]), 0, false);
    offset += Compressor.mOutput.nTrackClusters[i];
  }
  mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.qPtA, CompressorShadow.mPtrs.qPtA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.qPtA[0]), 0, false);
  mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.rowA, CompressorShadow.mPtrs.rowA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.rowA[0]), 0, false);
  mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.sliceA, CompressorShadow.mPtrs.sliceA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.sliceA[0]), 0, false);
  mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.timeA, CompressorShadow.mPtrs.timeA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.timeA[0]), 0, false);
  mRec->GPUMemCpyAlways(doGPU, Compressor.mOutput.padA, CompressorShadow.mPtrs.padA, Compressor.mOutput.nTracks * sizeof(Compressor.mOutput.padA[0]), 0, false);

  SynchronizeGPU();

  mIOPtrs.tpcCompressedClusters = &Compressor.mOutput;

  ReleaseThreadContext();
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

  Tracker.SetMaxData();
  if (GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    AllocateRegisteredMemory(Tracker.MemoryTracks());
    AllocateRegisteredMemory(Tracker.MemoryTracklets());
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
#ifdef GPUCA_BUILD_TRD
  bool doGPU = GetRecoStepsGPU() & RecoStep::TRDTracking;
  GPUTRDTracker& Tracker = processors()->trdTracker;
  GPUTRDTracker& TrackerShadow = doGPU ? processorsShadow()->trdTracker : Tracker;

  ActivateThreadContext();
  SetupGPUProcessor(&Tracker, false);
  TrackerShadow.SetGeometry(reinterpret_cast<GPUTRDGeometry*>(mFlatObjectsDevice.mTrdGeometry));

  WriteToConstantMemory((char*)&processors()->trdTracker - (char*)processors(), &TrackerShadow, sizeof(TrackerShadow), 0);
  TransferMemoryResourcesToGPU(&Tracker);

  runKernel<GPUTRDTrackerGPU>({ BlockCount(), TRDThreadCount(), 0 }, nullptr, krnlRunRangeNone);
  SynchronizeGPU();

  TransferMemoryResourcesToHost(&Tracker);
  SynchronizeGPU();

  if (GetDeviceProcessingSettings().debugLevel >= 2) {
    GPUInfo("GPU TRD tracker Finished");
  }

  ReleaseThreadContext();
#endif
  return (0);
}

int GPUChainTracking::RunChain()
{
  if (GetDeviceProcessingSettings().runCompressionStatistics && mCompressionStatistics == nullptr) {
    mCompressionStatistics.reset(new GPUTPCClusterStatistics);
  }
  const bool needQA = GPUQA::QAAvailable() && (GetDeviceProcessingSettings().runQA || (GetDeviceProcessingSettings().eventDisplay && mIOPtrs.nMCInfosTPC));
  if (needQA && mQAInitialized == false) {
    if (mQA->InitQA()) {
      return 1;
    }
    mQAInitialized = true;
  }
  static HighResTimer timerTracking, timerMerger, timerQA, timerTransform, timerCompression;
  static int nCount = 0;
  if (GetDeviceProcessingSettings().resetTimers) {
    timerTracking.Reset();
    timerMerger.Reset();
    timerQA.Reset();
    timerTransform.Reset();
    timerCompression.Reset();
    nCount = 0;
  }

  try {
    mRec->PrepareEvent();
  } catch (const std::bad_alloc& e) {
    printf("Memory Allocation Error\n");
    return (1);
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
    // printf("slice %d clusters %d tracks %d\n", i, mClusterData[i].NumberOfClusters(), processors()->tpcTrackers[i].Output()->NTracks());
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

  nCount++;
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
    if (mIOPtrs.clustersNative) {
      printf("TPC Transformation Time: %'d us\n", (int)(1000000 * timerTransform.GetElapsedTime() / nCount));
    }
    if (mIOPtrs.clustersNative) {
      printf("TPC Compression Time: %'d us\n", (int)(1000000 * timerCompression.GetElapsedTime() / nCount));
    }
  }

  if (GetDeviceProcessingSettings().debugLevel >= 1) {
    const char* tmpNames[10] = { "Initialisation", "Neighbours Finder", "Neighbours Cleaner", "Starts Hits Finder", "Start Hits Sorter", "Weight Cleaner", "Tracklet Constructor", "Tracklet Selector", "Global Tracking", "Write Output" };

    for (int i = 0; i < 10; i++) {
      double time = 0;
      for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
        time += timerTPCtracking[iSlice][i].GetElapsedTime();
        timerTPCtracking[iSlice][i].Reset();
      }
      time /= NSLICES;
      if (!(GetRecoStepsGPU() & RecoStep::TPCSliceTracking)) {
        time /= GetDeviceProcessingSettings().nThreads;
      }

      printf("Execution Time: Task: %20s Time: %'7d us\n", tmpNames[i], (int)(time * 1000000 / nCount));
    }
    printf("Execution Time: Task: %20s Time: %'7d us\n", "Merger", (int)(timerMerger.GetElapsedTime() * 1000000. / nCount));
    if (!GPUCA_TIMING_SUM) {
      timerTracking.Reset();
      timerMerger.Reset();
      timerQA.Reset();
      nCount = 0;
    }
  }

  if (GetRecoSteps().isSet(RecoStep::TRDTracking)) {
    if (mIOPtrs.nTRDTracklets) {
      HighResTimer timer;
      timer.Start();
      if (RunTRDTracking()) {
        return 1;
      }
      if (GetDeviceProcessingSettings().debugLevel >= 1) {
        printf("TRD tracking time: %'d us\n", (int)(1000000 * timer.GetCurrentElapsedTime()));
      }
    } else {
      processors()->trdTracker.Reset();
    }
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
      printf("Press key for next event!\n");
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
    printf("Loading next event\n");

    mEventDisplay->WaitForNextEvent();
  }
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
