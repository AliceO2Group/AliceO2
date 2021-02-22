// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTrackingCompression.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUTrackingInputProvider.h"

#ifdef HAVE_O2HEADERS
#include "GPUTPCCFChainContext.h"
#include "TPCClusterDecompressor.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

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

  if (ProcessingSettings().tpcCompressionGatherMode == 3) {
    mRec->AllocateVolatileDeviceMemory(0); // make future device memory allocation volatile
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
  size_t outputSize = AllocateRegisteredMemory(Compressor.mMemoryResOutputHost, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::compressedClusters)]);
  Compressor.mOutputFlat->set(outputSize, *Compressor.mOutput);
  char* hostFlatPtr = (char*)Compressor.mOutput->qTotU; // First array as allocated in GPUTPCCompression::SetPointersCompressedClusters
  size_t copySize = 0;
  if (ProcessingSettings().tpcCompressionGatherMode == 3) {
    CompressorShadow.mOutputA = Compressor.mOutput;
    copySize = AllocateRegisteredMemory(Compressor.mMemoryResOutputGPU); // We overwrite Compressor.mOutput with the allocated output pointers on the GPU
  }
  const o2::tpc::CompressedClustersPtrs* P = nullptr;
  HighResTimer* gatherTimer = nullptr;
  int outputStream = 0;
  if (ProcessingSettings().doublePipeline) {
    SynchronizeStream(mRec->NStreams() - 2); // Synchronize output copies running in parallel from memory that might be released, only the following async copy from stacked memory is safe after the chain finishes.
    outputStream = mRec->NStreams() - 2;
  }

  if (ProcessingSettings().tpcCompressionGatherMode >= 2) {
    if (ProcessingSettings().tpcCompressionGatherMode == 2) {
      void* devicePtr = mRec->getGPUPointer(Compressor.mOutputFlat);
      if (devicePtr != Compressor.mOutputFlat) {
        CompressedClustersPtrs& ptrs = *Compressor.mOutput; // We need to update the ptrs with the gpu-mapped version of the host address space
        for (unsigned int i = 0; i < sizeof(ptrs) / sizeof(void*); i++) {
          reinterpret_cast<char**>(&ptrs)[i] = reinterpret_cast<char**>(&ptrs)[i] + (reinterpret_cast<char*>(devicePtr) - reinterpret_cast<char*>(Compressor.mOutputFlat));
        }
      }
    }
    TransferMemoryResourcesToGPU(myStep, &Compressor, outputStream);
    constexpr unsigned int nBlocksDefault = 2;
    constexpr unsigned int nBlocksMulti = 1 + 2 * 200;
    switch (ProcessingSettings().tpcCompressionGatherModeKernel) {
      case 0:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::unbuffered>(GetGridBlkStep(nBlocksDefault, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::unbuffered>(RecoStep::TPCCompression, 0, outputSize);
        break;
      case 1:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered32>(GetGridBlkStep(nBlocksDefault, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered32>(RecoStep::TPCCompression, 0, outputSize);
        break;
      case 2:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered64>(GetGridBlkStep(nBlocksDefault, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered64>(RecoStep::TPCCompression, 0, outputSize);
        break;
      case 3:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered128>(GetGridBlkStep(nBlocksDefault, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered128>(RecoStep::TPCCompression, 0, outputSize);
        break;
      case 4:

        static_assert((nBlocksMulti & 1) && nBlocksMulti >= 3);
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::multiBlock>(GetGridBlkStep(nBlocksMulti, outputStream, RecoStep::TPCCompression), krnlRunRangeNone, krnlEventNone);
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::multiBlock>(RecoStep::TPCCompression, 0, outputSize);
        break;
      default:
        GPUError("Invalid compression kernel selected.");
        return 1;
    }
    if (ProcessingSettings().tpcCompressionGatherMode == 3) {
      RecordMarker(&mEvents->stream[outputStream], outputStream);
      char* deviceFlatPts = (char*)Compressor.mOutput->qTotU;
      if (GetProcessingSettings().doublePipeline) {
        const size_t blockSize = CAMath::nextMultipleOf<1024>(copySize / 30);
        const unsigned int n = (copySize + blockSize - 1) / blockSize;
        for (unsigned int i = 0; i < n; i++) {
          GPUMemCpy(myStep, hostFlatPtr + i * blockSize, deviceFlatPts + i * blockSize, CAMath::Min(blockSize, copySize - i * blockSize), outputStream, false);
        }
      } else {
        GPUMemCpy(myStep, hostFlatPtr, deviceFlatPts, copySize, outputStream, false);
      }
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
  if (ProcessingSettings().tpcCompressionGatherMode == 3) {
    SynchronizeEvents(&mEvents->stream[outputStream]);
    ReleaseEvent(&mEvents->stream[outputStream]);
    mRec->ReturnVolatileDeviceMemory();
  }

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
    this->AllocateRegisteredMemory(this->mInputsHost->mResourceClusterNativeOutput, this->mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
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
