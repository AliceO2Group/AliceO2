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

/// \file GPUChainTrackingCompression.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPUChainTrackingDebug.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUTrackingInputProvider.h"
#include "GPUTPCCFChainContext.h"
#include "TPCClusterDecompressor.h"
#include "GPUDefParametersRuntime.h"
#include "GPUConstantMem.h" // TODO: Try to get rid of as many GPUConstantMem includes as possible!
#include "GPUTPCCompressionKernels.h"
#include "GPUTPCDecompressionKernels.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "utils/strtag.h"

#include <numeric>

using namespace o2::gpu;
using namespace o2::tpc;

int32_t GPUChainTracking::RunTPCCompression()
{
  mRec->PushNonPersistentMemory(qStr2Tag("TPCCOMPR"));
  RecoStep myStep = RecoStep::TPCCompression;
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCCompression;
  int32_t gatherMode = mRec->GetProcessingSettings().tpcCompressionGatherMode == -1 ? mRec->getGPUParameters(doGPU).par_COMP_GATHER_MODE : mRec->GetProcessingSettings().tpcCompressionGatherMode;
  GPUTPCCompression& Compressor = processors()->tpcCompressor;
  GPUTPCCompression& CompressorShadow = doGPU ? processorsShadow()->tpcCompressor : Compressor;
  const auto& threadContext = GetThreadContext();
  if (mPipelineFinalizationCtx && GetProcessingSettings().doublePipelineClusterizer) {
    RecordMarker(&mEvents->single, 0);
  }

  if (gatherMode == 3) {
    mRec->MakeFutureDeviceMemoryAllocationsVolatile();
  }
  SetupGPUProcessor(&Compressor, true);
  new (Compressor.mMemory) GPUTPCCompression::memory;
  WriteToConstantMemory(myStep, (char*)&processors()->tpcCompressor - (char*)processors(), &CompressorShadow, sizeof(CompressorShadow), 0);
  TransferMemoryResourcesToGPU(myStep, &Compressor, 0);
  runKernel<GPUMemClean16>(GetGridAutoStep(0, RecoStep::TPCCompression), CompressorShadow.mClusterStatus, Compressor.mMaxClusters * sizeof(CompressorShadow.mClusterStatus[0]));
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step0attached>(GetGridAuto(0));
  if (GetProcessingSettings().tpcWriteClustersAfterRejection) {
    WriteReducedClusters();
  }
  runKernel<GPUTPCCompressionKernels, GPUTPCCompressionKernels::step1unattached>(GetGridAuto(0));
  TransferMemoryResourcesToHost(myStep, &Compressor, 0);
#ifndef GPUCA_RUN2
  if (mPipelineFinalizationCtx && GetProcessingSettings().doublePipelineClusterizer) {
    SynchronizeEventAndRelease(mEvents->single);
    auto* foreignChain = (GPUChainTracking*)GetNextChainInQueue();
    foreignChain->RunTPCClusterizer_prepare(false);
    foreignChain->mCFContext->ptrClusterNativeSave = processorsShadow()->ioPtrs.clustersNative;
  }
#endif
  SynchronizeStream(0);
  o2::tpc::CompressedClusters* O = Compressor.mOutput;
  memset((void*)O, 0, sizeof(*O));
  O->nTracks = Compressor.mMemory->nStoredTracks;
  O->nAttachedClusters = Compressor.mMemory->nStoredAttachedClusters;
  O->nUnattachedClusters = Compressor.mMemory->nStoredUnattachedClusters;
  O->nAttachedClustersReduced = O->nAttachedClusters - O->nTracks;
  O->nSliceRows = NSECTORS * GPUTPCGeometry::NROWS;
  O->nComppressionModes = param().rec.tpc.compressionTypeMask;
  O->solenoidBz = param().bzkG;
  O->maxTimeBin = param().continuousMaxTimeBin;
  size_t outputSize = AllocateRegisteredMemory(Compressor.mMemoryResOutputHost, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::compressedClusters)]);
  Compressor.mOutputFlat->set(outputSize, *Compressor.mOutput);
  char* hostFlatPtr = (char*)Compressor.mOutput->qTotU; // First array as allocated in GPUTPCCompression::SetPointersCompressedClusters
  size_t copySize = 0;
  if (gatherMode == 3) {
    CompressorShadow.mOutputA = Compressor.mOutput;
    copySize = AllocateRegisteredMemory(Compressor.mMemoryResOutputGPU); // We overwrite Compressor.mOutput with the allocated output pointers on the GPU
  }
  const o2::tpc::CompressedClustersPtrs* P = nullptr;
  HighResTimer* gatherTimer = nullptr;
  int32_t outputStream = 0;
  if (GetProcessingSettings().doublePipeline) {
    SynchronizeStream(OutputStream()); // Synchronize output copies running in parallel from memory that might be released, only the following async copy from stacked memory is safe after the chain finishes.
    outputStream = OutputStream();
  }
  if (gatherMode >= 2) {
    if (gatherMode == 2) {
      void* devicePtr = mRec->getGPUPointer(Compressor.mOutputFlat);
      if (devicePtr != Compressor.mOutputFlat) {
        CompressedClustersPtrs& ptrs = *Compressor.mOutput; // We need to update the ptrs with the gpu-mapped version of the host address space
        for (uint32_t i = 0; i < sizeof(ptrs) / sizeof(void*); i++) {
          reinterpret_cast<char**>(&ptrs)[i] = reinterpret_cast<char**>(&ptrs)[i] + (reinterpret_cast<char*>(devicePtr) - reinterpret_cast<char*>(Compressor.mOutputFlat));
        }
      }
    }
    TransferMemoryResourcesToGPU(myStep, &Compressor, outputStream);
    constexpr uint32_t nBlocksDefault = 2;
    constexpr uint32_t nBlocksMulti = 1 + 2 * 200;
    int32_t gatherModeKernel = mRec->GetProcessingSettings().tpcCompressionGatherModeKernel == -1 ? mRec->getGPUParameters(doGPU).par_COMP_GATHER_KERNEL : mRec->GetProcessingSettings().tpcCompressionGatherMode;
    switch (gatherModeKernel) {
      case 0:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::unbuffered>(GetGridBlkStep(nBlocksDefault, outputStream, RecoStep::TPCCompression));
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::unbuffered>(RecoStep::TPCCompression, 0, outputSize, false);
        break;
      case 1:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered32>(GetGridBlkStep(nBlocksDefault, outputStream, RecoStep::TPCCompression));
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered32>(RecoStep::TPCCompression, 0, outputSize, false);
        break;
      case 2:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered64>(GetGridBlkStep(nBlocksDefault, outputStream, RecoStep::TPCCompression));
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered64>(RecoStep::TPCCompression, 0, outputSize, false);
        break;
      case 3:
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered128>(GetGridBlkStep(nBlocksDefault, outputStream, RecoStep::TPCCompression));
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::buffered128>(RecoStep::TPCCompression, 0, outputSize, false);
        break;
      case 4:
        static_assert((nBlocksMulti & 1) && nBlocksMulti >= 3);
        runKernel<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::multiBlock>(GetGridBlkStep(nBlocksMulti, outputStream, RecoStep::TPCCompression));
        getKernelTimer<GPUTPCCompressionGatherKernels, GPUTPCCompressionGatherKernels::multiBlock>(RecoStep::TPCCompression, 0, outputSize, false);
        break;
      default:
        GPUError("Invalid compression kernel %d selected.", (int32_t)gatherModeKernel);
        return 1;
    }
    if (gatherMode == 3) {
      RecordMarker(&mEvents->stream[outputStream], outputStream);
      char* deviceFlatPts = (char*)Compressor.mOutput->qTotU;
      if (GetProcessingSettings().doublePipeline) {
        const size_t blockSize = CAMath::nextMultipleOf<1024>(copySize / 30);
        const uint32_t n = (copySize + blockSize - 1) / blockSize;
        for (uint32_t i = 0; i < n; i++) {
          GPUMemCpy(myStep, hostFlatPtr + i * blockSize, deviceFlatPts + i * blockSize, CAMath::Min(blockSize, copySize - i * blockSize), outputStream, false);
        }
      } else {
        GPUMemCpy(myStep, hostFlatPtr, deviceFlatPts, copySize, outputStream, false);
      }
    }
  } else {
    int8_t direction = 0;
    if (gatherMode == 0) {
      P = &CompressorShadow.mPtrs;
    } else if (gatherMode == 1) {
      P = &Compressor.mPtrs;
      direction = -1;
      gatherTimer = &getTimer<GPUTPCCompressionKernels>("GPUTPCCompression_GatherOnCPU", 0);
      gatherTimer->Start();
    }
    GPUMemCpyAlways(myStep, O->nSliceRowClusters, P->nSliceRowClusters, NSECTORS * GPUTPCGeometry::NROWS * sizeof(O->nSliceRowClusters[0]), outputStream, direction);
    GPUMemCpyAlways(myStep, O->nTrackClusters, P->nTrackClusters, O->nTracks * sizeof(O->nTrackClusters[0]), outputStream, direction);
    SynchronizeStream(outputStream);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUTPCGeometry::NROWS; j++) {
        uint32_t srcOffset = mIOPtrs.clustersNative->clusterOffset[i][j] * Compressor.mMaxClusterFactorBase1024 / 1024;
        GPUMemCpyAlways(myStep, O->qTotU + offset, P->qTotU + srcOffset, O->nSliceRowClusters[i * GPUTPCGeometry::NROWS + j] * sizeof(O->qTotU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->qMaxU + offset, P->qMaxU + srcOffset, O->nSliceRowClusters[i * GPUTPCGeometry::NROWS + j] * sizeof(O->qMaxU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->flagsU + offset, P->flagsU + srcOffset, O->nSliceRowClusters[i * GPUTPCGeometry::NROWS + j] * sizeof(O->flagsU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->padDiffU + offset, P->padDiffU + srcOffset, O->nSliceRowClusters[i * GPUTPCGeometry::NROWS + j] * sizeof(O->padDiffU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->timeDiffU + offset, P->timeDiffU + srcOffset, O->nSliceRowClusters[i * GPUTPCGeometry::NROWS + j] * sizeof(O->timeDiffU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->sigmaPadU + offset, P->sigmaPadU + srcOffset, O->nSliceRowClusters[i * GPUTPCGeometry::NROWS + j] * sizeof(O->sigmaPadU[0]), outputStream, direction);
        GPUMemCpyAlways(myStep, O->sigmaTimeU + offset, P->sigmaTimeU + srcOffset, O->nSliceRowClusters[i * GPUTPCGeometry::NROWS + j] * sizeof(O->sigmaTimeU[0]), outputStream, direction);
        offset += O->nSliceRowClusters[i * GPUTPCGeometry::NROWS + j];
      }
    }
    offset = 0;
    for (uint32_t i = 0; i < O->nTracks; i++) {
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
  if (gatherMode == 1) {
    gatherTimer->Stop();
  }
  mIOPtrs.tpcCompressedClusters = Compressor.mOutputFlat;
  if (gatherMode == 3) {
    SynchronizeEventAndRelease(mEvents->stream[outputStream]);
    mRec->ReturnVolatileDeviceMemory();
  }

  if (mPipelineFinalizationCtx == nullptr) {
    SynchronizeStream(outputStream);
  } else {
    ((GPUChainTracking*)GetNextChainInQueue())->mRec->BlockStackedMemory(mRec);
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCCompression, qStr2Tag("TPCCOMPR"));
  if (GetProcessingSettings().deterministicGPUReconstruction) {
    SynchronizeGPU();
    DebugSortCompressedClusters(Compressor.mOutputFlat);
  }
  DoDebugAndDump(RecoStep::TPCCompression, GPUChainTrackingDebugFlags::TPCCompressedClusters, Compressor, &GPUTPCCompression::DumpCompressedClusters, *mDebugFile);
  return 0;
}

int32_t GPUChainTracking::RunTPCDecompression()
{
  const bool needFullFiltering = GetProcessingSettings().tpcApplyCFCutsAtDecoding || (GetProcessingSettings().tpcApplyClusterFilterOnCPU > 0);
  const bool runTimeBinCutFiltering = param().tpcCutTimeBin > 0;
  if (needFullFiltering && !GetProcessingSettings().tpcUseOldCPUDecoding) {
    GPUFatal("tpcApplyCFCutsAtDecoding, tpcApplyClusterFilterOnCPU and tpcCutTimeBin currently require tpcUseOldCPUDecoding");
  }

  CompressedClusters cmprClsHost = *mIOPtrs.tpcCompressedClusters;
  const bool useTemporaryBz = cmprClsHost.nTracks && cmprClsHost.solenoidBz != -1e6f && cmprClsHost.solenoidBz != param().bzkG && !GetProcessingSettings().doublePipeline;
  std::unique_ptr<GPUParam> tmpParam;
  int32_t inputStream = 0;

  if (useTemporaryBz) {
    tmpParam = std::make_unique<GPUParam>(param());
    SynchronizeGPU();
    param().UpdateBzOnly(cmprClsHost.solenoidBz, mRec->GetGRPSettings().constBz);
    WriteConstantParams(inputStream);
  }
  if (GetProcessingSettings().tpcUseOldCPUDecoding) {
    const bool runFiltering = needFullFiltering || runTimeBinCutFiltering;
    const auto& threadContext = GetThreadContext();
    TPCClusterDecompressor decomp;
    auto allocatorFinal = [this](size_t size) {
      this->mInputsHost->mNClusterNative = this->mInputsShadow->mNClusterNative = size;
      this->AllocateRegisteredMemory(this->mInputsHost->mResourceClusterNativeOutput, this->mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
      return this->mInputsHost->mPclusterNativeOutput;
    };
    std::unique_ptr<ClusterNative[]> tmpBuffer;
    auto allocatorTmp = [&tmpBuffer](size_t size) {
      return ((tmpBuffer = std::make_unique<ClusterNative[]>(size))).get();
    };
    auto& decompressTimer = getTimer<TPCClusterDecompressor>("TPCDecompression", 0);
    auto allocatorUse = runFiltering ? std::function<ClusterNative*(size_t)>{allocatorTmp} : std::function<ClusterNative*(size_t)>{allocatorFinal};
    decompressTimer.Start();
    if (decomp.decompress(mIOPtrs.tpcCompressedClusters, *mClusterNativeAccess, allocatorUse, param(), GetProcessingSettings().deterministicGPUReconstruction)) {
      GPUError("Error decompressing clusters");
      return 1;
    }
    if (runFiltering) {
      RunTPCClusterFilter(mClusterNativeAccess.get(), allocatorFinal, GetProcessingSettings().tpcApplyCFCutsAtDecoding);
    }
    decompressTimer.Stop();
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
  } else {
    mRec->PushNonPersistentMemory(qStr2Tag("TPCDCMPR"));
    RecoStep myStep = RecoStep::TPCDecompression;
    bool doGPU = GetRecoStepsGPU() & RecoStep::TPCDecompression;
    GPUTPCDecompression& Decompressor = processors()->tpcDecompressor;
    GPUTPCDecompression& DecompressorShadow = doGPU ? processorsShadow()->tpcDecompressor : Decompressor;
    const auto& threadContext = GetThreadContext();
    CompressedClusters& inputGPU = Decompressor.mInputGPU;
    CompressedClusters& inputGPUShadow = DecompressorShadow.mInputGPU;

    if (cmprClsHost.nTracks && cmprClsHost.solenoidBz != -1e6f && cmprClsHost.solenoidBz != param().bzkG) {
      throw std::runtime_error("Configured solenoid Bz " + std::to_string(param().bzkG) + " does not match value used for track model encoding " + std::to_string(cmprClsHost.solenoidBz));
    }
    if (cmprClsHost.nTracks && cmprClsHost.maxTimeBin != -1e6 && cmprClsHost.maxTimeBin != param().continuousMaxTimeBin) {
      throw std::runtime_error("Configured max time bin " + std::to_string(param().continuousMaxTimeBin) + " does not match value used for track model encoding " + std::to_string(cmprClsHost.maxTimeBin));
    }

    int32_t unattachedStream = mRec->NStreams() - 1;
    inputGPU = cmprClsHost;
    SetupGPUProcessor(&Decompressor, true);
    WriteToConstantMemory(myStep, (char*)&processors()->tpcDecompressor - (char*)processors(), &DecompressorShadow, sizeof(DecompressorShadow), inputStream);
    inputGPU = cmprClsHost;

    bool toGPU = true;
    runKernel<GPUMemClean16>({GetGridAutoStep(inputStream, RecoStep::TPCDecompression), krnlRunRangeNone, &mEvents->init}, DecompressorShadow.mNativeClustersIndex, NSECTORS * GPUTPCGeometry::NROWS * sizeof(DecompressorShadow.mNativeClustersIndex[0]));
    int32_t nStreams = doGPU ? mRec->NStreams() - 1 : 1;
    if (cmprClsHost.nAttachedClusters != 0) {
      std::exclusive_scan(cmprClsHost.nTrackClusters, cmprClsHost.nTrackClusters + cmprClsHost.nTracks, Decompressor.mAttachedClustersOffsets, 0u); // computing clusters offsets for first kernel
      for (int32_t iStream = 0; iStream < nStreams; iStream++) {
        uint32_t startTrack = cmprClsHost.nTracks / nStreams * iStream;
        uint32_t endTrack = cmprClsHost.nTracks / nStreams * (iStream + 1) + (iStream < nStreams - 1 ? 0 : cmprClsHost.nTracks % nStreams); // index of last track (excluded from computation)
        uint32_t numTracks = endTrack - startTrack;
        uint32_t* offsets = Decompressor.mAttachedClustersOffsets;
        uint32_t numClusters = (endTrack == cmprClsHost.nTracks ? offsets[endTrack - 1] + cmprClsHost.nTrackClusters[endTrack - 1] : offsets[endTrack]) - offsets[startTrack];
        uint32_t numClustersRed = numClusters - numTracks;
        GPUMemCpy(myStep, DecompressorShadow.mAttachedClustersOffsets + startTrack, Decompressor.mAttachedClustersOffsets + startTrack, numTracks * sizeof(Decompressor.mAttachedClustersOffsets[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.nTrackClusters + startTrack, cmprClsHost.nTrackClusters + startTrack, numTracks * sizeof(cmprClsHost.nTrackClusters[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.qTotA + offsets[startTrack], cmprClsHost.qTotA + offsets[startTrack], numClusters * sizeof(cmprClsHost.qTotA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.qMaxA + offsets[startTrack], cmprClsHost.qMaxA + offsets[startTrack], numClusters * sizeof(cmprClsHost.qMaxA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.flagsA + offsets[startTrack], cmprClsHost.flagsA + offsets[startTrack], numClusters * sizeof(cmprClsHost.flagsA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.rowDiffA + offsets[startTrack] - startTrack, cmprClsHost.rowDiffA + offsets[startTrack] - startTrack, numClustersRed * sizeof(cmprClsHost.rowDiffA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.sliceLegDiffA + offsets[startTrack] - startTrack, cmprClsHost.sliceLegDiffA + offsets[startTrack] - startTrack, numClustersRed * sizeof(cmprClsHost.sliceLegDiffA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.padResA + offsets[startTrack] - startTrack, cmprClsHost.padResA + offsets[startTrack] - startTrack, numClustersRed * sizeof(cmprClsHost.padResA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.timeResA + offsets[startTrack] - startTrack, cmprClsHost.timeResA + offsets[startTrack] - startTrack, numClustersRed * sizeof(cmprClsHost.timeResA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.sigmaPadA + offsets[startTrack], cmprClsHost.sigmaPadA + offsets[startTrack], numClusters * sizeof(cmprClsHost.sigmaPadA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.sigmaTimeA + offsets[startTrack], cmprClsHost.sigmaTimeA + offsets[startTrack], numClusters * sizeof(cmprClsHost.sigmaTimeA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.qPtA + startTrack, cmprClsHost.qPtA + startTrack, numTracks * sizeof(cmprClsHost.qPtA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.rowA + startTrack, cmprClsHost.rowA + startTrack, numTracks * sizeof(cmprClsHost.rowA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.sliceA + startTrack, cmprClsHost.sliceA + startTrack, numTracks * sizeof(cmprClsHost.sliceA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.timeA + startTrack, cmprClsHost.timeA + startTrack, numTracks * sizeof(cmprClsHost.timeA[0]), iStream, toGPU);
        GPUMemCpy(myStep, inputGPUShadow.padA + startTrack, cmprClsHost.padA + startTrack, numTracks * sizeof(cmprClsHost.padA[0]), iStream, toGPU);
        runKernel<GPUTPCDecompressionKernels, GPUTPCDecompressionKernels::step0attached>({GetGridAuto(iStream), krnlRunRangeNone, {&mEvents->stream[iStream], &mEvents->init}}, startTrack, endTrack);
      }
    }
    GPUMemCpy(myStep, inputGPUShadow.nSliceRowClusters, cmprClsHost.nSliceRowClusters, NSECTORS * GPUTPCGeometry::NROWS * sizeof(cmprClsHost.nSliceRowClusters[0]), unattachedStream, toGPU);
    GPUMemCpy(myStep, inputGPUShadow.qTotU, cmprClsHost.qTotU, cmprClsHost.nUnattachedClusters * sizeof(cmprClsHost.qTotU[0]), unattachedStream, toGPU);
    GPUMemCpy(myStep, inputGPUShadow.qMaxU, cmprClsHost.qMaxU, cmprClsHost.nUnattachedClusters * sizeof(cmprClsHost.qMaxU[0]), unattachedStream, toGPU);
    GPUMemCpy(myStep, inputGPUShadow.flagsU, cmprClsHost.flagsU, cmprClsHost.nUnattachedClusters * sizeof(cmprClsHost.flagsU[0]), unattachedStream, toGPU);
    GPUMemCpy(myStep, inputGPUShadow.padDiffU, cmprClsHost.padDiffU, cmprClsHost.nUnattachedClusters * sizeof(cmprClsHost.padDiffU[0]), unattachedStream, toGPU);
    GPUMemCpy(myStep, inputGPUShadow.timeDiffU, cmprClsHost.timeDiffU, cmprClsHost.nUnattachedClusters * sizeof(cmprClsHost.timeDiffU[0]), unattachedStream, toGPU);
    GPUMemCpy(myStep, inputGPUShadow.sigmaPadU, cmprClsHost.sigmaPadU, cmprClsHost.nUnattachedClusters * sizeof(cmprClsHost.sigmaPadU[0]), unattachedStream, toGPU);
    GPUMemCpy(myStep, inputGPUShadow.sigmaTimeU, cmprClsHost.sigmaTimeU, cmprClsHost.nUnattachedClusters * sizeof(cmprClsHost.sigmaTimeU[0]), unattachedStream, toGPU);

    TransferMemoryResourceLinkToHost(RecoStep::TPCDecompression, Decompressor.mResourceTmpIndexes, inputStream, nullptr, mEvents->stream, nStreams);
    SynchronizeStream(inputStream);
    uint32_t offset = 0;
    uint32_t decodedAttachedClusters = 0;
    for (uint32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUTPCGeometry::NROWS; j++) {
        uint32_t linearIndex = i * GPUTPCGeometry::NROWS + j;
        uint32_t unattachedOffset = (linearIndex >= cmprClsHost.nSliceRows) ? 0 : cmprClsHost.nSliceRowClusters[linearIndex];
        (mClusterNativeAccess->nClusters)[i][j] = Decompressor.mNativeClustersIndex[linearIndex] + unattachedOffset;
        Decompressor.mUnattachedClustersOffsets[linearIndex] = offset;
        offset += unattachedOffset;
        decodedAttachedClusters += Decompressor.mNativeClustersIndex[linearIndex];
      }
    }
    TransferMemoryResourceLinkToGPU(RecoStep::TPCDecompression, Decompressor.mResourceTmpClustersOffsets, inputStream);
    if (decodedAttachedClusters != cmprClsHost.nAttachedClusters) {
      GPUWarning("%u / %u clusters failed track model decoding (%f %%)", cmprClsHost.nAttachedClusters - decodedAttachedClusters, cmprClsHost.nAttachedClusters, 100.f * (float)(cmprClsHost.nAttachedClusters - decodedAttachedClusters) / (float)cmprClsHost.nAttachedClusters);
    }
    if (runTimeBinCutFiltering) { // If filtering, allocate a temporary buffer and cluster native access in decompressor context
      Decompressor.mNClusterNativeBeforeFiltering = DecompressorShadow.mNClusterNativeBeforeFiltering = decodedAttachedClusters + cmprClsHost.nUnattachedClusters;
      AllocateRegisteredMemory(Decompressor.mResourceTmpBufferBeforeFiltering);
      AllocateRegisteredMemory(Decompressor.mResourceClusterNativeAccess);
      mClusterNativeAccess->clustersLinear = DecompressorShadow.mNativeClustersBuffer;
      mClusterNativeAccess->setOffsetPtrs();
      *Decompressor.mClusterNativeAccess = *mClusterNativeAccess;
      WriteToConstantMemory(myStep, (char*)&processors()->tpcDecompressor - (char*)processors(), &DecompressorShadow, sizeof(DecompressorShadow), inputStream);
      TransferMemoryResourceLinkToGPU(RecoStep::TPCDecompression, Decompressor.mResourceClusterNativeAccess, inputStream, &mEvents->single);
    } else { // If not filtering, directly allocate the final buffers
      mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = cmprClsHost.nAttachedClusters + cmprClsHost.nUnattachedClusters;
      AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
      AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
      DecompressorShadow.mNativeClustersBuffer = mInputsShadow->mPclusterNativeBuffer;
      Decompressor.mNativeClustersBuffer = mInputsHost->mPclusterNativeOutput;
      DecompressorShadow.mClusterNativeAccess = mInputsShadow->mPclusterNativeAccess;
      Decompressor.mClusterNativeAccess = mInputsHost->mPclusterNativeAccess;
      WriteToConstantMemory(myStep, (char*)&processors()->tpcDecompressor - (char*)processors(), &DecompressorShadow, sizeof(DecompressorShadow), inputStream);
      if (doGPU) {
        mClusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
        mClusterNativeAccess->setOffsetPtrs();
        *mInputsHost->mPclusterNativeAccess = *mClusterNativeAccess;
        processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
        WriteToConstantMemory(RecoStep::TPCDecompression, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), inputStream);
        TransferMemoryResourceLinkToGPU(RecoStep::TPCDecompression, mInputsHost->mResourceClusterNativeAccess, inputStream, &mEvents->single);
      }
      mIOPtrs.clustersNative = mClusterNativeAccess.get();
      mClusterNativeAccess->clustersLinear = mInputsHost->mPclusterNativeOutput;
      mClusterNativeAccess->setOffsetPtrs();
      *mInputsHost->mPclusterNativeAccess = *mClusterNativeAccess;
    }

    uint32_t batchSize = doGPU ? 6 : NSECTORS;
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector = iSector + batchSize) {
      int32_t iStream = (iSector / batchSize) % mRec->NStreams();
      runKernel<GPUTPCDecompressionKernels, GPUTPCDecompressionKernels::step1unattached>({GetGridAuto(iStream), krnlRunRangeNone, {nullptr, &mEvents->single}}, iSector, batchSize);
      uint32_t copySize = std::accumulate(mClusterNativeAccess->nClustersSector + iSector, mClusterNativeAccess->nClustersSector + iSector + batchSize, 0u);
      if (!runTimeBinCutFiltering) {
        GPUMemCpy(RecoStep::TPCDecompression, mInputsHost->mPclusterNativeOutput + mClusterNativeAccess->clusterOffset[iSector][0], DecompressorShadow.mNativeClustersBuffer + mClusterNativeAccess->clusterOffset[iSector][0], sizeof(Decompressor.mNativeClustersBuffer[0]) * copySize, iStream, false);
      }
    }
    SynchronizeGPU();

    if (runTimeBinCutFiltering) { // If filtering is applied, count how many clusters will remain after filtering and allocate final buffers accordingly
      AllocateRegisteredMemory(Decompressor.mResourceNClusterPerSectorRow);
      WriteToConstantMemory(myStep, (char*)&processors()->tpcDecompressor - (char*)processors(), &DecompressorShadow, sizeof(DecompressorShadow), unattachedStream);
      runKernel<GPUMemClean16>({GetGridAutoStep(unattachedStream, RecoStep::TPCDecompression), krnlRunRangeNone}, DecompressorShadow.mNClusterPerSectorRow, NSECTORS * GPUTPCGeometry::NROWS * sizeof(DecompressorShadow.mNClusterPerSectorRow[0]));
      runKernel<GPUTPCDecompressionUtilKernels, GPUTPCDecompressionUtilKernels::countFilteredClusters>(GetGridAutoStep(unattachedStream, RecoStep::TPCDecompression));
      TransferMemoryResourceLinkToHost(RecoStep::TPCDecompression, Decompressor.mResourceNClusterPerSectorRow, unattachedStream);
      SynchronizeStream(unattachedStream);
      uint32_t nClustersFinal = std::accumulate(Decompressor.mNClusterPerSectorRow, Decompressor.mNClusterPerSectorRow + inputGPU.nSliceRows, 0u);
      mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = nClustersFinal;
      AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
      AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
      DecompressorShadow.mNativeClustersBuffer = mInputsShadow->mPclusterNativeBuffer;
      Decompressor.mNativeClustersBuffer = mInputsHost->mPclusterNativeOutput;
      WriteToConstantMemory(myStep, (char*)&processors()->tpcDecompressor - (char*)processors(), &DecompressorShadow, sizeof(DecompressorShadow), unattachedStream);
      for (uint32_t i = 0; i < NSECTORS; i++) {
        for (uint32_t j = 0; j < GPUTPCGeometry::NROWS; j++) {
          mClusterNativeAccess->nClusters[i][j] = Decompressor.mNClusterPerSectorRow[i * GPUTPCGeometry::NROWS + j];
        }
      }
      if (doGPU) {
        mClusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
        mClusterNativeAccess->setOffsetPtrs();
        *mInputsHost->mPclusterNativeAccess = *mClusterNativeAccess;
        processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
        WriteToConstantMemory(RecoStep::TPCDecompression, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), unattachedStream);
        TransferMemoryResourceLinkToGPU(RecoStep::TPCDecompression, mInputsHost->mResourceClusterNativeAccess, unattachedStream);
      }
      mIOPtrs.clustersNative = mClusterNativeAccess.get();
      mClusterNativeAccess->clustersLinear = mInputsHost->mPclusterNativeOutput;
      mClusterNativeAccess->setOffsetPtrs();
      runKernel<GPUTPCDecompressionUtilKernels, GPUTPCDecompressionUtilKernels::storeFilteredClusters>(GetGridAutoStep(unattachedStream, RecoStep::TPCDecompression));
      GPUMemCpy(RecoStep::TPCDecompression, mInputsHost->mPclusterNativeOutput, DecompressorShadow.mNativeClustersBuffer, sizeof(Decompressor.mNativeClustersBuffer[0]) * nClustersFinal, unattachedStream, false);
      SynchronizeStream(unattachedStream);
    }
    if (GetProcessingSettings().deterministicGPUReconstruction || GetProcessingSettings().debugLevel >= 4) {
      runKernel<GPUTPCDecompressionUtilKernels, GPUTPCDecompressionUtilKernels::sortPerSectorRow>(GetGridAutoStep(unattachedStream, RecoStep::TPCDecompression));
      const ClusterNativeAccess* decoded = mIOPtrs.clustersNative;
      if (doGPU) {
        for (uint32_t i = 0; i < NSECTORS; i++) {
          for (uint32_t j = 0; j < GPUTPCGeometry::NROWS; j++) {
            ClusterNative* begin = mInputsHost->mPclusterNativeOutput + decoded->clusterOffset[i][j];
            ClusterNative* end = begin + decoded->nClusters[i][j];
            std::sort(begin, end);
          }
        }
      }
      SynchronizeStream(unattachedStream);
    }
    mRec->PopNonPersistentMemory(RecoStep::TPCDecompression, qStr2Tag("TPCDCMPR"));
  }
  if (useTemporaryBz) {
    SynchronizeGPU();
    param() = *tmpParam;
    tmpParam.reset();
    WriteConstantParams();
  }
  DoDebugDump(GPUChainTrackingDebugFlags::TPCDecompressedClusters, &GPUChainTracking::DumpClusters, *mDebugFile, mIOPtrs.clustersNative);
  return 0;
}

void GPUChainTracking::WriteReducedClusters()
{
  GPUTPCCompression& Compressor = processors()->tpcCompressor;
  mClusterNativeAccessReduced = std::make_unique<ClusterNativeAccess>();
  uint32_t nOutput = 0;
  for (uint32_t iSec = 0; iSec < GPUTPCGeometry::NSECTORS; iSec++) {
    for (uint32_t iRow = 0; iRow < GPUTPCGeometry::NROWS; iRow++) {
      mClusterNativeAccessReduced->nClusters[iSec][iRow] = 0;
      for (uint32_t i = 0; i < mIOPtrs.clustersNative->nClusters[iSec][iRow]; i++) {
        mClusterNativeAccessReduced->nClusters[iSec][iRow] += !Compressor.rejectCluster(mIOPtrs.clustersNative->clusterOffset[iSec][iRow] + i, param(), mIOPtrs);
      }
      nOutput += mClusterNativeAccessReduced->nClusters[iSec][iRow];
    }
  }

  GPUOutputControl* clOutput = mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)];
  if (!clOutput || !clOutput->allocator) {
    throw std::runtime_error("No output allocator for clusterNative available");
  }
  auto* clBuffer = (ClusterNative*)clOutput->allocator(nOutput * sizeof(ClusterNative));
  mClusterNativeAccessReduced->clustersLinear = clBuffer;
  mClusterNativeAccessReduced->setOffsetPtrs();

  std::pair<o2::dataformats::ConstMCLabelContainer*, o2::dataformats::ConstMCLabelContainerView*> labelBuffer;
  if (mIOPtrs.clustersNative->clustersMCTruth) {
    GPUOutputControl* labelOutput = mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)];
    if (!labelOutput || !labelOutput->allocator) {
      throw std::runtime_error("No output allocator for clusterNative labels available");
    }
    ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer* labelContainer = reinterpret_cast<ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer*>(labelOutput->allocator(0));
    labelBuffer = {&labelContainer->first, &labelContainer->second};
  }

  nOutput = 0;
  o2::dataformats::MCLabelContainer tmpContainer;
  for (uint32_t i = 0; i < mIOPtrs.clustersNative->nClustersTotal; i++) {
    if (!Compressor.rejectCluster(i, param(), mIOPtrs)) {
      if (mIOPtrs.clustersNative->clustersMCTruth) {
        for (const auto& element : mIOPtrs.clustersNative->clustersMCTruth->getLabels(i)) {
          tmpContainer.addElement(nOutput, element);
        }
      }
      clBuffer[nOutput++] = mIOPtrs.clustersNative->clustersLinear[i];
    }
  }
  mIOPtrs.clustersNativeReduced = mClusterNativeAccessReduced.get();
  if (mIOPtrs.clustersNative->clustersMCTruth) {
    tmpContainer.flatten_to(*labelBuffer.first);
    *labelBuffer.second = *labelBuffer.first;
    mClusterNativeAccessReduced->clustersMCTruth = labelBuffer.second;
  }
}
