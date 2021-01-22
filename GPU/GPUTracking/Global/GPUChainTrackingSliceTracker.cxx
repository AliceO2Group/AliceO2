// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTrackingSliceTracker.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTPCClusterData.h"
#include <fstream>

using namespace GPUCA_NAMESPACE::gpu;

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
  if (!param().par.earlyTpcTransform) {
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
  GPUCA_OPENMP(parallel for if(!doGPU && GetProcessingSettings().ompKernels != 1) num_threads(mRec->SetAndGetNestedLoopOmpFactor(!doGPU, NSLICES)))
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
    if (GetProcessingSettings().keepDisplayMemory && !doSliceDataOnGPU) {
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
  mRec->SetNestedLoopOmpFactor(1);
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
    GPUCA_OPENMP(parallel for if(!doGPU && GetProcessingSettings().ompKernels != 1) num_threads(mRec->SetAndGetNestedLoopOmpFactor(!doGPU, NSLICES)))
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (param().rec.GlobalTracking) {
        GlobalTracking(iSlice, 0);
      }
      if (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) {
        WriteOutput(iSlice, 0);
      }
    }
    mRec->SetNestedLoopOmpFactor(1);
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
