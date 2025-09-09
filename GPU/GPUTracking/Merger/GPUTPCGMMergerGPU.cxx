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

/// \file GPUTPCGMMergerGPU.cxx
/// \author David Rohr

#include "GPUTPCGMMergerGPU.h"
#include "GPUCommonAlgorithm.h"
#include "GPUReconstructionThreading.h"

using namespace o2::gpu;

template <>
GPUdii() void GPUTPCGMMergerTrackFit::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t mode)
{
  const int32_t iEnd = mode == -1 ? merger.Memory()->nRetryRefit : merger.NMergedTracks();
  GPUCA_TBB_KERNEL_LOOP(merger.GetRec(), int32_t, ii, iEnd, {
    const int32_t i = mode == -1 ? merger.RetryRefitIds()[ii] : mode ? merger.TrackOrderProcess()[ii] : ii;
    GPUTPCGMTrackParam::RefitTrack(merger.MergedTracks()[i], i, &merger, mode == -1);
  });
}

template <>
GPUdii() void GPUTPCGMMergerFollowLoopers::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  GPUCA_TBB_KERNEL_LOOP(merger.GetRec(), uint32_t, i, merger.Memory()->nLoopData, {
    GPUTPCGMTrackParam::RefitLoop(&merger, i);
  });
}

template <>
GPUdii() void GPUTPCGMMergerUnpackResetIds::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t iSector)
{
  merger.UnpackResetIds(nBlocks, nThreads, iBlock, iThread, iSector);
}

template <>
GPUdii() void GPUTPCGMMergerSectorRefit::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t iSector)
{
  merger.RefitSectorTracks(nBlocks, nThreads, iBlock, iThread, iSector);
}

template <>
GPUdii() void GPUTPCGMMergerUnpackGlobal::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t iSector)
{
  merger.UnpackSectorGlobal(nBlocks, nThreads, iBlock, iThread, iSector);
}

template <>
GPUdii() void GPUTPCGMMergerUnpackSaveNumber::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t id)
{
  if (iThread == 0 && iBlock == 0) {
    merger.UnpackSaveNumber(id);
  }
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.ResolveFindConnectedComponentsSetup(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.ResolveFindConnectedComponentsHookLinks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.ResolveFindConnectedComponentsHookNeighbors(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<3>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.ResolveFindConnectedComponentsMultiJump(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<4>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t useOrigTrackParam, int8_t mergeAll)
{
  merger.ResolveMergeSectors(smem, nBlocks, nThreads, iBlock, iThread, useOrigTrackParam, mergeAll);
}

template <>
GPUdii() void GPUTPCGMMergerClearLinks::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t output)
{
  merger.ClearTrackLinks(nBlocks, nThreads, iBlock, iThread, output);
}

template <>
GPUdii() void GPUTPCGMMergerMergeWithinPrepare::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.MergeWithinSectorsPrepare(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeSectorsPrepare::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t border0, int32_t border1, int8_t useOrigTrackParam)
{
  merger.MergeSectorsPrepare(nBlocks, nThreads, iBlock, iThread, border0, border1, useOrigTrackParam);
}

template <int32_t I, typename... Args>
GPUdii() void GPUTPCGMMergerMergeBorders::Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, Args... args)
{
  merger.MergeBorderTracks<I>(nBlocks, nThreads, iBlock, iThread, args...);
}
#if !defined(GPUCA_GPUCODE) || defined(GPUCA_GPUCODE_DEVICE) // FIXME: DR: WORKAROUND to avoid CUDA bug creating host symbols for device code.
template GPUdni() void GPUTPCGMMergerMergeBorders::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t iSector, int8_t withinSector, int8_t mergeMode);
template GPUdni() void GPUTPCGMMergerMergeBorders::Thread<2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t iSector, int8_t withinSector, int8_t mergeMode);
template GPUdni() void GPUTPCGMMergerMergeBorders::Thread<3>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, gputpcgmmergertypes::GPUTPCGMBorderRange* range, int32_t N, int32_t cmpMax);
#endif
template <>
GPUdii() void GPUTPCGMMergerMergeBorders::Thread<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int32_t iSector, int8_t withinSector, int8_t mergeMode)
{
  merger.MergeBorderTracks<1>(2, nThreads, iBlock & 1, iThread, iBlock / 2, withinSector, mergeMode);
}

template <>
GPUdii() void GPUTPCGMMergerMergeCE::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.MergeCE(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerLinkExtrapolatedTracks::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.LinkExtrapolatedTracks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerCollect::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.CollectMergedTracks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerSortTracks::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.SortTracks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerSortTracksQPt::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.SortTracksQPt(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerSortTracksPrepare::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.SortTracksPrepare(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerPrepareForFit::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.PrepareForFit0(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerPrepareForFit::Thread<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.PrepareForFit1(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerPrepareForFit::Thread<2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.PrepareForFit2(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerFinalize::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.Finalize0(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerFinalize::Thread<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.Finalize1(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerFinalize::Thread<2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.Finalize2(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeLoopers::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.MergeLoopersInit(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeLoopers::Thread<1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.MergeLoopersSort(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeLoopers::Thread<2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger)
{
  merger.MergeLoopersMain(nBlocks, nThreads, iBlock, iThread);
}
