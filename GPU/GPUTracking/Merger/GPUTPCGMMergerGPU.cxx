// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMMergerGPU.cxx
/// \author David Rohr

#include "GPUTPCGMMergerGPU.h"
#include "GPUCommonAlgorithm.h"
#if defined(WITH_OPENMP) && !defined(GPUCA_GPUCODE)
#include "GPUReconstruction.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCGMMergerTrackFit::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int mode)
{
  const int iEnd = mode == -1 ? merger.Memory()->nRetryRefit : merger.NOutputTracks();
  GPUCA_OPENMP(parallel for if(!merger.GetRec().GetProcessingSettings().ompKernels) num_threads(merger.GetRec().GetProcessingSettings().ompThreads))
  for (int ii = get_global_id(0); ii < iEnd; ii += get_global_size(0)) {
    const int i = mode == -1 ? merger.RetryRefitIds()[ii] : mode ? merger.TrackOrderProcess()[ii] : ii;
    GPUTPCGMTrackParam::RefitTrack(merger.OutputTracks()[i], i, &merger, mode == -1);
  }
}

template <>
GPUdii() void GPUTPCGMMergerFollowLoopers::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  GPUCA_OPENMP(parallel for if(!merger.GetRec().GetProcessingSettings().ompKernels) num_threads(merger.GetRec().GetProcessingSettings().ompThreads))
  for (unsigned int i = get_global_id(0); i < merger.Memory()->nLoopData; i += get_global_size(0)) {
    GPUTPCGMTrackParam::RefitLoop(&merger, i);
  }
}

template <>
GPUdii() void GPUTPCGMMergerUnpackResetIds::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int iSlice)
{
  merger.UnpackResetIds(nBlocks, nThreads, iBlock, iThread, iSlice);
}

template <>
GPUdii() void GPUTPCGMMergerSliceRefit::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int iSlice)
{
  merger.RefitSliceTracks(nBlocks, nThreads, iBlock, iThread, iSlice);
}

template <>
GPUdii() void GPUTPCGMMergerUnpackGlobal::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int iSlice)
{
  merger.UnpackSliceGlobal(nBlocks, nThreads, iBlock, iThread, iSlice);
}

template <>
GPUdii() void GPUTPCGMMergerUnpackSaveNumber::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int id)
{
  if (iThread == 0 && iBlock == 0) {
    merger.UnpackSaveNumber(id);
  }
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.ResolveFindConnectedComponentsSetup(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.ResolveFindConnectedComponentsHookLinks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<2>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.ResolveFindConnectedComponentsHookNeighbors(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<3>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.ResolveFindConnectedComponentsMultiJump(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerResolve::Thread<4>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, char useOrigTrackParam, char mergeAll)
{
  merger.ResolveMergeSlices(smem, nBlocks, nThreads, iBlock, iThread, useOrigTrackParam, mergeAll);
}

template <>
GPUdii() void GPUTPCGMMergerClearLinks::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, char nOutput)
{
  merger.ClearTrackLinks(nBlocks, nThreads, iBlock, iThread, nOutput);
}

template <>
GPUdii() void GPUTPCGMMergerMergeWithinPrepare::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.MergeWithinSlicesPrepare(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeSlicesPrepare::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int border0, int border1, char useOrigTrackParam)
{
  merger.MergeSlicesPrepare(nBlocks, nThreads, iBlock, iThread, border0, border1, useOrigTrackParam);
}

template <int I, typename... Args>
GPUdii() void GPUTPCGMMergerMergeBorders::Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, Args... args)
{
  merger.MergeBorderTracks<I>(nBlocks, nThreads, iBlock, iThread, args...);
}
template GPUd() void GPUTPCGMMergerMergeBorders::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int iSlice, char withinSlice, char mergeMode);
template GPUd() void GPUTPCGMMergerMergeBorders::Thread<2>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int iSlice, char withinSlice, char mergeMode);
template GPUd() void GPUTPCGMMergerMergeBorders::Thread<3>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, gputpcgmmergertypes::GPUTPCGMBorderRange* range, int N, int cmpMax);
template <>
GPUdii() void GPUTPCGMMergerMergeBorders::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int iSlice, char withinSlice, char mergeMode)
{
  merger.MergeBorderTracks<1>(2, nThreads, iBlock & 1, iThread, iBlock / 2, withinSlice, mergeMode);
}

template <>
GPUdii() void GPUTPCGMMergerMergeCE::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.MergeCE(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerLinkGlobalTracks::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.LinkGlobalTracks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerCollect::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.CollectMergedTracks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerSortTracks::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.SortTracks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerSortTracksQPt::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.SortTracksQPt(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerSortTracksPrepare::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.SortTracksPrepare(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerPrepareClusters::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.PrepareClustersForFit0(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerPrepareClusters::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.PrepareClustersForFit1(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerPrepareClusters::Thread<2>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.PrepareClustersForFit2(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerFinalize::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.Finalize0(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerFinalize::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.Finalize1(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerFinalize::Thread<2>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.Finalize2(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeLoopers::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  merger.MergeLoopers(nBlocks, nThreads, iBlock, iThread);
}
