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
#if defined(WITH_OPENMP) && !defined(GPUCA_GPUCODE)
#include "GPUReconstruction.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCGMMergerTrackFit::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger, int mode)
{
  const int iStart = mode <= 0 ? 0 : merger.NSlowTracks();
  const int iEnd = mode == -2 ? merger.Memory()->nRetryRefit : mode >= 0 ? merger.NOutputTracks() : merger.NSlowTracks();
#if defined(WITH_OPENMP) && !defined(GPUCA_GPUCODE)
#pragma omp parallel for num_threads(merger.GetRec().GetDeviceProcessingSettings().nThreads)
#endif
  for (int ii = iStart + get_global_id(0); ii < iEnd; ii += get_global_size(0)) {
    const int i = mode == -2 ? merger.RetryRefitIds()[ii] : mode ? merger.TrackOrderProcess()[ii] : ii;
    GPUTPCGMTrackParam::RefitTrack(merger.OutputTracks()[i], i, &merger, merger.Clusters(), mode == -2);
  }
}

template <>
GPUdii() void GPUTPCGMMergerFollowLoopers::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
#if defined(WITH_OPENMP) && !defined(GPUCA_GPUCODE)
#pragma omp parallel for num_threads(merger.GetRec().GetDeviceProcessingSettings().nThreads)
#endif
  for (unsigned int i = get_global_id(0); i < merger.Memory()->nLoopData; i += get_global_size(0)) {
    GPUTPCGMTrackParam::RefitLoop(&merger, i);
  }
}

template <>
GPUdii() void GPUTPCGMMergerUnpack::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.UnpackSlices(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeWithin::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.MergeWithingSlices(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeSlices::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.MergeSlices(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeCEInit::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.MergeCEInit(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerMergeCE::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.MergeCE(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerCollect::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.CollectMergedTracks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerSortTracks::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.SortTracks(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerSortTracksPrepare::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.SortTracksPrepare(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerPrepareClusters::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.PrepareClustersForFit(nBlocks, nThreads, iBlock, iThread);
}

template <>
GPUdii() void GPUTPCGMMergerFinalize::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  merger.Finalize(nBlocks, nThreads, iBlock, iThread);
}
