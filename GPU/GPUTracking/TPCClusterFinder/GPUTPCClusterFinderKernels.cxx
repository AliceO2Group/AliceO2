// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterFinderKernels.cxx
/// \author David Rohr

#include "GPUTPCClusterFinderKernels.h"
#include "GPUConstantMem.h"
#include "GPUO2DataTypes.h"
#include "GPUCommonAlgorithm.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"
#include "CfConsts.h"
#include "CfUtils.h"
#include "ChargeMapFiller.h"
#include "PeakFinder.h"
#include "ClusterAccumulator.h"
#include "Clusterizer.h"
#include "NoiseSuppression.h"
#include "Deconvolution.h"
#include "StreamCompaction.h"


using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::fillChargeMap>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  ChargeMapFiller::fillChargeMapImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPdigits, clusterer.mPchargeMap, clusterer.mPmemory->nDigits);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::resetMaps>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  ChargeMapFiller::resetMapsImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPdigits, clusterer.mPchargeMap, clusterer.mPpeakMap);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::findPeaks>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  PeakFinder::findPeaksImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPchargeMap, clusterer.mPdigits, clusterer.mPmemory->nDigits, clusterer.mPisPeak, clusterer.mPpeakMap);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::noiseSuppression>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  NoiseSuppression::noiseSuppressionImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPchargeMap, clusterer.mPpeakMap, clusterer.mPpeaks, clusterer.mPmemory->nPeaks, clusterer.mPisPeak);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::updatePeaks>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  NoiseSuppression::updatePeaksImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPpeaks, clusterer.mPisPeak, clusterer.mPpeakMap);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::countPeaks>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  Deconvolution::countPeaksImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPpeakMap, clusterer.mPchargeMap, clusterer.mPdigits, clusterer.mPmemory->nDigits);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::computeClusters>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  Clusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPchargeMap, clusterer.mPfilteredPeaks, clusterer.mPmemory->nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterer.mPclusterByRow);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::nativeScanUpStart>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf, int stage)
{
  int nElems = compactionElems(clusterer, stage);
  StreamCompaction::nativeScanUpStartImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPisPeak, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::nativeScanUp>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf, int nElems)
{
  StreamCompaction::nativeScanUpImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::nativeScanTop>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf, int nElems)
{
  StreamCompaction::nativeScanTopImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, nElems);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::nativeScanDown>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf, unsigned int offset, int nElems)
{
  StreamCompaction::nativeScanDownImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, offset, nElems);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::compactDigit>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf, int stage, GPUglobalref() deprecated::PackedDigit* in, GPUglobalref() deprecated::PackedDigit* out)
{
  unsigned int nElems = compactionElems(clusterer, stage);
  StreamCompaction::compactDigitImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, in, out, clusterer.mPisPeak, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems);
  unsigned int lastId = get_global_size(0) - 1;
  if ((unsigned int)get_global_id(0) == lastId) {
    if (stage) {
      clusterer.mPmemory->nClusters = clusterer.mPbuf[lastId];
    } else {
      clusterer.mPmemory->nPeaks = clusterer.mPbuf[lastId];
    }
  }
}

GPUd() int GPUTPCClusterFinderKernels::compactionElems(processorType& clusterer, int stage)
{
  return (stage) ? clusterer.mPmemory->nPeaks : clusterer.mPmemory->nDigits;
}
