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

using namespace GPUCA_NAMESPACE::gpu;

#include "cl/streamCompaction.cl"
#include "cl/clusterFinder.cl"

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  gpucf::fillChargeMap(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPdigits, clusterer.mPchargeMap);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  gpucf::resetMaps(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPdigits, clusterer.mPchargeMap, clusterer.mPpeakMap);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<2>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  gpucf::findPeaks(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPchargeMap, clusterer.mPdigits, clusterer.mPmemory->nDigits, clusterer.mPisPeak, clusterer.mPpeakMap);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<3>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  gpucf::noiseSuppression(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPchargeMap, clusterer.mPpeakMap, clusterer.mPpeaks, clusterer.mPmemory->nPeaks, clusterer.mPisPeak);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<4>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  gpucf::updatePeaks(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPpeaks, clusterer.mPisPeak, clusterer.mPpeakMap);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<5>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  gpucf::countPeaks(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPpeakMap, clusterer.mPchargeMap, clusterer.mPdigits, clusterer.mPmemory->nDigits);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<6>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer)
{
  gpucf::computeClusters(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPchargeMap, clusterer.mPdigits, clusterer.mPmemory->nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterer.mPclusterByRow);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<7>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf)
{
  gpucf::nativeScanUpStart(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPisPeak, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<8>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf)
{
  gpucf::nativeScanUp(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<9>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf)
{
  gpucf::nativeScanTop(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<10>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf)
{
  gpucf::nativeScanDown(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<11>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& clusterer, int iBuf, GPUglobalref() gpucf::PackedDigit* in, GPUglobalref() gpucf::PackedDigit* out)
{
  gpucf::compactDigit(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, in, out, clusterer.mPisPeak, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize);
}
