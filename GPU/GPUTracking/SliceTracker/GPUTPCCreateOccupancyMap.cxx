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

/// \file GPUTPCCreateOccupancyMap.cxx
/// \author David Rohr

#include "GPUTPCCreateOccupancyMap.h"
#include "GPUTPCClusterOccupancyMap.h"

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCCreateOccupancyMap::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, GPUTPCClusterOccupancyMapBin* GPUrestrict() map)
{
  const GPUTrackingInOutPointers& GPUrestrict() ioPtrs = processors.ioPtrs;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = ioPtrs.clustersNative;
  GPUParam& GPUrestrict() param = processors.param;
  const int iSliceRow = iBlock * nThreads + iThread;
  if (iSliceRow >= GPUCA_ROW_COUNT * GPUCA_NSLICES) {
    return;
  }
  const unsigned int iSlice = iSliceRow / GPUCA_ROW_COUNT;
  const unsigned int iRow = iSliceRow % GPUCA_ROW_COUNT;
  for (unsigned int i = 0; i < clusters->nClusters[iSlice][iRow]; i++) {
    const unsigned int bin = clusters->clusters[iSlice][iRow][i].getTime() / param.rec.tpc.occupancyMapTimeBins;
    map[bin].bin[iSlice][iRow]++;
  }
}

template <>
GPUdii() void GPUTPCCreateOccupancyMap::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, GPUTPCClusterOccupancyMapBin* GPUrestrict() map)
{
  GPUParam& GPUrestrict() param = processors.param;
  const int iSliceRow = iBlock * nThreads + iThread;
  if (iSliceRow > GPUCA_ROW_COUNT * GPUCA_NSLICES) {
    return;
  }
  const unsigned int iSlice = iSliceRow / GPUCA_ROW_COUNT;
  const unsigned int iRow = iSliceRow % GPUCA_ROW_COUNT;
  const unsigned int nBins = GPUTPCClusterOccupancyMapBin::getNBins(param);
  const unsigned int nFoldBins = CAMath::Min(5u, nBins);
  unsigned int sum = 0;
  for (unsigned int i = 0; i < nFoldBins; i++) {
    sum += map[i].bin[iSlice][iRow];
  }
  unsigned short lastVal;
  for (unsigned int i = 0; i < nBins; i++) {
    lastVal = map[i].bin[iSlice][iRow];
    map[i].bin[iSlice][iRow] = sum / nFoldBins;
    sum += map[CAMath::Min(i + nFoldBins, nBins - 1)].bin[iSlice][iRow] - lastVal;
  }
}
