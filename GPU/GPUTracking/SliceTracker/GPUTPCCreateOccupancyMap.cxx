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
GPUdii() void GPUTPCCreateOccupancyMap::Thread<GPUTPCCreateOccupancyMap::fill>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, GPUTPCClusterOccupancyMapBin* GPUrestrict() map)
{
  const GPUTrackingInOutPointers& GPUrestrict() ioPtrs = processors.ioPtrs;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = ioPtrs.clustersNative;
  GPUParam& GPUrestrict() param = processors.param;
  const int32_t iSliceRow = iBlock * nThreads + iThread;
  if (iSliceRow >= GPUCA_ROW_COUNT * GPUCA_NSLICES) {
    return;
  }
  const uint32_t iSlice = iSliceRow / GPUCA_ROW_COUNT;
  const uint32_t iRow = iSliceRow % GPUCA_ROW_COUNT;
  for (uint32_t i = 0; i < clusters->nClusters[iSlice][iRow]; i++) {
    const uint32_t bin = clusters->clusters[iSlice][iRow][i].getTime() / param.rec.tpc.occupancyMapTimeBins;
    map[bin].bin[iSlice][iRow]++;
  }
}

template <>
GPUdii() void GPUTPCCreateOccupancyMap::Thread<GPUTPCCreateOccupancyMap::fold>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, GPUTPCClusterOccupancyMapBin* GPUrestrict() map, uint32_t* GPUrestrict() output)
{
  GPUParam& GPUrestrict() param = processors.param;
  const uint32_t bin = iBlock * nThreads + iThread;
  if (bin >= GPUTPCClusterOccupancyMapBin::getNBins(param)) {
    return;
  }
  int32_t binmin = CAMath::Max<int32_t>(0, bin - param.rec.tpc.occupancyMapTimeBinsAverage);
  int32_t binmax = CAMath::Min<int32_t>(GPUTPCClusterOccupancyMapBin::getNBins(param), bin + param.rec.tpc.occupancyMapTimeBinsAverage + 1);
  uint32_t sum = 0;
  for (int32_t i = binmin; i < binmax; i++) {
    for (int32_t iSliceRow = 0; iSliceRow < GPUCA_NSLICES * GPUCA_ROW_COUNT; iSliceRow++) {
      sum += (&map[i].bin[0][0])[iSliceRow];
    }
  }
  sum /= binmax - binmin;
  output[bin] = sum;
}
