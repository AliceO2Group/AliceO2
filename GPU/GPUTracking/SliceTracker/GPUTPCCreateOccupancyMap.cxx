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
GPUdii() void GPUTPCCreateOccupancyMap::Thread<GPUTPCCreateOccupancyMap::fill>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, GPUTPCClusterOccupancyMapBin* GPUrestrict() map)
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
GPUdii() void GPUTPCCreateOccupancyMap::Thread<GPUTPCCreateOccupancyMap::fold>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, GPUTPCClusterOccupancyMapBin* GPUrestrict() map, unsigned int* GPUrestrict() output)
{
  GPUParam& GPUrestrict() param = processors.param;
  const unsigned int bin = iBlock * nThreads + iThread;
  if (bin >= GPUTPCClusterOccupancyMapBin::getNBins(param)) {
    return;
  }
  int binmin = CAMath::Max<int>(0, bin - param.rec.tpc.occupancyMapTimeBinsAverage);
  int binmax = CAMath::Min<int>(GPUTPCClusterOccupancyMapBin::getNBins(param), bin + param.rec.tpc.occupancyMapTimeBinsAverage + 1);
  unsigned int sum = 0;
  for (int i = binmin; i < binmax; i++) {
    for (int iSliceRow = 0; iSliceRow < GPUCA_NSLICES * GPUCA_ROW_COUNT; iSliceRow++) {
      sum += (&map[i].bin[0][0])[iSliceRow];
    }
  }
  sum /= binmax - binmin;
  output[bin] = sum;
}
