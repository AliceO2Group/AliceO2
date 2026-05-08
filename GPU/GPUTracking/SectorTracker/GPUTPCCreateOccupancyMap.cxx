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

using namespace o2::gpu;

template <>
GPUdii() void GPUTPCCreateOccupancyMap::Thread<GPUTPCCreateOccupancyMap::fill>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, GPUTPCClusterOccupancyMapBin* GPUrestrict() map)
{
  const GPUTrackingInOutPointers& GPUrestrict() ioPtrs = processors.ioPtrs;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = ioPtrs.clustersNative;
  GPUParam& GPUrestrict() param = processors.param;
  const uint32_t iSectorRow = iBlock * nThreads + iThread;
  if (iSectorRow >= GPUTPCGeometry::NROWS * GPUTPCGeometry::NSECTORS) {
    return;
  }
  const uint32_t iSector = iSectorRow / GPUTPCGeometry::NROWS;
  const uint32_t iRow = iSectorRow % GPUTPCGeometry::NROWS;
  for (uint32_t i = 0; i < clusters->nClusters[iSector][iRow]; i++) {
    const uint32_t bin = clusters->clusters[iSector][iRow][i].getTime() / param.rec.tpc.occupancyMapTimeBins;
    map[bin].bin[iSector][iRow]++;
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
    for (uint32_t iSectorRow = 0; iSectorRow < GPUTPCGeometry::NSECTORS * GPUTPCGeometry::NROWS; iSectorRow++) {
      sum += (&map[i].bin[0][0])[iSectorRow];
    }
  }
  sum /= binmax - binmin;
  output[bin] = sum;
}
