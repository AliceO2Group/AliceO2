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
GPUdii() void GPUTPCCreateOccupancyMap::Thread<GPUTPCCreateOccupancyMap::fold>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors, GPUTPCClusterOccupancyMapBin* GPUrestrict() map)
{
  GPUParam& GPUrestrict() param = processors.param;
  const int iSliceRow = iBlock * nThreads + iThread;
  if (iSliceRow > GPUCA_ROW_COUNT * GPUCA_NSLICES) {
    return;
  }
  static constexpr unsigned int FOLD_BINS_BEEFORE_AFTER = 2;
  static constexpr unsigned int FOLD_BINS = FOLD_BINS_BEEFORE_AFTER * 2 + 1;
  const unsigned int iSlice = iSliceRow / GPUCA_ROW_COUNT;
  const unsigned int iRow = iSliceRow % GPUCA_ROW_COUNT;
  const unsigned int nBins = GPUTPCClusterOccupancyMapBin::getNBins(param);
  if (nBins < FOLD_BINS) {
    return;
  }
  unsigned short lastVal[FOLD_BINS_BEEFORE_AFTER];
  unsigned int sum = (FOLD_BINS_BEEFORE_AFTER + 1) * map[0].bin[iSlice][iRow];
  for (unsigned int i = 0; i < FOLD_BINS_BEEFORE_AFTER; i++) {
    sum += map[i + 1].bin[iSlice][iRow];
    lastVal[i] = map[0].bin[iSlice][iRow];
  }
  unsigned int lastValIndex = 0;
  for (unsigned int i = 0; i < nBins; i++) {
    unsigned short useLastVal = lastVal[lastValIndex];
    lastVal[lastValIndex] = map[i].bin[iSlice][iRow];
    map[i].bin[iSlice][iRow] = sum / FOLD_BINS;
    sum += map[CAMath::Min(i + FOLD_BINS_BEEFORE_AFTER + 1, nBins - 1)].bin[iSlice][iRow] - useLastVal;
    lastValIndex = lastValIndex < FOLD_BINS_BEEFORE_AFTER - 1 ? lastValIndex + 1 : 0;
  }
}
