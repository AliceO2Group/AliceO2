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

/// \file GPUTPCSectorDebugSortKernels.cxx
/// \author David Rohr

#include "GPUParam.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCHit.h"
#include "GPUTPCSliceData.h"
#include "GPUProcessor.h"
#include "GPUO2DataTypes.h"
#include "GPUCommonMath.h"
#include "GPUCommonAlgorithm.h"
#include "GPUTPCSectorDebugSortKernels.h"

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCSectorDebugSortKernels::Thread<GPUTPCSectorDebugSortKernels::hitData>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() tracker)
{
  const uint32_t iRow = iBlock;
  const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() row = tracker.Data().Row(iRow);
  const MEM_GLOBAL(GPUTPCGrid) & GPUrestrict() grid = row.Grid();
  for (uint32_t i = iThread; i < grid.N(); i += nThreads) {
    uint32_t jMin = tracker.Data().FirstHitInBin(row, i);
    uint32_t jMax = tracker.Data().FirstHitInBin(row, i + 1);
    const uint32_t n = jMax - jMin;
    calink* GPUrestrict() tmp1 = tracker.Data().HitLinkUpData(row) + jMin;
    auto* GPUrestrict() tmp2 = tracker.Data().HitWeights() + row.HitNumberOffset() + jMin;
    cahit2* GPUrestrict() hitData = tracker.Data().HitData(row) + jMin;
    int32_t* GPUrestrict() clusterId = tracker.Data().ClusterDataIndex() + row.HitNumberOffset() + jMin;
    for (uint32_t j = 0; j < n; j++) {
      tmp1[j] = j;
    }
    GPUCommonAlgorithm::sort(tmp1, tmp1 + n, [&hitData, &clusterId](const calink& a, const calink& b) {
      if (hitData[a].x != hitData[b].x) {
        return hitData[a].x < hitData[b].x;
      }
      if (hitData[a].y != hitData[b].y) {
        return hitData[a].y < hitData[b].y;
      }
      return clusterId[a] < clusterId[b];
    });
    for (uint32_t j = 0; j < n; j++) {
      tmp2[j] = hitData[j].x;
    }
    for (uint32_t j = 0; j < n; j++) {
      hitData[j].x = tmp2[tmp1[j]];
    }
    for (uint32_t j = 0; j < n; j++) {
      tmp2[j] = hitData[j].y;
    }
    for (uint32_t j = 0; j < n; j++) {
      hitData[j].y = tmp2[tmp1[j]];
    }
    for (uint32_t j = 0; j < n; j++) {
      tmp2[j] = clusterId[j];
    }
    for (uint32_t j = 0; j < n; j++) {
      clusterId[j] = tmp2[tmp1[j]];
    }
  }
}

template <>
GPUdii() void GPUTPCSectorDebugSortKernels::Thread<GPUTPCSectorDebugSortKernels::startHits>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() tracker)
{
  if (iThread || iBlock) {
    return;
  }
  GPUCommonAlgorithm::sortDeviceDynamic(tracker.TrackletStartHits(), tracker.TrackletStartHits() + *tracker.NStartHits(), [](const GPUTPCHitId& a, const GPUTPCHitId& b) {
    if (a.RowIndex() != b.RowIndex()) {
      return (a.RowIndex() < b.RowIndex());
    }
    return (a.HitIndex() < b.HitIndex());
  });
}

template <>
GPUdii() void GPUTPCSectorDebugSortKernels::Thread<GPUTPCSectorDebugSortKernels::sliceTracks>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() tracker)
{
  if (iThread || iBlock) {
    return;
  }
  auto sorter = [&tracker](const GPUTPCTrack& trk1, const GPUTPCTrack& trk2) {
    if (trk1.NHits() != trk2.NHits()) {
      return trk1.NHits() > trk2.NHits();
    }
    if (trk1.Param().Y() != trk2.Param().Y()) {
      return trk1.Param().Y() > trk2.Param().Y();
    }
    if (trk1.Param().Z() != trk2.Param().Z()) {
      return trk1.Param().Z() > trk2.Param().Z();
    }
    if (tracker.TrackHits()[trk1.FirstHitID()].RowIndex() != tracker.TrackHits()[trk2.FirstHitID()].RowIndex()) {
      return tracker.TrackHits()[trk1.FirstHitID()].RowIndex() > tracker.TrackHits()[trk2.FirstHitID()].RowIndex();
    }
    return tracker.TrackHits()[trk1.FirstHitID()].HitIndex() > tracker.TrackHits()[trk2.FirstHitID()].HitIndex();
  };
  GPUCommonAlgorithm::sortDeviceDynamic(tracker.Tracks(), tracker.Tracks() + tracker.CommonMemory()->nLocalTracks, sorter);
  GPUCommonAlgorithm::sortDeviceDynamic(tracker.Tracks() + tracker.CommonMemory()->nLocalTracks, tracker.Tracks() + *tracker.NTracks(), sorter);
}
