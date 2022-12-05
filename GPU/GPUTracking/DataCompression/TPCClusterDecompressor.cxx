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

/// \file TPCClusterDecompressor.cxx
/// \author David Rohr

#include "TPCClusterDecompressor.h"
#include "GPUO2DataTypes.h"
#include "GPUParam.h"
#include "GPUTPCCompressionTrackModel.h"
#include <algorithm>
#include <cstring>
#include <atomic>
#include "TPCClusterDecompressor.inc"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

int TPCClusterDecompressor::decompress(const CompressedClustersFlat* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param)
{
  CompressedClusters c;
  const CompressedClusters* p;
  if (clustersCompressed->ptrForward) {
    p = clustersCompressed->ptrForward;
  } else {
    c = *clustersCompressed;
    p = &c;
  }
  return decompress(p, clustersNative, allocator, param);
}

int TPCClusterDecompressor::decompress(const CompressedClusters* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param)
{
  if (clustersCompressed->nTracks && clustersCompressed->solenoidBz != -1e6f && clustersCompressed->solenoidBz != param.bzkG) {
    throw std::runtime_error("Configured solenoid Bz does not match value used for track model encoding");
  }
  if (clustersCompressed->nTracks && clustersCompressed->maxTimeBin != -1e6 && clustersCompressed->maxTimeBin != param.par.continuousMaxTimeBin) {
    throw std::runtime_error("Configured max time bin does not match value used for track model encoding");
  }
  std::vector<ClusterNative> clusters[NSLICES][GPUCA_ROW_COUNT];
  std::atomic_flag locks[NSLICES][GPUCA_ROW_COUNT];
  for (unsigned int i = 0; i < NSLICES * GPUCA_ROW_COUNT; i++) {
    (&locks[0][0])[i].clear();
  }
  unsigned int offset = 0, lasti = 0;
  const unsigned int maxTime = (param.par.continuousMaxTimeBin + 1) * ClusterNative::scaleTimePacked - 1;
  GPUCA_OPENMP(parallel for firstprivate(offset, lasti))
  for (unsigned int i = 0; i < clustersCompressed->nTracks; i++) {
    if (i < lasti) {
      offset = lasti = 0; // dynamic OMP scheduling, need to reinitialize offset
    }
    while (lasti < i) {
      offset += clustersCompressed->nTrackClusters[lasti++];
    }
    lasti++;
    decompressTrack(clustersCompressed, param, maxTime, i, offset, clusters, locks);
  }
  size_t nTotalClusters = clustersCompressed->nAttachedClusters + clustersCompressed->nUnattachedClusters;
  ClusterNative* clusterBuffer = allocator(nTotalClusters);
  unsigned int offsets[NSLICES][GPUCA_ROW_COUNT];
  offset = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      clustersNative.nClusters[i][j] = clusters[i][j].size() + ((i * GPUCA_ROW_COUNT + j >= clustersCompressed->nSliceRows) ? 0 : clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j]);
      offsets[i][j] = offset;
      offset += (i * GPUCA_ROW_COUNT + j >= clustersCompressed->nSliceRows) ? 0 : clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
    }
  }
  clustersNative.clustersLinear = clusterBuffer;
  clustersNative.setOffsetPtrs();
  GPUCA_OPENMP(parallel for)
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      ClusterNative* buffer = &clusterBuffer[clustersNative.clusterOffset[i][j]];
      if (clusters[i][j].size()) {
        memcpy((void*)buffer, (const void*)clusters[i][j].data(), clusters[i][j].size() * sizeof(clusterBuffer[0]));
      }
      ClusterNative* cl = buffer + clusters[i][j].size();
      unsigned int end = offsets[i][j] + ((i * GPUCA_ROW_COUNT + j >= clustersCompressed->nSliceRows) ? 0 : clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j]);
      decompressHits(clustersCompressed, offsets[i][j], end, cl);
      if (param.rec.tpc.clustersShiftTimebins != 0.f) {
        for (unsigned int k = 0; k < clustersNative.nClusters[i][j]; k++) {
          auto& cl = buffer[k];
          float t = cl.getTime() + param.rec.tpc.clustersShiftTimebins;
          if (t < 0) {
            t = 0;
          }
          if (param.par.continuousMaxTimeBin > 0 && t > param.par.continuousMaxTimeBin) {
            t = param.par.continuousMaxTimeBin;
          }
          cl.setTime(t);
        }
      }
      std::sort(buffer, buffer + clustersNative.nClusters[i][j]);
    }
  }

  return 0;
}
