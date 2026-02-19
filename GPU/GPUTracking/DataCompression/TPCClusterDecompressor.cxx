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
#include "GPULogging.h"
#include <algorithm>
#include <cstring>
#include <atomic>
#include "TPCClusterDecompressionCore.inc"

#include <oneapi/tbb.h>

using namespace o2::gpu;
using namespace o2::tpc;

int32_t TPCClusterDecompressor::decompress(const CompressedClustersFlat* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param, bool deterministicRec)
{
  CompressedClusters c;
  const CompressedClusters* p;
  if (clustersCompressed->ptrForward) {
    p = clustersCompressed->ptrForward;
  } else {
    c = *clustersCompressed;
    p = &c;
  }
  return decompress(p, clustersNative, allocator, param, deterministicRec);
}

int32_t TPCClusterDecompressor::decompress(const CompressedClusters* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param, bool deterministicRec)
{
  if (clustersCompressed->nTracks && clustersCompressed->solenoidBz != -1e6f && clustersCompressed->solenoidBz != param.bzkG) {
    throw std::runtime_error("Configured solenoid Bz " + std::to_string(param.bzkG) + " does not match value used for track model encoding " + std::to_string(clustersCompressed->solenoidBz));
  }
  if (clustersCompressed->nTracks && clustersCompressed->maxTimeBin != -1e6 && clustersCompressed->maxTimeBin != param.continuousMaxTimeBin) {
    throw std::runtime_error("Configured max time bin " + std::to_string(param.continuousMaxTimeBin) + " does not match value used for track model encoding " + std::to_string(clustersCompressed->maxTimeBin));
  }
  std::vector<ClusterNative> clusters[NSECTORS][GPUCA_ROW_COUNT];
  std::atomic_flag locks[NSECTORS][GPUCA_ROW_COUNT];
  for (uint32_t i = 0; i < NSECTORS * GPUCA_ROW_COUNT; i++) {
    (&locks[0][0])[i].clear();
  }
  const uint32_t maxTime = param.continuousMaxTimeBin > 0 ? ((param.continuousMaxTimeBin + 1) * ClusterNative::scaleTimePacked - 1) : TPC_MAX_TIME_BIN_TRIGGERED;
  tbb::parallel_for(tbb::blocked_range<uint32_t>(0, clustersCompressed->nTracks), [&](const tbb::blocked_range<uint32_t>& range) {
    uint32_t offset = 0, lasti = 0;
    for (uint32_t i = range.begin(); i < range.end(); i++) {
      if (i < lasti) {
        offset = lasti = 0; // dynamic scheduling order, need to reinitialize offset
      }
      while (lasti < i) {
        offset += clustersCompressed->nTrackClusters[lasti++];
      }
      lasti++;
      TPCClusterDecompressionCore::decompressTrack(*clustersCompressed, param, maxTime, i, offset, clusters, locks);
    }
  });
  size_t nTotalClusters = clustersCompressed->nAttachedClusters + clustersCompressed->nUnattachedClusters;
  ClusterNative* clusterBuffer = allocator(nTotalClusters);
  uint32_t offsets[NSECTORS][GPUCA_ROW_COUNT];
  uint32_t offset = 0;
  uint32_t decodedAttachedClusters = 0;
  for (uint32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
      clustersNative.nClusters[i][j] = clusters[i][j].size() + ((i * GPUCA_ROW_COUNT + j >= clustersCompressed->nSliceRows) ? 0 : clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j]);
      offsets[i][j] = offset;
      offset += (i * GPUCA_ROW_COUNT + j >= clustersCompressed->nSliceRows) ? 0 : clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
      decodedAttachedClusters += clusters[i][j].size();
    }
  }
  if (decodedAttachedClusters != clustersCompressed->nAttachedClusters) {
    GPUWarning("%u / %u clusters failed track model decoding (%f %%)", clustersCompressed->nAttachedClusters - decodedAttachedClusters, clustersCompressed->nAttachedClusters, 100.f * (float)(clustersCompressed->nAttachedClusters - decodedAttachedClusters) / (float)clustersCompressed->nAttachedClusters);
  }
  clustersNative.clustersLinear = clusterBuffer;
  clustersNative.setOffsetPtrs();
  tbb::parallel_for<uint32_t>(0, NSECTORS, [&](auto i) {
    for (uint32_t j = 0; j < GPUCA_ROW_COUNT; j++) {
      ClusterNative* buffer = &clusterBuffer[clustersNative.clusterOffset[i][j]];
      if (clusters[i][j].size()) {
        memcpy((void*)buffer, (const void*)clusters[i][j].data(), clusters[i][j].size() * sizeof(clusterBuffer[0]));
      }
      ClusterNative* clout = buffer + clusters[i][j].size();
      uint32_t end = offsets[i][j] + ((i * GPUCA_ROW_COUNT + j >= clustersCompressed->nSliceRows) ? 0 : clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j]);
      TPCClusterDecompressionCore::decompressHits(*clustersCompressed, offsets[i][j], end, clout);
      if (param.rec.tpc.clustersEdgeFixDistance > 0.f) {
        constexpr GPUTPCGeometry geo;
        for (uint32_t k = 0; k < clustersNative.nClusters[i][j]; k++) {
          auto& cluster = buffer[k];
          if (cluster.getFlags() & ClusterNative::flagEdge) {
            auto padF = cluster.getPad();
            float distEdge = padF < geo.NPads(j) / 2 ? padF : geo.NPads(j) - 1 - padF;
            if (distEdge > param.rec.tpc.clustersEdgeFixDistance) {
              cluster.setFlags(cluster.getFlags() ^ ClusterNative::flagEdge);
            }
          }
        }
      }
      if (param.rec.tpc.clustersShiftTimebins != 0.f) {
        for (uint32_t k = 0; k < clustersNative.nClusters[i][j]; k++) {
          auto& cl = buffer[k];
          float t = cl.getTime() + param.rec.tpc.clustersShiftTimebins;
          if (t < 0) {
            t = 0;
          }
          if (param.continuousMaxTimeBin > 0 && t > param.continuousMaxTimeBin) {
            t = param.continuousMaxTimeBin;
          }
          cl.setTime(t);
        }
      }
      if (deterministicRec) {
        std::sort(buffer, buffer + clustersNative.nClusters[i][j]);
      }
    } // clang-format off
  }, tbb::simple_partitioner()); // clang-format on
  return 0;
}
