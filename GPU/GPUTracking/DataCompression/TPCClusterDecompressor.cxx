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
  std::vector<ClusterNative> clusters[NSLICES][GPUCA_ROW_COUNT];
  std::atomic_flag locks[NSLICES][GPUCA_ROW_COUNT];
  for (unsigned int i = 0; i < NSLICES * GPUCA_ROW_COUNT; i++) {
    (&locks[0][0])[i].clear();
  }
  unsigned int offset = 0, lasti = 0;
  GPUCA_OPENMP(parallel for firstprivate(offset, lasti))
  for (unsigned int i = 0; i < clustersCompressed->nTracks; i++) {
    while (lasti < i) {
      offset += clustersCompressed->nTrackClusters[lasti++];
    }
    lasti++;
    float zOffset = 0;
    unsigned int slice = clustersCompressed->sliceA[i];
    unsigned int row = clustersCompressed->rowA[i];
    GPUTPCCompressionTrackModel track;
    unsigned int j;
    for (j = 0; j < clustersCompressed->nTrackClusters[i]; j++) {
      unsigned int pad = 0, time = 0;
      if (j) {
        unsigned char tmpSlice = clustersCompressed->sliceLegDiffA[offset - i - 1];
        bool changeLeg = (tmpSlice >= NSLICES);
        if (changeLeg) {
          tmpSlice -= NSLICES;
        }
        if (clustersCompressed->nComppressionModes & GPUSettings::CompressionDifferences) {
          slice += tmpSlice;
          if (slice >= NSLICES) {
            slice -= NSLICES;
          }
          row += clustersCompressed->rowDiffA[offset - i - 1];
          if (row >= GPUCA_ROW_COUNT) {
            row -= GPUCA_ROW_COUNT;
          }
        } else {
          slice = tmpSlice;
          row = clustersCompressed->rowDiffA[offset - i - 1];
        }
        if (changeLeg && track.Mirror()) {
          break;
        }
        if (track.Propagate(param.tpcGeometry.Row2X(row), param.SliceParam[slice].Alpha)) {
          break;
        }
        unsigned int timeTmp = clustersCompressed->timeResA[offset - i - 1];
        if (timeTmp & 800000) {
          timeTmp |= 0xFF000000;
        }
        time = timeTmp + ClusterNative::packTime(CAMath::Max(0.f, param.tpcGeometry.LinearZ2Time(slice, track.Z() + zOffset)));
        float tmpPad = CAMath::Max(0.f, CAMath::Min((float)param.tpcGeometry.NPads(GPUCA_ROW_COUNT - 1), param.tpcGeometry.LinearY2Pad(slice, row, track.Y())));
        pad = clustersCompressed->padResA[offset - i - 1] + ClusterNative::packPad(tmpPad);
      } else {
        time = clustersCompressed->timeA[i];
        pad = clustersCompressed->padA[i];
      }
      std::vector<ClusterNative>& clusterVector = clusters[slice][row];
      auto& lock = locks[slice][row];
      while (lock.test_and_set(std::memory_order_acquire)) {
      }
      clusterVector.emplace_back(time, clustersCompressed->flagsA[offset], pad, clustersCompressed->sigmaTimeA[offset], clustersCompressed->sigmaPadA[offset], clustersCompressed->qMaxA[offset], clustersCompressed->qTotA[offset]);
      auto& cluster = clusterVector.back();
      float y = param.tpcGeometry.LinearPad2Y(slice, row, cluster.getPad());
      float z = param.tpcGeometry.LinearTime2Z(slice, cluster.getTime());
      lock.clear(std::memory_order_release);
      if (j == 0) {
        zOffset = z;
        track.Init(param.tpcGeometry.Row2X(row), y, z - zOffset, param.SliceParam[slice].Alpha, clustersCompressed->qPtA[i], param);
      }
      if (j + 1 < clustersCompressed->nTrackClusters[i] && track.Filter(y, z - zOffset, row)) {
        break;
      }
      offset++;
    }
    offset += clustersCompressed->nTrackClusters[i] - j;
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
      unsigned int time = 0;
      unsigned short pad = 0;
      ClusterNative* cl = buffer + clusters[i][j].size();
      unsigned int end = offsets[i][j] + ((i * GPUCA_ROW_COUNT + j >= clustersCompressed->nSliceRows) ? 0 : clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j]);
      for (unsigned int k = offsets[i][j]; k < end; k++) {
        /*if (cl >= clustersNative.clustersLinear + nTotalClusters) {
          throw std::runtime_error("Bad TPC CTF data, decoded more clusters than announced");
        }*/
        if (clustersCompressed->nComppressionModes & GPUSettings::CompressionDifferences) {
          unsigned int timeTmp = clustersCompressed->timeDiffU[k];
          if (timeTmp & 800000) {
            timeTmp |= 0xFF000000;
          }
          time += timeTmp;
          pad += clustersCompressed->padDiffU[k];
        } else {
          time = clustersCompressed->timeDiffU[k];
          pad = clustersCompressed->padDiffU[k];
        }
        *(cl++) = ClusterNative(time, clustersCompressed->flagsU[k], pad, clustersCompressed->sigmaTimeU[k], clustersCompressed->sigmaPadU[k], clustersCompressed->qMaxU[k], clustersCompressed->qTotU[k]);
      }
      std::sort(buffer, buffer + clustersNative.nClusters[i][j]);
    }
  }

  return 0;
}
