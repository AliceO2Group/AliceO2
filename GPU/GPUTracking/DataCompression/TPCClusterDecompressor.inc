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

/// \file TPCClusterDecompressor.inc
/// \author David Rohr

#include "TPCClusterDecompressor.h"
#include "GPUO2DataTypes.h"
#include "GPUParam.h"
#include "GPUTPCCompressionTrackModel.h"
#include <algorithm>
#include <cstring>
#include <atomic>
#include <functional>

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

static inline auto decompressTrackStore(const o2::tpc::CompressedClusters* clustersCompressed, const unsigned int offset, unsigned int slice, unsigned int row, unsigned int pad, unsigned int time, std::function<void(const ClusterNative&, unsigned int)> func)
{
  const auto cluster = ClusterNative(time, clustersCompressed->flagsA[offset], pad, clustersCompressed->sigmaTimeA[offset], clustersCompressed->sigmaPadA[offset], clustersCompressed->qMaxA[offset], clustersCompressed->qTotA[offset]);
  func(cluster, offset);
  return cluster;
}

static inline const auto& decompressTrackStore(const o2::tpc::CompressedClusters* clustersCompressed, const unsigned int offset, unsigned int slice, unsigned int row, unsigned int pad, unsigned int time, std::vector<ClusterNative>& clusterVector)
{
  clusterVector.emplace_back(time, clustersCompressed->flagsA[offset], pad, clustersCompressed->sigmaTimeA[offset], clustersCompressed->sigmaPadA[offset], clustersCompressed->qMaxA[offset], clustersCompressed->qTotA[offset]);
  return clusterVector.back();
}

static inline const auto& decompressTrackStore(const o2::tpc::CompressedClusters* clustersCompressed, const unsigned int offset, unsigned int slice, unsigned int row, unsigned int pad, unsigned int time, std::vector<ClusterNative> (&clusters)[GPUCA_NSLICES][GPUCA_ROW_COUNT], std::atomic_flag (&locks)[GPUCA_NSLICES][GPUCA_ROW_COUNT])
{
  std::vector<ClusterNative>& clusterVector = clusters[slice][row];
  auto& lock = locks[slice][row];
  while (lock.test_and_set(std::memory_order_acquire)) {
  }
  auto& cluster = decompressTrackStore(clustersCompressed, offset, slice, row, pad, time, clusterVector);
  lock.clear(std::memory_order_release);
  return cluster;
}

template <typename... Args>
inline void TPCClusterDecompressor::decompressTrack(const CompressedClusters* clustersCompressed, const GPUParam& param, const unsigned int maxTime, const unsigned int i, unsigned int& offset, Args&... args)
{
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
      time = time & 0xFFFFFF;
      pad = (unsigned short)pad;
      if (pad >= param.tpcGeometry.NPads(row) * ClusterNative::scalePadPacked) {
        if (pad >= 0xFFFF - 11968) { // Constant 11968 = (2^15 - MAX_PADS(138) * scalePadPacked(64)) / 2
          pad = 0;
        } else {
          pad = param.tpcGeometry.NPads(row) * ClusterNative::scalePadPacked - 1;
        }
      }
      if (param.par.continuousMaxTimeBin > 0 && time >= maxTime) {
        if (time >= 0xFFFFFF - 544768) { // Constant 544768 = (2^23 - LHCMAXBUNCHES(3564) * MAXORBITS(256) * scaleTimePacked(64) / BCPERTIMEBIN(8)) / 2)
          time = 0;
        } else {
          time = maxTime;
        }
      }
    } else {
      time = clustersCompressed->timeA[i];
      pad = clustersCompressed->padA[i];
    }
    const auto& cluster = decompressTrackStore(clustersCompressed, offset, slice, row, pad, time, args...);
    float y = param.tpcGeometry.LinearPad2Y(slice, row, cluster.getPad());
    float z = param.tpcGeometry.LinearTime2Z(slice, cluster.getTime());
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

static inline const auto& decompressHitsStore(const CompressedClusters* clustersCompressed, unsigned int k, unsigned int time, unsigned short pad, ClusterNative* &cl)
{
  return ((*(cl++) = ClusterNative(time, clustersCompressed->flagsU[k], pad, clustersCompressed->sigmaTimeU[k], clustersCompressed->sigmaPadU[k], clustersCompressed->qMaxU[k], clustersCompressed->qTotU[k])));
}

static inline auto decompressHitsStore(const CompressedClusters* clustersCompressed, unsigned int k, unsigned int time, unsigned short pad, std::function<void(const ClusterNative&, unsigned int)> func)
{
  const auto cluster = ClusterNative(time, clustersCompressed->flagsU[k], pad, clustersCompressed->sigmaTimeU[k], clustersCompressed->sigmaPadU[k], clustersCompressed->qMaxU[k], clustersCompressed->qTotU[k]);
  func(cluster, k);
  return cluster;
}

template <typename... Args>
inline void TPCClusterDecompressor::decompressHits(const CompressedClusters* clustersCompressed, const unsigned int start, const unsigned int end, Args&... args)
{
  unsigned int time = 0;
  unsigned short pad = 0;
  for (unsigned int k = start; k < end; k++) {
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
    decompressHitsStore(clustersCompressed, k, time, pad, args...);
  }
}
