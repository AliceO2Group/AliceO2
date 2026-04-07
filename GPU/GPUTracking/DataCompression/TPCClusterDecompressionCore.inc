// Copyright 2024-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCCLusterDecompressionCore.inc
/// \author Gabriele Cimador

#ifndef TPCCLUSTERDECOMPRESSOR_INC
#define TPCCLUSTERDECOMPRESSOR_INC

#include "GPUTPCDecompression.h"
#include "GPUConstantMem.h"
#include "GPUTPCCompressionTrackModel.h"
#include "GPUCommonAlgorithm.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCGeometry.h"

#ifndef GPUCA_GPUCODE
#include <functional>
#endif

using namespace o2::tpc;

namespace o2::gpu
{

class TPCClusterDecompressionCore
{
 public:
#ifndef GPUCA_GPUCODE
  GPUhi() static auto decompressTrackStore(const CompressedClusters& clustersCompressed, const uint32_t offset, uint32_t slice, uint32_t row, uint32_t pad, uint32_t time, std::function<void(const ClusterNative&, uint32_t)> func)
  {
    const auto cluster = ClusterNative(time, clustersCompressed.flagsA[offset], pad, clustersCompressed.sigmaTimeA[offset], clustersCompressed.sigmaPadA[offset], clustersCompressed.qMaxA[offset], clustersCompressed.qTotA[offset]);
    func(cluster, offset);
    return cluster;
  }

  GPUhi() static const auto& decompressTrackStore(const CompressedClusters& clustersCompressed, const uint32_t offset, uint32_t slice, uint32_t row, uint32_t pad, uint32_t time, std::vector<ClusterNative>& clusterVector)
  {
    clusterVector.emplace_back(time, clustersCompressed.flagsA[offset], pad, clustersCompressed.sigmaTimeA[offset], clustersCompressed.sigmaPadA[offset], clustersCompressed.qMaxA[offset], clustersCompressed.qTotA[offset]);
    return clusterVector.back();
  }

  GPUhi() static auto decompressTrackStore(const CompressedClusters& clustersCompressed, const uint32_t offset, uint32_t slice, uint32_t row, uint32_t pad, uint32_t time, std::vector<ClusterNative> (&clusters)[GPUCA_NSECTORS][GPUCA_NROWS], std::atomic_flag (&locks)[GPUCA_NSECTORS][GPUCA_NROWS])
  {
    std::vector<ClusterNative>& clusterVector = clusters[slice][row];
    auto& lock = locks[slice][row];
    while (lock.test_and_set(std::memory_order_acquire)) {
    }
    ClusterNative retVal = decompressTrackStore(clustersCompressed, offset, slice, row, pad, time, clusterVector);
    lock.clear(std::memory_order_release);
    return retVal;
  }
#endif

  GPUdi() static ClusterNative decompressTrackStore(const CompressedClusters& cmprClusters, const uint32_t clusterOffset, uint32_t slice, uint32_t row, uint32_t pad, uint32_t time, GPUTPCDecompression& decompressor)
  {
    uint32_t tmpBufferIndex = slice * (GPUCA_NROWS * decompressor.mMaxNativeClustersPerBuffer) + row * decompressor.mMaxNativeClustersPerBuffer;
    uint32_t currentClusterIndex = CAMath::AtomicAdd(decompressor.mNativeClustersIndex + (slice * GPUCA_NROWS + row), 1u);
    const ClusterNative c(time, cmprClusters.flagsA[clusterOffset], pad, cmprClusters.sigmaTimeA[clusterOffset], cmprClusters.sigmaPadA[clusterOffset], cmprClusters.qMaxA[clusterOffset], cmprClusters.qTotA[clusterOffset]);
    if (currentClusterIndex < decompressor.mMaxNativeClustersPerBuffer) {
      decompressor.mTmpNativeClusters[tmpBufferIndex + currentClusterIndex] = c;
    } else {
      decompressor.raiseError(GPUErrors::ERROR_DECOMPRESSION_ATTACHED_CLUSTER_OVERFLOW, slice * 1000 + row, currentClusterIndex, decompressor.mMaxNativeClustersPerBuffer);
      CAMath::AtomicExch(decompressor.mNativeClustersIndex + (slice * GPUCA_NROWS + row), decompressor.mMaxNativeClustersPerBuffer);
    }
    return c;
  }

  template <typename... Args>
  GPUdi() static void decompressTrack(const CompressedClusters& cmprClusters, const GPUParam& param, const uint32_t maxTime, const uint32_t& trackIndex, uint32_t& clusterOffset, Args&... args)
  {
    float zOffset = 0;
    uint32_t slice = cmprClusters.sliceA[trackIndex];
    uint32_t row = cmprClusters.rowA[trackIndex];
    GPUTPCCompressionTrackModel track;
    constexpr GPUTPCGeometry geo;
    uint32_t clusterIndex;
    for (clusterIndex = 0; clusterIndex < cmprClusters.nTrackClusters[trackIndex]; clusterIndex++) {
      uint32_t pad = 0, time = 0;
      if (clusterIndex != 0) {
        uint8_t tmpSlice = cmprClusters.sliceLegDiffA[clusterOffset - trackIndex - 1];
        bool changeLeg = (tmpSlice >= GPUCA_NSECTORS);
        if (changeLeg) {
          tmpSlice -= GPUCA_NSECTORS;
        }
        if (cmprClusters.nComppressionModes & GPUSettings::CompressionDifferences) {
          slice += tmpSlice;
          if (slice >= GPUCA_NSECTORS) {
            slice -= GPUCA_NSECTORS;
          }
          row += cmprClusters.rowDiffA[clusterOffset - trackIndex - 1];
          if (row >= GPUCA_NROWS) {
            row -= GPUCA_NROWS;
          }
        } else {
          slice = tmpSlice;
          row = cmprClusters.rowDiffA[clusterOffset - trackIndex - 1];
        }
        if (changeLeg && track.Mirror()) {
          break;
        }
        if (track.Propagate(geo.Row2X(row), param.SectorParam[slice].Alpha)) {
          break;
        }
        uint32_t timeTmp = cmprClusters.timeResA[clusterOffset - trackIndex - 1];
        if (timeTmp & 800000) {
          timeTmp |= 0xFF000000;
        }
        time = timeTmp + ClusterNative::packTime(CAMath::Max(0.f, geo.LinearZ2Time(slice, track.Z() + zOffset)));
        float tmpPad = CAMath::Max(0.f, CAMath::Min((float)geo.NPads(GPUCA_NROWS - 1), track.LinearY2Pad(slice, track.Y(), geo.PadWidth(row), geo.NPads(row))));
        pad = cmprClusters.padResA[clusterOffset - trackIndex - 1] + ClusterNative::packPad(tmpPad);
        time = time & 0xFFFFFF;
        pad = (uint16_t)pad;
        if (pad >= geo.NPads(row) * ClusterNative::scalePadPacked) {
          if (pad >= 0xFFFF - 11968) { // Constant 11968 = (2^15 - MAX_PADS(138) * scalePadPacked(64)) / 2
            pad = 0;
          } else {
            pad = geo.NPads(row) * ClusterNative::scalePadPacked - 1;
          }
        }
        if (param.continuousMaxTimeBin > 0 && time >= maxTime) {
          if (time >= 0xFFFFFF - 544768) { // Constant 544768 = (2^23 - LHCMAXBUNCHES(3564) * MAXORBITS(256) * scaleTimePacked(64) / BCPERTIMEBIN(8)) / 2)
            time = 0;
          } else {
            time = maxTime;
          }
        }
      } else {
        time = cmprClusters.timeA[trackIndex];
        pad = cmprClusters.padA[trackIndex];
      }
      const auto cluster = decompressTrackStore(cmprClusters, clusterOffset, slice, row, pad, time, args...);
      float y = track.LinearPad2Y(slice, cluster.getPad(), geo.PadWidth(row), geo.NPads(row));
      float z = geo.LinearTime2Z(slice, cluster.getTime());
      if (clusterIndex == 0) {
        zOffset = z;
        track.Init(geo.Row2X(row), y, z - zOffset, param.SectorParam[slice].Alpha, cmprClusters.qPtA[trackIndex], param);
      }
      if (clusterIndex + 1 < cmprClusters.nTrackClusters[trackIndex] && track.Filter(y, z - zOffset, row)) {
        break;
      }
      clusterOffset++;
    }
    clusterOffset += cmprClusters.nTrackClusters[trackIndex] - clusterIndex;
  }

  GPUdi() static const auto& decompressHitsStore(const CompressedClusters& cmprClusters, uint32_t k, uint32_t time, uint16_t pad, ClusterNative*& clusterBuffer)
  {
    return ((*(clusterBuffer++) = ClusterNative(time, cmprClusters.flagsU[k], pad, cmprClusters.sigmaTimeU[k], cmprClusters.sigmaPadU[k], cmprClusters.qMaxU[k], cmprClusters.qTotU[k])));
  }

#ifndef GPUCA_GPUCODE
  GPUhi() static auto decompressHitsStore(const CompressedClusters& cmprClusters, uint32_t k, uint32_t time, uint16_t pad, std::function<void(const ClusterNative&, uint32_t)> func)
  {
    const auto cluster = ClusterNative(time, cmprClusters.flagsU[k], pad, cmprClusters.sigmaTimeU[k], cmprClusters.sigmaPadU[k], cmprClusters.qMaxU[k], cmprClusters.qTotU[k]);
    func(cluster, k);
    return cluster;
  }
#endif

  template <typename... Args>
  GPUdi() static void decompressHits(const CompressedClusters& cmprClusters, const uint32_t start, const uint32_t end, Args&... args)
  {
    uint32_t time = 0;
    uint16_t pad = 0;
    for (uint32_t k = start; k < end; k++) {
      if (cmprClusters.nComppressionModes & GPUSettings::CompressionDifferences) {
        uint32_t timeTmp = cmprClusters.timeDiffU[k];
        if (timeTmp & 800000) {
          timeTmp |= 0xFF000000;
        }
        time += timeTmp;
        pad += cmprClusters.padDiffU[k];
      } else {
        time = cmprClusters.timeDiffU[k];
        pad = cmprClusters.padDiffU[k];
      }
      decompressHitsStore(cmprClusters, k, time, pad, args...);
    }
  }
};
} // namespace o2::gpu

#endif
