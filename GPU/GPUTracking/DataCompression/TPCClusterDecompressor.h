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

/// \file TPCClusterDecompressor.h
/// \author David Rohr

#ifndef TPCCLUSTERDECOMPRESSOR_H
#define TPCCLUSTERDECOMPRESSOR_H

#include "GPUTPCCompression.h"
#include <vector>
#include <functional>

namespace o2::tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
} // namespace o2::tpc

namespace GPUCA_NAMESPACE::gpu
{
struct GPUParam;

class TPCClusterDecompressor
{
 public:
  static constexpr unsigned int NSLICES = GPUCA_NSLICES;
  int decompress(const o2::tpc::CompressedClustersFlat* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param);
  int decompress(const o2::tpc::CompressedClusters* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param);

  template <typename... Args>
  void decompressTrack(const o2::tpc::CompressedClusters* clustersCompressed, const GPUParam& param, const unsigned int maxTime, const unsigned int i, unsigned int& offset, Args&... args);
  template <typename... Args>
  void decompressHits(const o2::tpc::CompressedClusters* clustersCompressed, const unsigned int start, const unsigned int end, Args&... args);

 protected:
  template <typename... Args>
  auto& decompressTrackStore(const o2::tpc::CompressedClusters* clustersCompressed, const unsigned int offset, unsigned int slice, unsigned int row, unsigned int pad, unsigned int time, Args&... args);
  template <typename... Args>
  auto& decompressHitsStore(const o2::tpc::CompressedClusters* clustersCompressed, unsigned int k, unsigned int time, unsigned short pad, Args&... args);
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
