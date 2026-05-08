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

/// \file GPUTPCClusterFilter.h
/// \author David Rohr

#ifndef GPUTPCCLUSTERFILTER_H
#define GPUTPCCLUSTERFILTER_H

#include <memory>
#include <cstdint>
#include <vector>
#include "GPUTPCGeometry.h"

namespace o2::tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
} // namespace o2::tpc

namespace o2::gpu
{
class GPUTPCClusterFilter
{
 public:
  GPUTPCClusterFilter(const o2::tpc::ClusterNativeAccess& clusters, uint8_t filterType);
  bool filter(uint32_t sector, uint32_t row, o2::tpc::ClusterNative& cl);

 private:
  static constexpr uint32_t MaxTimeBin = 14256;
  static constexpr uint32_t MaxStacks = GPUTPCGeometry::NSECTORS * 4;
  uint8_t mFilterType = 0; //< 0: off, 1: custom, 2: PbPb23

  std::unique_ptr<std::vector<int>[]> mClusterStats; //< Number of clusters per stack and time bin

  uint32_t getGlobalStack(uint32_t sector, uint32_t row) const
  {
    int stack = 3;
    if (row < 63) {
      stack = 0;
    } else if (row < 97) {
      stack = 1;
    } else if (row < 127) {
      stack = 2;
    }

    return sector * 4 + stack;
  };
};
} // namespace o2::gpu

#endif
