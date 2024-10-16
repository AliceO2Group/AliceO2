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

#include <cstdint>

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
  GPUTPCClusterFilter(const o2::tpc::ClusterNativeAccess& clusters);
  bool filter(uint32_t sector, uint32_t row, o2::tpc::ClusterNative& cl);
};
} // namespace o2::gpu

#endif
