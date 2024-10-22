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

/// \file GPUTPCClusterFilter.cxx
/// \author David Rohr

#include "GPUTPCClusterFilter.h"
#include "DataFormatsTPC/ClusterNative.h"

using namespace o2::gpu;

GPUTPCClusterFilter::GPUTPCClusterFilter(const o2::tpc::ClusterNativeAccess& clusters)
{
  // Could initialize private variables based on the clusters here
}

bool GPUTPCClusterFilter::filter(uint32_t sector, uint32_t row, o2::tpc::ClusterNative& cl)
{
  // Return true to keep the cluster, false to drop it.
  // May change cluster properties by modifying the cl reference.
  // Note that this function might be called multiple times for the same cluster, in which case the final modified cl reference goes into the output clusters.
  return true;
}
