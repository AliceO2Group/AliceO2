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

#include "GPUCommonLogger.h"
#include "GPUTPCClusterFilter.h"
#include "DataFormatsTPC/ClusterNative.h"

using namespace o2::gpu;

GPUTPCClusterFilter::GPUTPCClusterFilter(const o2::tpc::ClusterNativeAccess& clusters, uint8_t filterType)
  : mFilterType(filterType)
{
  if (filterType == 1) {
    // Custom filter settings go here

  } else if (filterType == 2) {
    // PbPb23 filter
    mClusterStats = std::make_unique<std::vector<int>[]>(MaxStacks);
    static bool called = false;
    if (!called) {
      LOGP(info, "GPUTPCClusterFilter called for PbPb 2023 settings");
      called = true;
    }

    for (uint32_t iSector = 0; iSector < GPUTPCGeometry::NSECTORS; iSector++) {
      for (uint32_t iRow = 0; iRow < GPUTPCGeometry::NROWS; iRow++) {
        const uint32_t globalStack = getGlobalStack(iSector, iRow);
        mClusterStats[globalStack].resize(MaxTimeBin);

        for (uint32_t k = 0; k < clusters.nClusters[iSector][iRow]; k++) {
          const o2::tpc::ClusterNative& cl = clusters.clusters[iSector][iRow][k];
          const int clTime = static_cast<int>(cl.getTime());
          const float clQmax = cl.getQmax();

          if (clQmax < 12) {
            if (clTime >= static_cast<int>(mClusterStats[globalStack].size())) {
              mClusterStats[globalStack].resize(mClusterStats[globalStack].size() + 445);
            }
            ++mClusterStats[globalStack][clTime];
          }
        }
      }
    }
  }
}

bool GPUTPCClusterFilter::filter(uint32_t sector, uint32_t row, o2::tpc::ClusterNative& cl)
{
  // Return true to keep the cluster, false to drop it.
  // May change cluster properties by modifying the cl reference.
  // Note that this function might be called multiple times for the same cluster, in which case the final modified cl reference goes into the output clusters.
  if (mFilterType == 2) {
    const uint32_t globalStack = getGlobalStack(sector, row);
    const int clTime = static_cast<int>(cl.getTime());
    const float clQmax = cl.getQmax();
    if ((mClusterStats[globalStack][clTime] > 40 && clQmax < 12) || (mClusterStats[globalStack][clTime] > 200)) {
      return false;
    }
  }

  return true;
}
