// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterStatistics.cxx
/// \author David Rohr

#include "GPUTPCClusterStatistics.h"
#include "ClusterNativeAccessExt.h"
#include <algorithm>
#include <cstring>

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCClusterStatistics::RunStatistics(const ClusterNativeAccessExt* clustersNative, const o2::TPC::CompressedClusters* clustersCompressed)
{
  o2::TPC::ClusterNativeAccessFullTPC clustersNativeDecoded;
  std::vector<o2::TPC::ClusterNative> clusterBuffer;
  mDecoder.decompress(clustersCompressed, clustersNativeDecoded, clusterBuffer);
  std::vector<o2::TPC::ClusterNative> tmpClusters;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      if (clustersNative->nClusters[i][j] != clustersNativeDecoded.nClusters[i][j]) {
        printf("Number of clusters mismatch slice %u row %u: %d v.s. %d\n", i, j, clustersNative->nClusters[i][j], clustersNativeDecoded.nClusters[i][j]);
        return;
      }
      tmpClusters.resize(clustersNative->nClusters[i][j]);
      memcpy((void*) tmpClusters.data(), (const void*) clustersNative->clusters[i][j], clustersNative->nClusters[i][j] * sizeof(tmpClusters[0]));
      std::sort(tmpClusters.begin(), tmpClusters.end());
      for (unsigned int k = 0; k < clustersNative->nClusters[i][j]; k++) {
        const o2::TPC::ClusterNative& c1 = tmpClusters[k];
        const o2::TPC::ClusterNative& c2 = clustersNativeDecoded.clusters[i][j][k];
        if (c1.timeFlagsPacked != c2.timeFlagsPacked || c1.padPacked != c2.padPacked || c1.sigmaTimePacked != c2.sigmaTimePacked || c1.sigmaPadPacked != c2.sigmaPadPacked || c1.qMax != c2.qMax || c1.qTot != c2.qTot) {
          printf("Cluster mismatch: slice %2u row %3u hit %5u: %6d %3d %4d %3d %3d %4d %4d\n", i, j, k, (int)c1.getTimePacked(), (int)c1.getFlags(), (int)c1.padPacked, (int)c1.sigmaTimePacked, (int)c1.sigmaPadPacked, (int)c1.qMax, (int)c1.qTot);
          printf("%45s %6d %3d %4d %3d %3d %4d %4d\n", "", (int)c2.getTimePacked(), (int)c2.getFlags(), (int)c2.padPacked, (int)c2.sigmaTimePacked, (int)c2.sigmaPadPacked, (int)c2.qMax, (int)c2.qTot);
          return;
        }
      }
    }
  }
  printf("Cluster decoding verification: PASSED\n");
}
