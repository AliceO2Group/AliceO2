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

/// \file TrackCuts.h
/// \brief Class to store some methods used in TrackCuts
/// \author amelia.lindner@cern.ch

#ifndef ALICEO2_TRACKMETHODS_H
#define ALICEO2_TRACKMETHODS_H

#include "DataFormatsTPC/TrackTPC.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "DataFormatsITS/TrackITS.h"
#include <set>
#include <vector>

namespace o2
{
class TrackMethods
{
 public:
  static void countTPCClusters(const o2::tpc::TrackTPC& track,
                               const gsl::span<const o2::tpc::TPCClRefElem>& tpcClusRefs,
                               const gsl::span<const unsigned char>& tpcClusShMap,
                               const o2::tpc::ClusterNativeAccess& tpcClusAcc,
                               uint8_t& shared, uint8_t& found, uint8_t& crossed)
  {
    LOGP(debug, "tpcClusRefs {}/{}", (void*)tpcClusRefs.data(), tpcClusRefs.size());
    LOGP(debug, "tpcClusShMap {}/{}", (void*)tpcClusShMap.data(), tpcClusShMap.size());
    LOGP(debug, "tpcClusAcc {}/{}", (void*)tpcClusAcc.clustersLinear, tpcClusAcc.nClustersTotal);
    constexpr int maxRows = 152;
    constexpr int neighbour = 2;
    std::array<bool, maxRows> clMap{}, shMap{};
    uint8_t sectorIndex;
    uint8_t rowIndex;
    uint32_t clusterIndex;
    shared = 0;
    for (int i = 0; i < track.getNClusterReferences(); i++) {
      o2::tpc::TrackTPC::getClusterReference(tpcClusRefs, i, sectorIndex, rowIndex, clusterIndex, track.getClusterRef());
      unsigned int absoluteIndex = tpcClusAcc.clusterOffset[sectorIndex][rowIndex] + clusterIndex;
      clMap[rowIndex] = true;
      if (tpcClusShMap[absoluteIndex] & GPUCA_NAMESPACE::gpu::GPUTPCGMMergedTrackHit::flagShared) {
        if (!shMap[rowIndex]) {
          shared++;
        }
        shMap[rowIndex] = true;
      }
    }

    crossed = 0;
    found = 0;
    int last = -1;
    for (int i = 0; i < maxRows; i++) {
      if (clMap[i]) {
        crossed++;
        found++;
        last = i;
      } else if ((i - last) <= neighbour) {
        crossed++;
      } else {
        int lim = std::min(i + 1 + neighbour, maxRows);
        for (int j = i + 1; j < lim; j++) {
          if (clMap[j]) {
            crossed++;
          }
        }
      }
    }
  }
  static bool FulfillsITSHitRequirements(uint8_t itsClusterMap, std::vector<std::pair<int8_t, std::set<uint8_t>>> mRequiredITSHits)
  {
    constexpr uint8_t bit = 1;
    for (auto& itsRequirement : mRequiredITSHits) {
      auto hits = std::count_if(itsRequirement.second.begin(), itsRequirement.second.end(), [&](auto&& requiredLayer) { return itsClusterMap & (bit << requiredLayer); });
      if ((itsRequirement.first == -1) && (hits > 0)) {
        return false; // no hits were required in specified layers
      } else if (hits < itsRequirement.first) {
        return false; // not enough hits found in specified layers
      }
    }
    return true;
  }

  ClassDefNV(TrackMethods, 1);
};
} // namespace o2

#endif
