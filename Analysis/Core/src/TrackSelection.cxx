// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
// Class for track selection
//

#include "Analysis/TrackSelection.h"
#include <TH1F.h>

ClassImp(TrackSelection)

  TrackSelection::TrackSelection()
  : TObject(), mMinPt{0.}, mMaxPt{1e10}, mMinEta{-1e10}, mMaxEta{1e10}, mMinNClustersTPC{0}, mMinNCrossedRowsTPC{0}, mMinNClustersITS{0}, mMaxChi2PerClusterTPC{1e10}, mMaxChi2PerClusterITS{1e10}, mMinNCrossedRowsOverFindableClustersTPC{0}, mMaxDcaXY{1e10}, mMaxDcaZ{1e10}, mRequireITSRefit{false}, mRequireTPCRefit{false}
{
}

bool TrackSelection::FulfillsITSHitRequirements(uint8_t itsClusterMap)
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
};
