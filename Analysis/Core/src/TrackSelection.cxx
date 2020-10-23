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
}

const std::string TrackSelection::mCutNames[static_cast<int>(TrackSelection::TrackCuts::kNCuts)] = {"TrackType", "PtRange", "EtaRange", "TPCNCls", "TPCCrossedRows", "TPCCrossedRowsOverNCls", "TPCChi2NDF", "TPCRefit", "ITSNCls", "ITSChi2NDF", "ITSRefit", "ITSHits", "GoldenChi2", "DCAxy", "DCAz"};
