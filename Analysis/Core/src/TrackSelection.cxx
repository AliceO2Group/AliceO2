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

using namespace o2;
using namespace o2::framework;

ClassImp(TrackSelection)

  TrackSelection::TrackSelection()
  : TObject(), mMinPt{0.}, mMaxPt{1e10}, mMinEta{0.}, mMaxEta{1e10}, mMinNClustersTPC{0}, mMinNCrossedRowsTPC{0}, mMinNClustersITS{0}, mMaxChi2PerClusterTPC{1e10}, mMaxChi2PerClusterITS{1e10}, mMinNCrossedRowsOverFindableClustersTPC{1e10}, mMaxDcaXY{1e10}, mMaxDcaZ{1e10}, mRequireITSRefit{false}, mRequireTPCRefit{false}
{
}

bool TrackSelection::IsSelected(
  soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra>::iterator const& track)
{
  if (track.pt() >= mMinPt && track.pt() < mMaxPt && track.eta() >= mMinEta &&
      track.eta() < mMaxEta && track.tpcNClsFound() >= mMinNClustersTPC &&
      track.tpcNClsCrossedRows() >= mMinNCrossedRowsTPC &&
      track.tpcCrossedRowsOverFindableCls() >=
        mMinNCrossedRowsOverFindableClustersTPC &&
      (track.itsNCls() >= mMinNClustersITS) &&
      (track.itsChi2NCl() < mMaxChi2PerClusterITS) &&
      (track.tpcChi2Ncl() < mMaxChi2PerClusterTPC) &&
      (mRequireITSRefit && (track.flags() & 0x4)) &&
      (mRequireTPCRefit && (track.flags() & 0x40)) &&
      FulfillsITSHitRequirements(track.itsClusterMap())) {
    return true;
  } else {
    return false;
  }
}

bool TrackSelection::FulfillsITSHitRequirements(uint8_t itsClusterMap)
{
  constexpr uint8_t bit = 1;
  for (auto& itsRequirement : mRequiredITSHits) {
    uint8_t hits = 0;
    for (auto& requiredLayer : itsRequirement.second) {
      if (itsClusterMap & (bit << requiredLayer))
        hits++;
    }
    if ((itsRequirement.first == -1) && (hits > 0)) {
      return false; // no hits were required in specified layers
    } else if (hits < itsRequirement.first) {
      return false; // not enough hits found in specified layers
    }
  }
  return true;
};
