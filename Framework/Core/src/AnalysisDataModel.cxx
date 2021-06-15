// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/AnalysisDataModel.h"

namespace o2::soa
{
template struct Join<aod::BCs, aod::Timestamps>;
template struct Join<aod::Tracks, aod::TracksCov, aod::TracksExtra>;
template struct Join<aod::FwdTracks, aod::FwdTracksCov>;
template struct Join<aod::TransientV0s, aod::StoredV0s>;
template struct Join<aod::TransientCascades, aod::StoredCascades>;
template struct Join<aod::Collisions, aod::Run2MatchedSparse>;
template struct Join<aod::Collisions, aod::Run3MatchedSparse>;

template struct Join<aod::TracksExtension, aod::StoredTracks>;
} // namespace o2::soa
