// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ANALYSIS_TRACKSELECTIONTABLES_H_
#define O2_ANALYSIS_TRACKSELECTIONTABLES_H_

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace track
{
DECLARE_SOA_COLUMN(DcaXY, dcaXY, float);
DECLARE_SOA_COLUMN(DcaZ, dcaZ, float);

// Columns to store track filter decisions
DECLARE_SOA_COLUMN(IsGlobalTrack, isGlobalTrack, bool);
DECLARE_SOA_COLUMN(IsGlobalTrackSDD, isGlobalTrackSDD, bool);

} // namespace track
DECLARE_SOA_TABLE(TracksExtended, "AOD", "TRACKEXTENDED", track::DcaXY,
                  track::DcaZ);

DECLARE_SOA_TABLE(TrackSelection, "AOD", "TRACKSELECTION", track::IsGlobalTrack,
                  track::IsGlobalTrackSDD);
} // namespace o2::aod

#endif // O2_ANALYSIS_TRACKSELECTIONTABLES_H_
