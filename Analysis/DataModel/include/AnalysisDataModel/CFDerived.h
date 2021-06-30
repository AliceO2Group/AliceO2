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
#ifndef O2_ANALYSIS_CFDERIVED_H
#define O2_ANALYSIS_CFDERIVED_H

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/Centrality.h"

namespace o2::aod
{
DECLARE_SOA_TABLE(CFCollisions, "AOD", "CFCOLLISION", //!
                  o2::soa::Index<>,
                  bc::RunNumber, collision::PosZ,
                  cent::CentV0M, timestamp::Timestamp);
using CFCollision = CFCollisions::iterator;

namespace cftrack
{
DECLARE_SOA_INDEX_COLUMN(CFCollision, cfCollision); //!
DECLARE_SOA_COLUMN(Pt, pt, float);                  //!
DECLARE_SOA_COLUMN(Eta, eta, float);                //!
DECLARE_SOA_COLUMN(Phi, phi, float);                //!
DECLARE_SOA_COLUMN(Sign, sign, int8_t);             //!
} // namespace cftrack
DECLARE_SOA_TABLE(CFTracks, "AOD", "CFTRACK", //!
                  o2::soa::Index<>,
                  cftrack::CFCollisionId,
                  cftrack::Pt, cftrack::Eta, cftrack::Phi,
                  cftrack::Sign, track::TrackType);
using CFTrack = CFTracks::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_CFDERIVED_H
