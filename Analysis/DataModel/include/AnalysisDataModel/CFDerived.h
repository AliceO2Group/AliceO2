// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
DECLARE_SOA_TABLE(CFCollisions, "AOD", "CFCOLLISION", o2::soa::Index<>,
                  o2::aod::bc::RunNumber, o2::aod::collision::PosZ, o2::aod::cent::CentV0M);
using CFCollision = CFCollisions::iterator;

namespace cftrack
{
DECLARE_SOA_INDEX_COLUMN(CFCollision, cfCollision);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Charge, charge, int8_t);
} // namespace cftrack
DECLARE_SOA_TABLE(CFTracks, "AOD", "CFTRACK", o2::soa::Index<>,
                  cftrack::CFCollisionId,
                  cftrack::Pt, cftrack::Eta, cftrack::Phi,
                  cftrack::Charge, track::TrackType);
using CFTrack = CFTracks::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_CFDERIVED_H
