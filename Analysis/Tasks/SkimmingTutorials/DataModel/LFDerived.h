// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_LFDERIVED_H
#define O2_ANALYSIS_LFDERIVED_H

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
//DECLARE_SOA_TABLE(LFCollisions, "AOD", "LFCOLLISION", o2::soa::Index<>,
//               o2::aod::bc::RunNumber, o2::aod::collision::PosZ);
//using LFCollision = LFCollisions::iterator;

namespace lftrack
{
//DECLARE_SOA_INDEX_COLUMN(LFCollision, lfCollision);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(P, p, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(TPCNSigma, tpcNSigma, float[9]);
} // namespace lftrack
DECLARE_SOA_TABLE(LFTracks, "AOD", "LFTRACK", o2::soa::Index<>,
                  //lftrack::LFCollisionId,
                  lftrack::Pt, lftrack::P, lftrack::Eta,
                  lftrack::TPCNSigma);
using LFTrack = LFTracks::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_LFDERIVED_H
