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
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//

#ifndef O2_Analysis_ReducedInfoTables_H_
#define O2_Analysis_ReducedInfoTables_H_

#include "Framework/AnalysisDataModel.h"
#include "Analysis/Centrality.h"
#include "MathUtils/Utils.h"
#include <cmath>

namespace o2::aod
{
namespace reducedevent
{

// basic event information
DECLARE_SOA_COLUMN(Tag, tag, uint64_t);

} // namespace reducedevent

DECLARE_SOA_TABLE(ReducedEvents, "AOD", "REDUCEDEVENT", o2::soa::Index<>,
                  reducedevent::Tag, bc::RunNumber,
                  collision::PosX, collision::PosY, collision::PosZ, collision::NumContrib);

DECLARE_SOA_TABLE(ReducedEventsExtended, "AOD", "REEXTENDED",
                  bc::GlobalBC, bc::TriggerMask, collision::CollisionTime, cent::CentV0M);

DECLARE_SOA_TABLE(ReducedEventsVtxCov, "AOD", "REVTXCOV",
                  collision::CovXX, collision::CovXY, collision::CovXZ,
                  collision::CovYY, collision::CovYZ, collision::CovZZ, collision::Chi2);

using ReducedEvent = ReducedEvents::iterator;
using ReducedEventExtended = ReducedEventsExtended::iterator;
using ReducedEventVtxCov = ReducedEventsVtxCov::iterator;

namespace reducedtrack
{
// basic track information
DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent);
DECLARE_SOA_COLUMN(Index, index, uint16_t);
// ----  flags reserved for storing various information during filtering
DECLARE_SOA_COLUMN(FilteringFlags, filteringFlags, uint64_t);
// BIT 0: track is from MUON arm      (if not toggled then this is a barrel track)
// -----------------------------------------------------
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Charge, charge, short);
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float pt, float phi) -> float { return abs(pt) * cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float pt, float phi) -> float { return abs(pt) * sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float pt, float eta) -> float { return abs(pt) * sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(Pmom, pmom, [](float pt, float eta) -> float { return abs(pt) * cosh(eta); });

// MUON tracks extra information
DECLARE_SOA_COLUMN(MuonChi2, muonChi2, float);
DECLARE_SOA_COLUMN(MuonChi2MatchTrigger, muonChi2MatchTrigger, float);

} //namespace reducedtrack

// basic track information
DECLARE_SOA_TABLE(ReducedTracks, "AOD", "REDUCEDTRACK",
                  o2::soa::Index<>, reducedtrack::ReducedEventId, reducedtrack::Index, reducedtrack::FilteringFlags,
                  reducedtrack::Pt, reducedtrack::Eta, reducedtrack::Phi, reducedtrack::Charge,
                  reducedtrack::Px<reducedtrack::Pt, reducedtrack::Phi>,
                  reducedtrack::Py<reducedtrack::Pt, reducedtrack::Phi>,
                  reducedtrack::Pz<reducedtrack::Pt, reducedtrack::Eta>,
                  reducedtrack::Pmom<reducedtrack::Pt, reducedtrack::Eta>);

// barrel track information
DECLARE_SOA_TABLE(ReducedTracksBarrel, "AOD", "RTBARREL",
                  track::TPCInnerParam, track::Flags, // tracking status flags
                  track::ITSClusterMap, track::ITSChi2NCl,
                  track::TPCNClsFindable, track::TPCNClsFindableMinusFound, track::TPCNClsFindableMinusCrossedRows,
                  track::TPCNClsShared, track::TPCChi2NCl,
                  track::TPCSignal, track::TRDSignal, track::TOFSignal,
                  track::TRDChi2, track::TOFChi2, track::Length,
                  track::TPCNClsFound<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                  track::TPCNClsCrossedRows<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>);

// barrel covariance matrix
DECLARE_SOA_TABLE(ReducedTracksBarrelCov, "AOD", "RTBARRELCOV",
                  track::CYY, track::CZZ, track::CSnpSnp,
                  track::CTglTgl, track::C1Pt21Pt2);

// TODO: change names of these columns in the AnalysisDataModel to identify the members as muon quantities
DECLARE_SOA_TABLE(ReducedTracksMuon, "AOD", "RTMUON",
                  reducedtrack::MuonChi2, reducedtrack::MuonChi2MatchTrigger);

// iterators
using ReducedTrack = ReducedTracks::iterator;
using ReducedTrackBarrel = ReducedTracksBarrel::iterator;
using ReducedTrackBarrelCov = ReducedTracksBarrelCov::iterator;
using ReducedTrackMuon = ReducedTracksMuon::iterator;

} // namespace o2::aod

#endif // O2_Analysis_ReducedInfoTables_H_
