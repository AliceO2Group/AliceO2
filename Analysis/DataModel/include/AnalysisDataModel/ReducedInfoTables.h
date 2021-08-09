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
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//

#ifndef O2_Analysis_ReducedInfoTables_H_
#define O2_Analysis_ReducedInfoTables_H_

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "MathUtils/Utils.h"
#include <cmath>

namespace o2::aod
{

namespace dqppfilter
{
DECLARE_SOA_COLUMN(EventFilter, eventFilter, uint64_t); //! Bit-field used for the high level event triggering
}

DECLARE_SOA_TABLE(DQEventFilter, "AOD", "EVENTFILTER", //! Store event-level decisions (DQ high level triggers)
                  dqppfilter::EventFilter);

namespace reducedevent
{

// basic event information
DECLARE_SOA_COLUMN(Tag, tag, uint64_t);                   //!  Bit-field for storing event information (e.g. high level info, cut decisions)
DECLARE_SOA_COLUMN(TriggerAlias, triggerAlias, uint32_t); //!  Trigger aliases bit field
DECLARE_SOA_COLUMN(MCPosX, mcPosX, float);                //!
DECLARE_SOA_COLUMN(MCPosY, mcPosY, float);                //!
DECLARE_SOA_COLUMN(MCPosZ, mcPosZ, float);                //!
} // namespace reducedevent

DECLARE_SOA_TABLE(ReducedEvents, "AOD", "REDUCEDEVENT", //!   Main event information table
                  o2::soa::Index<>,
                  reducedevent::Tag, bc::RunNumber,
                  collision::PosX, collision::PosY, collision::PosZ, collision::NumContrib);

DECLARE_SOA_TABLE(ReducedEventsExtended, "AOD", "REEXTENDED", //!  Extended event information
                  bc::GlobalBC, bc::TriggerMask, timestamp::Timestamp, reducedevent::TriggerAlias, cent::CentV0M);

DECLARE_SOA_TABLE(ReducedEventsVtxCov, "AOD", "REVTXCOV", //!    Event vertex covariance matrix
                  collision::CovXX, collision::CovXY, collision::CovXZ,
                  collision::CovYY, collision::CovYZ, collision::CovZZ, collision::Chi2);

// TODO and NOTE: This table is just an extension of the ReducedEvents table
//       There is no explicit accounting for MC events which were not reconstructed!!!
//       However, for analysis which will require these events, a special skimming process function
//           can be constructed and the same data model could be used
DECLARE_SOA_TABLE(ReducedEventsMC, "AOD", "REMC", //!   Event level MC truth information
                  mccollision::GeneratorsID, reducedevent::MCPosX, reducedevent::MCPosY, reducedevent::MCPosZ,
                  mccollision::T, mccollision::Weight, mccollision::ImpactParameter);

using ReducedEvent = ReducedEvents::iterator;
using ReducedEventExtended = ReducedEventsExtended::iterator;
using ReducedEventVtxCov = ReducedEventsVtxCov::iterator;
using ReducedEventMC = ReducedEventsMC::iterator;

namespace reducedtrack
{
// basic track information
DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent); //!
DECLARE_SOA_COLUMN(Idx, idx, uint16_t);               //!
// ----  flags reserved for storing various information during filtering
DECLARE_SOA_COLUMN(FilteringFlags, filteringFlags, uint64_t); //!
// -----------------------------------------------------
DECLARE_SOA_COLUMN(Pt, pt, float);       //!
DECLARE_SOA_COLUMN(Eta, eta, float);     //!
DECLARE_SOA_COLUMN(Phi, phi, float);     //!
DECLARE_SOA_COLUMN(Sign, sign, int);     //!
DECLARE_SOA_COLUMN(DcaXY, dcaXY, float); //!
DECLARE_SOA_COLUMN(DcaZ, dcaZ, float);   //!
DECLARE_SOA_DYNAMIC_COLUMN(Px, px,       //!
                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, //!
                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} //namespace reducedtrack

// basic track information
DECLARE_SOA_TABLE(ReducedTracks, "AOD", "REDUCEDTRACK", //!
                  o2::soa::Index<>, reducedtrack::ReducedEventId, reducedtrack::Idx, reducedtrack::FilteringFlags,
                  reducedtrack::Pt, reducedtrack::Eta, reducedtrack::Phi, reducedtrack::Sign,
                  reducedtrack::Px<reducedtrack::Pt, reducedtrack::Phi>,
                  reducedtrack::Py<reducedtrack::Pt, reducedtrack::Phi>,
                  reducedtrack::Pz<reducedtrack::Pt, reducedtrack::Eta>,
                  reducedtrack::P<reducedtrack::Pt, reducedtrack::Eta>);

// barrel track information
DECLARE_SOA_TABLE(ReducedTracksBarrel, "AOD", "RTBARREL", //!
                  track::TPCInnerParam, track::Flags,     // tracking status flags
                  track::ITSClusterMap, track::ITSChi2NCl,
                  track::TPCNClsFindable, track::TPCNClsFindableMinusFound, track::TPCNClsFindableMinusCrossedRows,
                  track::TPCNClsShared, track::TPCChi2NCl,
                  track::TRDChi2, track::TRDPattern, track::TOFChi2, track::Length, reducedtrack::DcaXY, reducedtrack::DcaZ,
                  track::TPCNClsFound<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                  track::TPCNClsCrossedRows<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>);

// barrel covariance matrix  TODO: add all the elements required for secondary vertexing
DECLARE_SOA_TABLE(ReducedTracksBarrelCov, "AOD", "RTBARRELCOV", //!
                  track::X, track::Alpha,
                  track::Y, track::Z, track::Snp, track::Tgl, track::Signed1Pt,
                  track::CYY, track::CZY, track::CZZ, track::CSnpY, track::CSnpZ,
                  track::CSnpSnp, track::CTglY, track::CTglZ, track::CTglSnp, track::CTglTgl,
                  track::C1PtY, track::C1PtZ, track::C1PtSnp, track::C1PtTgl, track::C1Pt21Pt2);

// barrel PID information
DECLARE_SOA_TABLE(ReducedTracksBarrelPID, "AOD", "RTBARRELPID", //!
                  track::TPCSignal,
                  pidtpc::TPCNSigmaEl, pidtpc::TPCNSigmaMu,
                  pidtpc::TPCNSigmaPi, pidtpc::TPCNSigmaKa, pidtpc::TPCNSigmaPr,
                  pidtofbeta::Beta,
                  pidtof::TOFNSigmaEl, pidtof::TOFNSigmaMu,
                  pidtof::TOFNSigmaPi, pidtof::TOFNSigmaKa, pidtof::TOFNSigmaPr,
                  track::TRDSignal);

using ReducedTrack = ReducedTracks::iterator;
using ReducedTrackBarrel = ReducedTracksBarrel::iterator;
using ReducedTrackBarrelCov = ReducedTracksBarrelCov::iterator;
using ReducedTrackBarrelPID = ReducedTracksBarrelPID::iterator;

namespace reducedtrackMC
{
DECLARE_SOA_COLUMN(McMask, mcMask, uint16_t);
DECLARE_SOA_COLUMN(McReducedFlags, mcReducedFlags, uint16_t);
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(Mother0, mother0, int, "ReducedMCTracks_Mother0");       //! Track index of the first mother
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(Mother1, mother1, int, "ReducedMCTracks_Mother1");       //! Track index of the last mother
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(Daughter0, daughter0, int, "ReducedMCTracks_Daughter0"); //! Track index of the first daugther
DECLARE_SOA_SELF_INDEX_COLUMN_FULL(Daughter1, daughter1, int, "ReducedMCTracks_Daughter1"); //! Track index of the last daugther
DECLARE_SOA_COLUMN(Pt, pt, float);                                                          //!
DECLARE_SOA_COLUMN(Eta, eta, float);                                                        //!
DECLARE_SOA_COLUMN(Phi, phi, float);                                                        //!
DECLARE_SOA_COLUMN(E, e, float);                                                            //!
DECLARE_SOA_DYNAMIC_COLUMN(Px, px,                                                          //!
                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, //!
                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(Y, y, //! Particle rapidity
                           [](float pt, float eta, float e) -> float {
                             float pz = pt * std::sinh(eta);
                             if ((e - pz) > static_cast<float>(1e-7)) {
                               return 0.5f * std::log((e + pz) / (e - pz));
                             } else {
                               return -999.0f;
                             }
                           });
} // namespace reducedtrackMC
// NOTE: This table is nearly identical to the one from Framework (except that it points to the event ID, not the BC id)
//       This table contains all MC truth tracks (both barrel and muon)
DECLARE_SOA_TABLE_FULL(ReducedMCTracks, "ReducedMCTracks", "AOD", "RTMC", //!  MC track information (on disk)
                       o2::soa::Index<>, reducedtrack::ReducedEventId,
                       mcparticle::PdgCode, mcparticle::StatusCode, mcparticle::Flags,
                       reducedtrackMC::Mother0Id, reducedtrackMC::Mother1Id,
                       reducedtrackMC::Daughter0Id, reducedtrackMC::Daughter1Id, mcparticle::Weight,
                       reducedtrackMC::Pt, reducedtrackMC::Eta, reducedtrackMC::Phi, reducedtrackMC::E,
                       mcparticle::Vx, mcparticle::Vy, mcparticle::Vz, mcparticle::Vt,
                       reducedtrackMC::McReducedFlags,
                       reducedtrackMC::Px<reducedtrackMC::Pt, reducedtrackMC::Phi>,
                       reducedtrackMC::Py<reducedtrackMC::Pt, reducedtrackMC::Phi>,
                       reducedtrackMC::Pz<reducedtrackMC::Pt, reducedtrackMC::Eta>,
                       reducedtrackMC::P<reducedtrackMC::Pt, reducedtrackMC::Eta>,
                       reducedtrackMC::Y<reducedtrackMC::Pt, reducedtrackMC::Eta, reducedtrackMC::E>,
                       mcparticle::ProducedByGenerator<mcparticle::Flags>);

/*DECLARE_SOA_EXTENDED_TABLE(ReducedMCTracks, StoredReducedMCTracks, "RTMC", //! Basic derived track properties
                           aod::reducedtrackMC::Px,
                           aod::reducedtrackMC::Py,
                           aod::reducedtrackMC::Pz,
                           aod::reducedtrackMC::P,
                           aod::reducedtrackMC::Y);*/

using ReducedMCTrack = ReducedMCTracks::iterator;

namespace reducedtrackMC
{
DECLARE_SOA_INDEX_COLUMN(ReducedMCTrack, reducedMCTrack); //!
}

// NOTE: MC labels. This table has one entry for each reconstructed track (joinable with the track tables)
//          The McParticleId points to the position of the MC truth track from the ReducedTracksMC table
DECLARE_SOA_TABLE(ReducedTracksBarrelLabels, "AOD", "RTBARRELLABELS", //!
                  reducedtrackMC::ReducedMCTrackId, reducedtrackMC::McMask, reducedtrackMC::McReducedFlags);

using ReducedTrackBarrelLabel = ReducedTracksBarrelLabels::iterator;

// muon quantities
namespace reducedmuon
{
DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent);        //!
DECLARE_SOA_COLUMN(FilteringFlags, filteringFlags, uint8_t); //!
// the (pt,eta,phi,sign) will be computed in the skimming task //!
DECLARE_SOA_COLUMN(Pt, pt, float);   //!
DECLARE_SOA_COLUMN(Eta, eta, float); //!
DECLARE_SOA_COLUMN(Phi, phi, float); //!
DECLARE_SOA_COLUMN(Sign, sign, int); //!
DECLARE_SOA_DYNAMIC_COLUMN(Px, px,   //!
                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, //!
                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
DECLARE_SOA_COLUMN(RawPhi, rawPhi, float); //!
} // namespace reducedmuon

// Muon track kinematics
DECLARE_SOA_TABLE(ReducedMuons, "AOD", "RTMUON", //!
                  o2::soa::Index<>, reducedmuon::ReducedEventId, reducedmuon::FilteringFlags,
                  reducedmuon::Pt, reducedmuon::Eta, reducedmuon::Phi, reducedmuon::Sign,
                  reducedmuon::Px<reducedmuon::Pt, reducedmuon::Phi>,
                  reducedmuon::Py<reducedmuon::Pt, reducedmuon::Phi>,
                  reducedmuon::Pz<reducedmuon::Pt, reducedmuon::Eta>,
                  reducedmuon::P<reducedmuon::Pt, reducedmuon::Eta>);

// Muon track quality details
DECLARE_SOA_TABLE(ReducedMuonsExtra, "AOD", "RTMUONEXTRA", //!
                  fwdtrack::NClusters, fwdtrack::PDca, fwdtrack::RAtAbsorberEnd,
                  fwdtrack::Chi2, fwdtrack::Chi2MatchMCHMID, fwdtrack::Chi2MatchMCHMFT,
                  fwdtrack::MatchScoreMCHMFT, fwdtrack::MatchMFTTrackID, fwdtrack::MatchMCHTrackID);
// Muon covariance, TODO: the rest of the matrix should be added when needed
DECLARE_SOA_TABLE(ReducedMuonsCov, "AOD", "RTMUONCOV",
                  fwdtrack::X, fwdtrack::Y, fwdtrack::Z, reducedmuon::RawPhi, fwdtrack::Tgl, fwdtrack::Signed1Pt,
                  fwdtrack::CXX, fwdtrack::CXY, fwdtrack::CYY, fwdtrack::CPhiX, fwdtrack::CPhiY, fwdtrack::CPhiPhi,
                  fwdtrack::CTglX, fwdtrack::CTglY, fwdtrack::CTglPhi, fwdtrack::CTglTgl, fwdtrack::C1PtX,
                  fwdtrack::C1PtY, fwdtrack::C1PtPhi, fwdtrack::C1PtTgl, fwdtrack::C1Pt21Pt2);

// iterators
using ReducedMuon = ReducedMuons::iterator;
using ReducedMuonExtra = ReducedMuonsExtra::iterator;
using ReducedMuonCov = ReducedMuonsCov::iterator;

// pair information
namespace reducedpair
{
DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent); //!
DECLARE_SOA_COLUMN(Mass, mass, float);                //!
DECLARE_SOA_COLUMN(Pt, pt, float);                    //!
DECLARE_SOA_COLUMN(Eta, eta, float);                  //!
DECLARE_SOA_COLUMN(Phi, phi, float);                  //!
DECLARE_SOA_COLUMN(Sign, sign, int);                  //!
DECLARE_SOA_COLUMN(FilterMap, filterMap, uint32_t);   //!
DECLARE_SOA_DYNAMIC_COLUMN(Px, px,                    //!
                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, //!
                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} // namespace reducedpair

DECLARE_SOA_TABLE(Dileptons, "AOD", "RTDILEPTON", //!
                  reducedpair::ReducedEventId, reducedpair::Mass,
                  reducedpair::Pt, reducedpair::Eta, reducedpair::Phi, reducedpair::Sign,
                  reducedpair::FilterMap,
                  reducedpair::Px<reducedpair::Pt, reducedpair::Phi>,
                  reducedpair::Py<reducedpair::Pt, reducedpair::Phi>,
                  reducedpair::Pz<reducedpair::Pt, reducedpair::Eta>,
                  reducedpair::P<reducedpair::Pt, reducedpair::Eta>);

using Dilepton = Dileptons::iterator;

namespace v0bits
{
DECLARE_SOA_COLUMN(PIDBit, pidbit, uint8_t); //!
} // namespace v0bits

// bit information for particle species.
DECLARE_SOA_TABLE(V0Bits, "AOD", "V0BITS", //!
                  v0bits::PIDBit);

// iterators
using V0Bit = V0Bits::iterator;

} // namespace o2::aod

#endif // O2_Analysis_ReducedInfoTables_H_
