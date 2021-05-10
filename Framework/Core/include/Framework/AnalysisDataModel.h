// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_ANALYSISDATAMODEL_H_
#define O2_FRAMEWORK_ANALYSISDATAMODEL_H_

#include "Framework/ASoA.h"
#include "MathUtils/Utils.h"
#include <cmath>
#include "Framework/DataTypes.h"

namespace o2
{
namespace aod
{
// This is required to register SOA_TABLEs inside
// the o2::aod namespace.
DECLARE_SOA_STORE();

namespace bc
{
DECLARE_SOA_COLUMN(RunNumber, runNumber, int);          //!
DECLARE_SOA_COLUMN(GlobalBC, globalBC, uint64_t);       //! Bunch crossing number
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint64_t); //!
} // namespace bc

DECLARE_SOA_TABLE(BCs, "AOD", "BC", o2::soa::Index<>, //!
                  bc::RunNumber, bc::GlobalBC,
                  bc::TriggerMask);
using BC = BCs::iterator;

namespace timestamp
{
DECLARE_SOA_COLUMN(Timestamp, timestamp, uint64_t); //!
} // namespace timestamp

DECLARE_SOA_TABLE(Timestamps, "AOD", "TIMESTAMPS", //!
                  timestamp::Timestamp);

using BCsWithTimestamps = soa::Join<aod::BCs, aod::Timestamps>;

namespace collision
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                                  //!
DECLARE_SOA_COLUMN(PosX, posX, float);                             //! Vertex position
DECLARE_SOA_COLUMN(PosY, posY, float);                             //!
DECLARE_SOA_COLUMN(PosZ, posZ, float);                             //!
DECLARE_SOA_COLUMN(CovXX, covXX, float);                           //! Vertex covariance matrix
DECLARE_SOA_COLUMN(CovXY, covXY, float);                           //!
DECLARE_SOA_COLUMN(CovXZ, covXZ, float);                           //!
DECLARE_SOA_COLUMN(CovYY, covYY, float);                           //!
DECLARE_SOA_COLUMN(CovYZ, covYZ, float);                           //!
DECLARE_SOA_COLUMN(CovZZ, covZZ, float);                           //!
DECLARE_SOA_COLUMN(Flags, flags, uint16_t);                        //! Run2, see CollisionFlagsRun2 | Run 3, see Vertex::Flags
DECLARE_SOA_COLUMN(Chi2, chi2, float);                             //!
DECLARE_SOA_COLUMN(NumContrib, numContrib, uint16_t);              //! Number of tracks at vertex
DECLARE_SOA_COLUMN(CollisionTime, collisionTime, float);           //!
DECLARE_SOA_COLUMN(CollisionTimeRes, collisionTimeRes, float);     //!
DECLARE_SOA_COLUMN(CollisionTimeMask, collisionTimeMask, uint8_t); //! Nature of CollisionTimeRes, MSB 0 = exact range / 1 = Gaussian uncertainty
} // namespace collision

DECLARE_SOA_TABLE(Collisions, "AOD", "COLLISION", //! Time and vertex information of collision
                  o2::soa::Index<>, collision::BCId,
                  collision::PosX, collision::PosY, collision::PosZ,
                  collision::CovXX, collision::CovXY, collision::CovXZ, collision::CovYY, collision::CovYZ, collision::CovZZ,
                  collision::Flags, collision::Chi2, collision::NumContrib,
                  collision::CollisionTime, collision::CollisionTimeRes, collision::CollisionTimeMask);

using Collision = Collisions::iterator;

// NOTE Relation between Collisions and BC table
// (important for pp in case of ambigous assignment)
// A collision entry points to the entry in the BC table based on the calculated BC from the collision time
// To study other compatible triggers with the collision time, use this helper (not yet implemented :)):
// auto compatibleBCs = getCompatibleBCs(collision, BCs, /* sigma */ 3);

namespace track
{
// TRACKPAR TABLE definition
DECLARE_SOA_INDEX_COLUMN(Collision, collision);      //!
DECLARE_SOA_COLUMN(TrackType, trackType, uint8_t);   //! TODO change to TrackTypeEnum when enums are supported
DECLARE_SOA_COLUMN(X, x, float);                     //!
DECLARE_SOA_COLUMN(Alpha, alpha, float);             //!
DECLARE_SOA_COLUMN(Y, y, float);                     //!
DECLARE_SOA_COLUMN(Z, z, float);                     //!
DECLARE_SOA_COLUMN(Snp, snp, float);                 //!
DECLARE_SOA_COLUMN(Tgl, tgl, float);                 //!
DECLARE_SOA_COLUMN(Signed1Pt, signed1Pt, float);     //! (sign of charge)/Pt [c/GeV]
DECLARE_SOA_EXPRESSION_COLUMN(RawPhi, phiraw, float, //!
                              nasin(aod::track::snp) + aod::track::alpha);
// FIXME: make expression column when conditional nodes are supported in Gandiva
DECLARE_SOA_DYNAMIC_COLUMN(NormalizedPhi, phi, //!
                           [](float phi) -> float {
                             constexpr float twopi = 2.0f * static_cast<float>(M_PI);
                             if (phi < 0)
                               phi += twopi;
                             if (phi > twopi)
                               phi -= twopi;
                             return phi;
                           });
DECLARE_SOA_EXPRESSION_COLUMN(Eta, eta, float, //! Pseudo rapidity
                              -1.f * nlog(ntan(0.25f * static_cast<float>(M_PI) - 0.5f * natan(aod::track::tgl))));
DECLARE_SOA_EXPRESSION_COLUMN(Pt, pt, float, //!
                              nabs(1.f / aod::track::signed1Pt));

DECLARE_SOA_DYNAMIC_COLUMN(Sign, sign, //! Charge: positive: 1, negative: -1
                           [](float signed1Pt) -> short { return (signed1Pt > 0) ? 1 : -1; });
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, //! Momentum in x-direction [GeV/c]
                           [](float signed1Pt, float snp, float alpha) -> float {
                             auto pt = 1.f / std::abs(signed1Pt);
                             float cs, sn;
                             math_utils::sincos(alpha, sn, cs);
                             auto r = std::sqrt((1.f - snp) * (1.f + snp));
                             return pt * (r * cs - snp * sn);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //! Momentum in y-direction [GeV/c]
                           [](float signed1Pt, float snp, float alpha) -> float {
                             auto pt = 1.f / std::abs(signed1Pt);
                             float cs, sn;
                             math_utils::sincos(alpha, sn, cs);
                             auto r = std::sqrt((1.f - snp) * (1.f + snp));
                             return pt * (snp * cs + r * sn);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //! Momentum in z-direction [GeV/c]
                           [](float signed1Pt, float tgl) -> float {
                             auto pt = 1.f / std::abs(signed1Pt);
                             return pt * tgl;
                           });

DECLARE_SOA_EXPRESSION_COLUMN(P, p, float, //! Absolute momentum [Gev/c]
                              0.5f * (ntan(0.25f * static_cast<float>(M_PI) - 0.5f * natan(aod::track::tgl)) + 1.f / ntan(0.25f * static_cast<float>(M_PI) - 0.5f * natan(aod::track::tgl))) / nabs(aod::track::signed1Pt));

// TRACKPARCOV TABLE definition
DECLARE_SOA_COLUMN(SigmaY, sigmaY, float);        //!
DECLARE_SOA_COLUMN(SigmaZ, sigmaZ, float);        //!
DECLARE_SOA_COLUMN(SigmaSnp, sigmaSnp, float);    //!
DECLARE_SOA_COLUMN(SigmaTgl, sigmaTgl, float);    //!
DECLARE_SOA_COLUMN(Sigma1Pt, sigma1Pt, float);    //!
DECLARE_SOA_COLUMN(RhoZY, rhoZY, int8_t);         //!
DECLARE_SOA_COLUMN(RhoSnpY, rhoSnpY, int8_t);     //!
DECLARE_SOA_COLUMN(RhoSnpZ, rhoSnpZ, int8_t);     //!
DECLARE_SOA_COLUMN(RhoTglY, rhoTglY, int8_t);     //!
DECLARE_SOA_COLUMN(RhoTglZ, rhoTglZ, int8_t);     //!
DECLARE_SOA_COLUMN(RhoTglSnp, rhoTglSnp, int8_t); //!
DECLARE_SOA_COLUMN(Rho1PtY, rho1PtY, int8_t);     //!
DECLARE_SOA_COLUMN(Rho1PtZ, rho1PtZ, int8_t);     //!
DECLARE_SOA_COLUMN(Rho1PtSnp, rho1PtSnp, int8_t); //!
DECLARE_SOA_COLUMN(Rho1PtTgl, rho1PtTgl, int8_t); //!

DECLARE_SOA_EXPRESSION_COLUMN(CYY, cYY, float, //!
                              aod::track::sigmaY* aod::track::sigmaY);
DECLARE_SOA_EXPRESSION_COLUMN(CZY, cZY, float, //!
                              (aod::track::rhoZY / 128.f) * (aod::track::sigmaZ * aod::track::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CZZ, cZZ, float, //!
                              aod::track::sigmaZ* aod::track::sigmaZ);
DECLARE_SOA_EXPRESSION_COLUMN(CSnpY, cSnpY, float, //!
                              (aod::track::rhoSnpY / 128.f) * (aod::track::sigmaSnp * aod::track::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CSnpZ, cSnpZ, float, //!
                              (aod::track::rhoSnpZ / 128.f) * (aod::track::sigmaSnp * aod::track::sigmaZ));
DECLARE_SOA_EXPRESSION_COLUMN(CSnpSnp, cSnpSnp, float, //!
                              aod::track::sigmaSnp* aod::track::sigmaSnp);
DECLARE_SOA_EXPRESSION_COLUMN(CTglY, cTglY, float, //!
                              (aod::track::rhoTglY / 128.f) * (aod::track::sigmaTgl * aod::track::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CTglZ, cTglZ, float, //!
                              (aod::track::rhoTglZ / 128.f) * (aod::track::sigmaTgl * aod::track::sigmaZ));
DECLARE_SOA_EXPRESSION_COLUMN(CTglSnp, cTglSnp, float, //!
                              (aod::track::rhoTglSnp / 128.f) * (aod::track::sigmaTgl * aod::track::sigmaSnp));
DECLARE_SOA_EXPRESSION_COLUMN(CTglTgl, cTglTgl, float, //!
                              aod::track::sigmaTgl* aod::track::sigmaTgl);
DECLARE_SOA_EXPRESSION_COLUMN(C1PtY, c1PtY, float, //!
                              (aod::track::rho1PtY / 128.f) * (aod::track::sigma1Pt * aod::track::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtZ, c1PtZ, float, //!
                              (aod::track::rho1PtZ / 128.f) * (aod::track::sigma1Pt * aod::track::sigmaZ));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtSnp, c1PtSnp, float, //!
                              (aod::track::rho1PtSnp / 128.f) * (aod::track::sigma1Pt * aod::track::sigmaSnp));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtTgl, c1PtTgl, float, //!
                              (aod::track::rho1PtTgl / 128.f) * (aod::track::sigma1Pt * aod::track::sigmaTgl));
DECLARE_SOA_EXPRESSION_COLUMN(C1Pt21Pt2, c1Pt21Pt2, float, //!
                              aod::track::sigma1Pt* aod::track::sigma1Pt);

// TRACKEXTRA TABLE definition
DECLARE_SOA_COLUMN(TPCInnerParam, tpcInnerParam, float);                                      //!
DECLARE_SOA_COLUMN(Flags, flags, uint32_t);                                                   //!
DECLARE_SOA_COLUMN(ITSClusterMap, itsClusterMap, uint8_t);                                    //!
DECLARE_SOA_COLUMN(TPCNClsFindable, tpcNClsFindable, uint8_t);                                //!
DECLARE_SOA_COLUMN(TPCNClsFindableMinusFound, tpcNClsFindableMinusFound, int8_t);             //!
DECLARE_SOA_COLUMN(TPCNClsFindableMinusCrossedRows, tpcNClsFindableMinusCrossedRows, int8_t); //!
DECLARE_SOA_COLUMN(TPCNClsShared, tpcNClsShared, uint8_t);                                    //!
DECLARE_SOA_COLUMN(TRDPattern, trdPattern, uint8_t);                                          //!
DECLARE_SOA_COLUMN(ITSChi2NCl, itsChi2NCl, float);                                            //!
DECLARE_SOA_COLUMN(TPCChi2NCl, tpcChi2NCl, float);                                            //!
DECLARE_SOA_COLUMN(TRDChi2, trdChi2, float);                                                  //!
DECLARE_SOA_COLUMN(TOFChi2, tofChi2, float);                                                  //!
DECLARE_SOA_COLUMN(TPCSignal, tpcSignal, float);                                              //!
DECLARE_SOA_COLUMN(TRDSignal, trdSignal, float);                                              //!
DECLARE_SOA_COLUMN(TOFSignal, tofSignal, float);                                              //!
DECLARE_SOA_COLUMN(Length, length, float);                                                    //!
DECLARE_SOA_COLUMN(TOFExpMom, tofExpMom, float);                                              //!
DECLARE_SOA_COLUMN(TrackEtaEMCAL, trackEtaEmcal, float);                                      //!
DECLARE_SOA_COLUMN(TrackPhiEMCAL, trackPhiEmcal, float);                                      //!
DECLARE_SOA_DYNAMIC_COLUMN(PIDForTracking, pidForTracking,                                    //!
                           [](uint32_t flags) -> uint32_t { return flags >> 28; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCNClsFound, tpcNClsFound, //!
                           [](uint8_t tpcNClsFindable, int8_t tpcNClsFindableMinusFound) -> int16_t { return (int16_t)tpcNClsFindable - tpcNClsFindableMinusFound; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCNClsCrossedRows, tpcNClsCrossedRows, //!
                           [](uint8_t tpcNClsFindable, int8_t TPCNClsFindableMinusCrossedRows) -> int16_t { return (int16_t)tpcNClsFindable - TPCNClsFindableMinusCrossedRows; });
DECLARE_SOA_DYNAMIC_COLUMN(ITSNCls, itsNCls, //!
                           [](uint8_t itsClusterMap) -> uint8_t {
                             uint8_t itsNcls = 0;
                             constexpr uint8_t bit = 1;
                             for (int layer = 0; layer < 7; layer++) {
                               if (itsClusterMap & (bit << layer))
                                 itsNcls++;
                             }
                             return itsNcls;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(ITSNClsInnerBarrel, itsNClsInnerBarrel, //!
                           [](uint8_t itsClusterMap) -> uint8_t {
                             uint8_t itsNclsInnerBarrel = 0;
                             constexpr uint8_t bit = 1;
                             for (int layer = 0; layer < 3; layer++) {
                               if (itsClusterMap & (bit << layer))
                                 itsNclsInnerBarrel++;
                             }
                             return itsNclsInnerBarrel;
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TPCCrossedRowsOverFindableCls, tpcCrossedRowsOverFindableCls, //!
                           [](uint8_t tpcNClsFindable, int8_t tpcNClsFindableMinusCrossedRows) -> float {
                             int16_t tpcNClsCrossedRows = (int16_t)tpcNClsFindable - tpcNClsFindableMinusCrossedRows;
                             return (float)tpcNClsCrossedRows / (float)tpcNClsFindable;
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TPCFractionSharedCls, tpcFractionSharedCls, //!
                           [](uint8_t tpcNClsShared, uint8_t tpcNClsFindable, int8_t tpcNClsFindableMinusFound) -> float {
                             int16_t tpcNClsFound = (int16_t)tpcNClsFindable - tpcNClsFindableMinusFound;
                             return (float)tpcNClsShared / (float)tpcNClsFound;
                           });
} // namespace track

DECLARE_SOA_TABLE_FULL(StoredTracks, "Tracks", "AOD", "TRACK", //!
                       o2::soa::Index<>, track::CollisionId, track::TrackType,
                       track::X, track::Alpha,
                       track::Y, track::Z, track::Snp, track::Tgl,
                       track::Signed1Pt,
                       track::NormalizedPhi<track::RawPhi>,
                       track::Px<track::Signed1Pt, track::Snp, track::Alpha>,
                       track::Py<track::Signed1Pt, track::Snp, track::Alpha>,
                       track::Pz<track::Signed1Pt, track::Tgl>,
                       track::Sign<track::Signed1Pt>);

DECLARE_SOA_EXTENDED_TABLE(Tracks, StoredTracks, "TRACK", //!
                           aod::track::Pt,
                           aod::track::P,
                           aod::track::Eta,
                           aod::track::RawPhi);

DECLARE_SOA_TABLE_FULL(StoredTracksCov, "TracksCov", "AOD", "TRACKCOV",
                       track::SigmaY, track::SigmaZ, track::SigmaSnp, track::SigmaTgl, track::Sigma1Pt,
                       track::RhoZY, track::RhoSnpY, track::RhoSnpZ, track::RhoTglY, track::RhoTglZ,
                       track::RhoTglSnp, track::Rho1PtY, track::Rho1PtZ, track::Rho1PtSnp, track::Rho1PtTgl);

DECLARE_SOA_EXTENDED_TABLE(TracksCov, StoredTracksCov, "TRACKCOV", //!
                           aod::track::CYY,
                           aod::track::CZY,
                           aod::track::CZZ,
                           aod::track::CSnpY,
                           aod::track::CSnpZ,
                           aod::track::CSnpSnp,
                           aod::track::CTglY,
                           aod::track::CTglZ,
                           aod::track::CTglSnp,
                           aod::track::CTglTgl,
                           aod::track::C1PtY,
                           aod::track::C1PtZ,
                           aod::track::C1PtSnp,
                           aod::track::C1PtTgl,
                           aod::track::C1Pt21Pt2);

DECLARE_SOA_TABLE(TracksExtra, "AOD", "TRACKEXTRA", //!
                  track::TPCInnerParam, track::Flags, track::ITSClusterMap,
                  track::TPCNClsFindable, track::TPCNClsFindableMinusFound, track::TPCNClsFindableMinusCrossedRows,
                  track::TPCNClsShared, track::TRDPattern, track::ITSChi2NCl,
                  track::TPCChi2NCl, track::TRDChi2, track::TOFChi2,
                  track::TPCSignal, track::TRDSignal, track::TOFSignal, track::Length, track::TOFExpMom,
                  track::PIDForTracking<track::Flags>,
                  track::TPCNClsFound<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                  track::TPCNClsCrossedRows<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                  track::ITSNCls<track::ITSClusterMap>, track::ITSNClsInnerBarrel<track::ITSClusterMap>,
                  track::TPCCrossedRowsOverFindableCls<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                  track::TPCFractionSharedCls<track::TPCNClsShared, track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                  track::TrackEtaEMCAL, track::TrackPhiEMCAL);

using Track = Tracks::iterator;
using TrackCov = TracksCov::iterator;
using TrackExtra = TracksExtra::iterator;

using FullTracks = soa::Join<Tracks, TracksCov, TracksExtra>;
using FullTrack = FullTracks::iterator;

namespace fwdtrack
{
// FwdTracks and MFTTracks Columns definitions
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                //!
DECLARE_SOA_INDEX_COLUMN(BC, bc);                              //!
DECLARE_SOA_COLUMN(TrackType, trackType, uint8_t);             //! TODO change to ForwardTrackTypeEnum when enums are supported
DECLARE_SOA_COLUMN(X, x, float);                               //! TrackParFwd parameters: x, y, z, phi, tan(lamba), q/pt
DECLARE_SOA_COLUMN(Y, y, float);                               //!
DECLARE_SOA_COLUMN(Z, z, float);                               //!
DECLARE_SOA_COLUMN(Phi, phi, float);                           //!
DECLARE_SOA_COLUMN(Tgl, tgl, float);                           //!
DECLARE_SOA_COLUMN(Signed1Pt, signed1Pt, float);               //!
DECLARE_SOA_COLUMN(NClusters, nClusters, int8_t);              //!
DECLARE_SOA_COLUMN(Chi2, chi2, float);                         //!
DECLARE_SOA_COLUMN(PDca, pDca, float);                         //! PDca for MUONStandalone
DECLARE_SOA_COLUMN(RAtAbsorberEnd, rAtAbsorberEnd, float);     //! RAtAbsorberEnd for MUONStandalone tracks and GlobalMuonTrackstracks
DECLARE_SOA_COLUMN(Chi2MatchMCHMID, chi2MatchMCHMID, float);   //! MCH-MID Match Chi2 for MUONStandalone tracks
DECLARE_SOA_COLUMN(Chi2MatchMCHMFT, chi2MatchMCHMFT, float);   //! MCH-MFT Match Chi2 for GlobalMuonTracks
DECLARE_SOA_COLUMN(MatchScoreMCHMFT, matchScoreMCHMFT, float); //! MCH-MFT Machine Learning Matching Score for GlobalMuonTracks
DECLARE_SOA_COLUMN(MatchMFTTrackID, matchMFTTrackID, int);     //! ID of matching MFT track for GlobalMuonTrack (ints while self indexing not available)
DECLARE_SOA_COLUMN(MatchMCHTrackID, matchMCHTrackID, int);     //! ID of matching MCH track for GlobalMuonTracks  (ints while self indexing not available)

DECLARE_SOA_DYNAMIC_COLUMN(Sign, sign, //!
                           [](float signed1Pt) -> short { return (signed1Pt > 0) ? 1 : -1; });
DECLARE_SOA_EXPRESSION_COLUMN(Eta, eta, float, //!
                              -1.f * nlog(ntan(0.25f * static_cast<float>(M_PI) - 0.5f * natan(aod::fwdtrack::tgl))));
DECLARE_SOA_EXPRESSION_COLUMN(Pt, pt, float, //!
                              nabs(1.f / aod::fwdtrack::signed1Pt));
DECLARE_SOA_EXPRESSION_COLUMN(P, p, float, //!
                              0.5f * (ntan(0.25f * static_cast<float>(M_PI) - 0.5f * natan(aod::fwdtrack::tgl)) + 1.f / ntan(0.25f * static_cast<float>(M_PI) - 0.5f * natan(aod::fwdtrack::tgl))) / nabs(aod::fwdtrack::signed1Pt));
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, //!
                           [](float pt, float phi) -> float {
                             return pt * std::cos(phi);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float {
                             return pt * std::sin(phi);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float tgl) -> float {
                             return pt * tgl;
                           });

// FwdTracksCov columns definitions
DECLARE_SOA_COLUMN(SigmaX, sigmaX, float);        //!
DECLARE_SOA_COLUMN(SigmaY, sigmaY, float);        //!
DECLARE_SOA_COLUMN(SigmaPhi, sigmaPhi, float);    //!
DECLARE_SOA_COLUMN(SigmaTgl, sigmaTgl, float);    //!
DECLARE_SOA_COLUMN(Sigma1Pt, sigma1Pt, float);    //!
DECLARE_SOA_COLUMN(RhoXY, rhoXY, int8_t);         //!
DECLARE_SOA_COLUMN(RhoPhiX, rhoPhiX, int8_t);     //!
DECLARE_SOA_COLUMN(RhoPhiY, rhoPhiY, int8_t);     //!
DECLARE_SOA_COLUMN(RhoTglX, rhoTglX, int8_t);     //!
DECLARE_SOA_COLUMN(RhoTglY, rhoTglY, int8_t);     //!
DECLARE_SOA_COLUMN(RhoTglPhi, rhoTglPhi, int8_t); //!
DECLARE_SOA_COLUMN(Rho1PtX, rho1PtX, int8_t);     //!
DECLARE_SOA_COLUMN(Rho1PtY, rho1PtY, int8_t);     //!
DECLARE_SOA_COLUMN(Rho1PtPhi, rho1PtPhi, int8_t); //!
DECLARE_SOA_COLUMN(Rho1PtTgl, rho1PtTgl, int8_t); //!

DECLARE_SOA_EXPRESSION_COLUMN(CXX, cXX, float, //!
                              aod::fwdtrack::sigmaX* aod::fwdtrack::sigmaX);
DECLARE_SOA_EXPRESSION_COLUMN(CXY, cXY, float, //!
                              (aod::fwdtrack::rhoXY / 128.f) * (aod::fwdtrack::sigmaX * aod::fwdtrack::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CYY, cYY, float, //!
                              aod::fwdtrack::sigmaY* aod::fwdtrack::sigmaY);
DECLARE_SOA_EXPRESSION_COLUMN(CPhiX, cPhiX, float, //!
                              (aod::fwdtrack::rhoPhiX / 128.f) * (aod::fwdtrack::sigmaPhi * aod::fwdtrack::sigmaX));
DECLARE_SOA_EXPRESSION_COLUMN(CPhiY, cPhiY, float, //!
                              (aod::fwdtrack::rhoPhiY / 128.f) * (aod::fwdtrack::sigmaPhi * aod::fwdtrack::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CPhiPhi, cPhiPhi, float, //!
                              aod::fwdtrack::sigmaPhi* aod::fwdtrack::sigmaPhi);
DECLARE_SOA_EXPRESSION_COLUMN(CTglX, cTglX, float, //!
                              (aod::fwdtrack::rhoTglX / 128.f) * (aod::fwdtrack::sigmaTgl * aod::fwdtrack::sigmaX));
DECLARE_SOA_EXPRESSION_COLUMN(CTglY, cTglY, float, //!
                              (aod::fwdtrack::rhoTglY / 128.f) * (aod::fwdtrack::sigmaTgl * aod::fwdtrack::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(CTglPhi, cTglPhi, float, //!
                              (aod::fwdtrack::rhoTglPhi / 128.f) * (aod::fwdtrack::sigmaTgl * aod::fwdtrack::sigmaPhi));
DECLARE_SOA_EXPRESSION_COLUMN(CTglTgl, cTglTgl, float, //!
                              aod::fwdtrack::sigmaTgl* aod::fwdtrack::sigmaTgl);
DECLARE_SOA_EXPRESSION_COLUMN(C1PtY, c1PtY, float, //!
                              (aod::fwdtrack::rho1PtY / 128.f) * (aod::fwdtrack::sigma1Pt * aod::fwdtrack::sigmaY));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtX, c1PtX, float, //!
                              (aod::fwdtrack::rho1PtX / 128.f) * (aod::fwdtrack::sigma1Pt * aod::fwdtrack::sigmaX));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtPhi, c1PtPhi, float, //!
                              (aod::fwdtrack::rho1PtPhi / 128.f) * (aod::fwdtrack::sigma1Pt * aod::fwdtrack::sigmaPhi));
DECLARE_SOA_EXPRESSION_COLUMN(C1PtTgl, c1PtTgl, float, //!
                              (aod::fwdtrack::rho1PtTgl / 128.f) * (aod::fwdtrack::sigma1Pt * aod::fwdtrack::sigmaTgl));
DECLARE_SOA_EXPRESSION_COLUMN(C1Pt21Pt2, c1Pt21Pt2, float, //!
                              aod::fwdtrack::sigma1Pt* aod::fwdtrack::sigma1Pt);
} // namespace fwdtrack

// MFTStandalone tracks
DECLARE_SOA_TABLE_FULL(StoredMFTTracks, "MFTTracks", "AOD", "MFTTRACK", //!
                       o2::soa::Index<>, fwdtrack::CollisionId,
                       fwdtrack::X, fwdtrack::Y, fwdtrack::Z, fwdtrack::Phi, fwdtrack::Tgl,
                       fwdtrack::Signed1Pt, fwdtrack::NClusters,
                       fwdtrack::Px<fwdtrack::Pt, fwdtrack::Phi>,
                       fwdtrack::Py<fwdtrack::Pt, fwdtrack::Phi>,
                       fwdtrack::Pz<fwdtrack::Pt, fwdtrack::Tgl>,
                       fwdtrack::Sign<fwdtrack::Signed1Pt>, fwdtrack::Chi2);

DECLARE_SOA_EXTENDED_TABLE(MFTTracks, StoredMFTTracks, "MFTTRACK", //!
                           aod::fwdtrack::Pt,
                           aod::fwdtrack::Eta,
                           aod::fwdtrack::P);

using MFTTrack = MFTTracks::iterator;

// Tracks including MCH and/or MCH (plus optionally MFT)          //!
DECLARE_SOA_TABLE_FULL(StoredFwdTracks, "FwdTracks", "AOD", "FWDTRACK",
                       o2::soa::Index<>, fwdtrack::CollisionId, fwdtrack::BCId, fwdtrack::TrackType,
                       fwdtrack::X, fwdtrack::Y, fwdtrack::Z, fwdtrack::Phi, fwdtrack::Tgl,
                       fwdtrack::Signed1Pt, fwdtrack::NClusters, fwdtrack::PDca, fwdtrack::RAtAbsorberEnd,
                       fwdtrack::Px<fwdtrack::Pt, fwdtrack::Phi>,
                       fwdtrack::Py<fwdtrack::Pt, fwdtrack::Phi>,
                       fwdtrack::Pz<fwdtrack::Pt, fwdtrack::Tgl>,
                       fwdtrack::Sign<fwdtrack::Signed1Pt>,
                       fwdtrack::Chi2, fwdtrack::Chi2MatchMCHMID, fwdtrack::Chi2MatchMCHMFT,
                       fwdtrack::MatchScoreMCHMFT, fwdtrack::MatchMFTTrackID, fwdtrack::MatchMCHTrackID);

DECLARE_SOA_EXTENDED_TABLE(FwdTracks, StoredFwdTracks, "FWDTRACK", //!
                           aod::fwdtrack::Eta,                     // NOTE the order is different here than in MFTTracks as table extension has to be unique
                           aod::fwdtrack::Pt,
                           aod::fwdtrack::P);

DECLARE_SOA_TABLE_FULL(StoredFwdTracksCov, "FwdTracksCov", "AOD", "FWDTRACKCOV", //!
                       fwdtrack::SigmaX, fwdtrack::SigmaY, fwdtrack::SigmaPhi, fwdtrack::SigmaTgl, fwdtrack::Sigma1Pt,
                       fwdtrack::RhoXY, fwdtrack::RhoPhiY, fwdtrack::RhoPhiX, fwdtrack::RhoTglX, fwdtrack::RhoTglY,
                       fwdtrack::RhoTglPhi, fwdtrack::Rho1PtX, fwdtrack::Rho1PtY, fwdtrack::Rho1PtPhi, fwdtrack::Rho1PtTgl);

DECLARE_SOA_EXTENDED_TABLE(FwdTracksCov, StoredFwdTracksCov, "FWDTRACKCOV", //!
                           aod::fwdtrack::CXX,
                           aod::fwdtrack::CXY,
                           aod::fwdtrack::CYY,
                           aod::fwdtrack::CPhiX,
                           aod::fwdtrack::CPhiY,
                           aod::fwdtrack::CPhiPhi,
                           aod::fwdtrack::CTglX,
                           aod::fwdtrack::CTglY,
                           aod::fwdtrack::CTglPhi,
                           aod::fwdtrack::CTglTgl,
                           aod::fwdtrack::C1PtX,
                           aod::fwdtrack::C1PtY,
                           aod::fwdtrack::C1PtPhi,
                           aod::fwdtrack::C1PtTgl,
                           aod::fwdtrack::C1Pt21Pt2);

using FwdTrack = FwdTracks::iterator;
using FwdTrackCovFwd = FwdTracksCov::iterator;

using FullFwdTracks = soa::Join<FwdTracks, FwdTracksCov>;
using FullFwdTrack = FullFwdTracks::iterator;

namespace unassignedtracks
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision); //!
DECLARE_SOA_INDEX_COLUMN(Track, track);         //!
} // namespace unassignedtracks

DECLARE_SOA_TABLE(UnassignedTracks, "AOD", "UNASSIGNEDTRACK", //!
                  unassignedtracks::CollisionId, unassignedtracks::TrackId);

using UnassignedTrack = UnassignedTracks::iterator;

namespace unassignedmfttracks
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision); //!
DECLARE_SOA_INDEX_COLUMN(MFTTrack, mfttrack);   //!
} // namespace unassignedmfttracks

DECLARE_SOA_TABLE(UnassignedMFTTracks, "AOD", "UNASSIGNEDMFTTR", //!
                  unassignedmfttracks::CollisionId, unassignedmfttracks::MFTTrackId);

using UnassignedMFTTrack = UnassignedMFTTracks::iterator;

// HMPID information
namespace hmpid
{
DECLARE_SOA_INDEX_COLUMN(Track, track);                  //!
DECLARE_SOA_COLUMN(HMPIDSignal, hmpidSignal, float);     //!
DECLARE_SOA_COLUMN(HMPIDDistance, hmpidDistance, float); //!
DECLARE_SOA_COLUMN(HMPIDNPhotons, hmpidNPhotons, short); //!
DECLARE_SOA_COLUMN(HMPIDQMip, hmpidQMip, short);         //!
} // namespace hmpid

DECLARE_SOA_TABLE(HMPIDs, "AOD", "HMPID", //!
                  o2::soa::Index<>,
                  hmpid::TrackId,
                  hmpid::HMPIDSignal,
                  hmpid::HMPIDDistance,
                  hmpid::HMPIDNPhotons,
                  hmpid::HMPIDQMip);
using HMPID = HMPIDs::iterator;

namespace calo
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                    //!
DECLARE_SOA_COLUMN(CellNumber, cellNumber, int16_t); //!
DECLARE_SOA_COLUMN(Amplitude, amplitude, float);     //!
DECLARE_SOA_COLUMN(Time, time, float);               //!
DECLARE_SOA_COLUMN(CellType, cellType, int8_t);      //!
DECLARE_SOA_COLUMN(CaloType, caloType, int8_t);      //!
} // namespace calo

DECLARE_SOA_TABLE(Calos, "AOD", "CALO", calo::BCId, //!
                  calo::CellNumber, calo::Amplitude, calo::Time,
                  calo::CellType, calo::CaloType);
using Calo = Calos::iterator;

namespace calotrigger
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                      //!
DECLARE_SOA_COLUMN(FastOrAbsId, fastOrAbsId, int32_t); //!
DECLARE_SOA_COLUMN(L0Amplitude, l0Amplitude, float);   //!
DECLARE_SOA_COLUMN(L0Time, l0Time, float);             //!
DECLARE_SOA_COLUMN(L1TimeSum, l1TimeSum, int32_t);     //!
DECLARE_SOA_COLUMN(NL0Times, nl0Times, int8_t);        //!
DECLARE_SOA_COLUMN(TriggerBits, triggerBits, int32_t); //!
DECLARE_SOA_COLUMN(CaloType, caloType, int8_t);        //!
} // namespace calotrigger

DECLARE_SOA_TABLE(CaloTriggers, "AOD", "CALOTRIGGER", //!
                  calotrigger::BCId, calotrigger::FastOrAbsId,
                  calotrigger::L0Amplitude, calotrigger::L0Time,
                  calotrigger::L1TimeSum, calotrigger::NL0Times,
                  calotrigger::TriggerBits, calotrigger::CaloType);
using CaloTrigger = CaloTriggers::iterator;

namespace muon
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                                          //!
DECLARE_SOA_COLUMN(InverseBendingMomentum, inverseBendingMomentum, float); //!
DECLARE_SOA_COLUMN(ThetaX, thetaX, float);                                 //!
DECLARE_SOA_COLUMN(ThetaY, thetaY, float);                                 //!
DECLARE_SOA_COLUMN(ZMu, zMu, float);                                       //!
DECLARE_SOA_COLUMN(BendingCoor, bendingCoor, float);                       //!
DECLARE_SOA_COLUMN(NonBendingCoor, nonBendingCoor, float);                 //!
DECLARE_SOA_COLUMN(Covariances, covariances, float[15]);                   //!
DECLARE_SOA_COLUMN(Chi2, chi2, float);                                     //!
DECLARE_SOA_COLUMN(Chi2MatchTrigger, chi2MatchTrigger, float);             //!
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta,                                       //!
                           [](float inverseBendingMomentum, float thetaX, float thetaY) -> float {
                             float pz = -std::sqrt(1.0 + std::tan(thetaY) * std::tan(thetaY)) / std::abs(inverseBendingMomentum);
                             float pt = std::abs(pz) * std::sqrt(std::tan(thetaX) * std::tan(thetaX) + std::tan(thetaY) * std::tan(thetaY));
                             float eta = std::acos(pz / std::sqrt(pt * pt + pz * pz));
                             eta = std::tan(0.5 * eta);
                             if (eta > 0.0)
                               return -std::log(eta);
                             else
                               return 0.0;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi, //!
                           [](float thetaX, float thetaY) -> float {
                             float phi = std::atan2(std::tan(thetaY), std::tan(thetaX));
                             constexpr float twopi = 2.0f * static_cast<float>(M_PI);
                             return (phi >= 0.0 ? phi : phi + twopi);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(RAtAbsorberEnd, rAtAbsorberEnd, //! linear extrapolation of the coordinates of the track to the position of the end of the absorber (-505 cm)
                           [](float bendingCoor, float nonBendingCoor, float zMu, float thetaX, float thetaY) -> float {
                             float dZ = -505. - zMu;
                             float NonBendingSlope = std::tan(thetaX);
                             float BendingSlope = std::tan(thetaY);
                             float xAbs = nonBendingCoor + NonBendingSlope * dZ;
                             float yAbs = bendingCoor + BendingSlope * dZ;
                             float rAtAbsorberEnd = std::sqrt(xAbs * xAbs + yAbs * yAbs);
                             return rAtAbsorberEnd;
                           });
DECLARE_SOA_DYNAMIC_COLUMN(PDca, pDca, //! linear extrapolation of the coordinates of the track to the position of the end of the absorber (-505 cm)
                           [](float inverseBendingMomentum, float thetaX, float thetaY, float bendingCoor, float nonBendingCoor, float zMu) -> float {
                             float dca = std::sqrt(bendingCoor * bendingCoor + nonBendingCoor * nonBendingCoor + zMu * zMu);
                             float pz = -std::sqrt(1.0 + std::tan(thetaY) * std::tan(thetaY)) / std::abs(inverseBendingMomentum);
                             float pt = std::abs(pz) * std::sqrt(std::tan(thetaX) * std::tan(thetaX) + std::tan(thetaY) * std::tan(thetaY));
                             float pTot = std::sqrt(pt * pt + pz * pz);
                             float pDca = pTot * dca;
                             return pDca;
                           });
DECLARE_SOA_EXPRESSION_COLUMN(Pt, pt, float, //!
                              nsqrt(1.0f + ntan(aod::muon::thetaY) * ntan(aod::muon::thetaY)) * nsqrt(ntan(aod::muon::thetaX) * ntan(aod::muon::thetaX) + ntan(aod::muon::thetaY) * ntan(aod::muon::thetaY)) / nabs(aod::muon::inverseBendingMomentum));
DECLARE_SOA_EXPRESSION_COLUMN(Px, px, float, //!
                              -1.0f * ntan(aod::muon::thetaX) * nsqrt(1.0f + ntan(aod::muon::thetaY) * ntan(aod::muon::thetaY)) / nabs(aod::muon::inverseBendingMomentum));
DECLARE_SOA_EXPRESSION_COLUMN(Py, py, float, //!
                              -1.0f * ntan(aod::muon::thetaY) * nsqrt(1.0f + ntan(aod::muon::thetaY) * ntan(aod::muon::thetaY)) / nabs(aod::muon::inverseBendingMomentum));
DECLARE_SOA_EXPRESSION_COLUMN(Pz, pz, float, //!
                              -1.0f * nsqrt(1.0f + ntan(aod::muon::thetaY) * ntan(aod::muon::thetaY)) / nabs(aod::muon::inverseBendingMomentum));
DECLARE_SOA_DYNAMIC_COLUMN(Sign, sign, //!
                           [](float inverseBendingMomentum) -> short { return (inverseBendingMomentum > 0.0f) ? 1 : -1; });
} // namespace muon

DECLARE_SOA_TABLE_FULL(StoredMuons, "Muons", "AOD", "MUON", //!
                       muon::BCId, muon::InverseBendingMomentum,
                       muon::ThetaX, muon::ThetaY, muon::ZMu,
                       muon::BendingCoor, muon::NonBendingCoor,
                       muon::Covariances, muon::Chi2, muon::Chi2MatchTrigger,
                       muon::Eta<muon::InverseBendingMomentum, muon::ThetaX, muon::ThetaY>,
                       muon::Phi<muon::ThetaX, muon::ThetaY>,
                       muon::RAtAbsorberEnd<muon::BendingCoor, muon::NonBendingCoor, muon::ThetaX, muon::ThetaY, muon::ZMu>,
                       muon::PDca<muon::InverseBendingMomentum, muon::ThetaX, muon::ThetaY, muon::BendingCoor, muon::NonBendingCoor, muon::ZMu>,
                       muon::Sign<muon::InverseBendingMomentum>);

DECLARE_SOA_EXTENDED_TABLE(Muons, StoredMuons, "MUON", //!
                           aod::muon::Pt,
                           aod::muon::Px,
                           aod::muon::Py,
                           aod::muon::Pz);

using Muon = Muons::iterator;

// NOTE for now muon tracks are uniquely assigned to a BC / GlobalBC assuming they contain an MID hit. Discussion on tracks without MID hit is ongoing.

namespace muoncluster
{
DECLARE_SOA_INDEX_COLUMN_FULL(Track, track, int, Muons, ""); //! points to a muon track in the Muon table
DECLARE_SOA_COLUMN(X, x, float);                             //!
DECLARE_SOA_COLUMN(Y, y, float);                             //!
DECLARE_SOA_COLUMN(Z, z, float);                             //!
DECLARE_SOA_COLUMN(ErrX, errX, float);                       //!
DECLARE_SOA_COLUMN(ErrY, errY, float);                       //!
DECLARE_SOA_COLUMN(Charge, charge, float);                   //!
DECLARE_SOA_COLUMN(Chi2, chi2, float);                       //!
} // namespace muoncluster

DECLARE_SOA_TABLE(MuonClusters, "AOD", "MUONCLUSTER", //!
                  muoncluster::TrackId,
                  muoncluster::X, muoncluster::Y, muoncluster::Z,
                  muoncluster::ErrX, muoncluster::ErrY,
                  muoncluster::Charge, muoncluster::Chi2);

using MuonCluster = MuonClusters::iterator;

namespace zdc
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                               //!
DECLARE_SOA_COLUMN(EnergyZEM1, energyZEM1, float);              //!
DECLARE_SOA_COLUMN(EnergyZEM2, energyZEM2, float);              //!
DECLARE_SOA_COLUMN(EnergyCommonZNA, energyCommonZNA, float);    //!
DECLARE_SOA_COLUMN(EnergyCommonZNC, energyCommonZNC, float);    //!
DECLARE_SOA_COLUMN(EnergyCommonZPA, energyCommonZPA, float);    //!
DECLARE_SOA_COLUMN(EnergyCommonZPC, energyCommonZPC, float);    //!
DECLARE_SOA_COLUMN(EnergySectorZNA, energySectorZNA, float[4]); //!
DECLARE_SOA_COLUMN(EnergySectorZNC, energySectorZNC, float[4]); //!
DECLARE_SOA_COLUMN(EnergySectorZPA, energySectorZPA, float[4]); //!
DECLARE_SOA_COLUMN(EnergySectorZPC, energySectorZPC, float[4]); //!
DECLARE_SOA_COLUMN(TimeZEM1, timeZEM1, float);                  //!
DECLARE_SOA_COLUMN(TimeZEM2, timeZEM2, float);                  //!
DECLARE_SOA_COLUMN(TimeZNA, timeZNA, float);                    //!
DECLARE_SOA_COLUMN(TimeZNC, timeZNC, float);                    //!
DECLARE_SOA_COLUMN(TimeZPA, timeZPA, float);                    //!
DECLARE_SOA_COLUMN(TimeZPC, timeZPC, float);                    //!
} // namespace zdc

DECLARE_SOA_TABLE(Zdcs, "AOD", "ZDC", //!
                  o2::soa::Index<>, zdc::BCId, zdc::EnergyZEM1, zdc::EnergyZEM2,
                  zdc::EnergyCommonZNA, zdc::EnergyCommonZNC, zdc::EnergyCommonZPA, zdc::EnergyCommonZPC,
                  zdc::EnergySectorZNA, zdc::EnergySectorZNC, zdc::EnergySectorZPA, zdc::EnergySectorZPC,
                  zdc::TimeZEM1, zdc::TimeZEM2, zdc::TimeZNA, zdc::TimeZNC, zdc::TimeZPA, zdc::TimeZPC);
using Zdc = Zdcs::iterator;

namespace fv0a
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                      //!
DECLARE_SOA_COLUMN(Amplitude, amplitude, float[48]);   //!
DECLARE_SOA_COLUMN(Time, time, float);                 //!
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint8_t); //!
} // namespace fv0a

DECLARE_SOA_TABLE(FV0As, "AOD", "FV0A", //!
                  o2::soa::Index<>, fv0a::BCId, fv0a::Amplitude, fv0a::Time, fv0a::TriggerMask);
using FV0A = FV0As::iterator;

// V0C table for Run2 only
namespace fv0c
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                    //!
DECLARE_SOA_COLUMN(Amplitude, amplitude, float[32]); //!
DECLARE_SOA_COLUMN(Time, time, float);               //!
} // namespace fv0c

DECLARE_SOA_TABLE(FV0Cs, "AOD", "FV0C", //!
                  o2::soa::Index<>, fv0c::BCId, fv0c::Amplitude, fv0c::Time);
using FV0C = FV0Cs::iterator;

namespace ft0
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                       //!
DECLARE_SOA_COLUMN(AmplitudeA, amplitudeA, float[96]);  //!
DECLARE_SOA_COLUMN(AmplitudeC, amplitudeC, float[112]); //!
DECLARE_SOA_COLUMN(TimeA, timeA, float);                //!
DECLARE_SOA_COLUMN(TimeC, timeC, float);                //!
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint8_t);  //!
} // namespace ft0

DECLARE_SOA_TABLE(FT0s, "AOD", "FT0", //!
                  o2::soa::Index<>, ft0::BCId,
                  ft0::AmplitudeA, ft0::AmplitudeC, ft0::TimeA, ft0::TimeC,
                  ft0::TriggerMask);
using FT0 = FT0s::iterator;

namespace fdd
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                      //!
DECLARE_SOA_COLUMN(AmplitudeA, amplitudeA, float[4]);  //!
DECLARE_SOA_COLUMN(AmplitudeC, amplitudeC, float[4]);  //!
DECLARE_SOA_COLUMN(TimeA, timeA, float);               //!
DECLARE_SOA_COLUMN(TimeC, timeC, float);               //!
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint8_t); //!
} // namespace fdd

DECLARE_SOA_TABLE(FDDs, "AOD", "FDD", //!
                  o2::soa::Index<>, fdd::BCId,
                  fdd::AmplitudeA, fdd::AmplitudeC,
                  fdd::TimeA, fdd::TimeC,
                  fdd::TriggerMask);
using FDD = FDDs::iterator;

namespace v0
{
DECLARE_SOA_INDEX_COLUMN_FULL(PosTrack, posTrack, int, Tracks, "_Pos"); //!
DECLARE_SOA_INDEX_COLUMN_FULL(NegTrack, negTrack, int, Tracks, "_Neg"); //!
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
} // namespace v0

DECLARE_SOA_TABLE(StoredV0s, "AOD", "V0", //!
                  o2::soa::Index<>,
                  v0::PosTrackId, v0::NegTrackId);
DECLARE_SOA_TABLE(TransientV0s, "AOD", "V0INDEX", //!
                  v0::CollisionId);

using V0s = soa::Join<TransientV0s, StoredV0s>;
using V0 = V0s::iterator;

namespace cascade
{
DECLARE_SOA_INDEX_COLUMN(V0, v0);                                   //!
DECLARE_SOA_INDEX_COLUMN_FULL(Bachelor, bachelor, int, Tracks, ""); //!
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                     //!
} // namespace cascade

DECLARE_SOA_TABLE(StoredCascades, "AOD", "CASCADE", //!
                  o2::soa::Index<>, cascade::V0Id, cascade::BachelorId);
DECLARE_SOA_TABLE(TransientCascades, "AOD", "CASCADEINDEX", //!
                  cascade::CollisionId);

using Cascades = soa::Join<TransientCascades, StoredCascades>;
using Cascade = Cascades::iterator;

// ---- Run 2 tables ----
namespace run2
{
DECLARE_SOA_COLUMN(EventCuts, eventCuts, uint32_t);                   //!
DECLARE_SOA_COLUMN(TriggerMaskNext50, triggerMaskNext50, uint64_t);   //!
DECLARE_SOA_COLUMN(L0TriggerInputMask, l0TriggerInputMask, uint32_t); //!
DECLARE_SOA_COLUMN(SPDClustersL0, spdClustersL0, uint16_t);           //!
DECLARE_SOA_COLUMN(SPDClustersL1, spdClustersL1, uint16_t);           //!
DECLARE_SOA_COLUMN(SPDFiredChipsL0, spdFiredChipsL0, uint16_t);       //!
DECLARE_SOA_COLUMN(SPDFiredChipsL1, spdFiredChipsL1, uint16_t);       //!
DECLARE_SOA_COLUMN(SPDFiredFastOrL0, spdFiredFastOrL0, uint16_t);     //!
DECLARE_SOA_COLUMN(SPDFiredFastOrL1, spdFiredFastOrL1, uint16_t);     //!
DECLARE_SOA_COLUMN(V0TriggerChargeA, v0TriggerChargeA, uint16_t);     //!
DECLARE_SOA_COLUMN(V0TriggerChargeC, v0TriggerChargeC, uint16_t);     //!
} // namespace run2

DECLARE_SOA_TABLE(Run2BCInfos, "AOD", "RUN2BCINFO", run2::EventCuts, //!
                  run2::TriggerMaskNext50, run2::L0TriggerInputMask,
                  run2::SPDClustersL0, run2::SPDClustersL1,
                  run2::SPDFiredChipsL0, run2::SPDFiredChipsL1,
                  run2::SPDFiredFastOrL0, run2::SPDFiredFastOrL1,
                  run2::V0TriggerChargeA, run2::V0TriggerChargeC);
using Run2BCInfo = Run2BCInfos::iterator;

// ---- MC tables ----
namespace mccollision
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);                            //!
DECLARE_SOA_COLUMN(GeneratorsID, generatorsID, short);       //!
DECLARE_SOA_COLUMN(PosX, posX, float);                       //!
DECLARE_SOA_COLUMN(PosY, posY, float);                       //!
DECLARE_SOA_COLUMN(PosZ, posZ, float);                       //!
DECLARE_SOA_COLUMN(T, t, float);                             //!
DECLARE_SOA_COLUMN(Weight, weight, float);                   //!
DECLARE_SOA_COLUMN(ImpactParameter, impactParameter, float); //!
} // namespace mccollision

DECLARE_SOA_TABLE(McCollisions, "AOD", "MCCOLLISION", //!
                  o2::soa::Index<>, mccollision::BCId,
                  mccollision::GeneratorsID,
                  mccollision::PosX, mccollision::PosY, mccollision::PosZ,
                  mccollision::T, mccollision::Weight,
                  mccollision::ImpactParameter);
using McCollision = McCollisions::iterator;

namespace mcparticle
{
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision); //!
DECLARE_SOA_COLUMN(PdgCode, pdgCode, int);          //!
DECLARE_SOA_COLUMN(StatusCode, statusCode, int);    //!
DECLARE_SOA_COLUMN(Flags, flags, uint8_t);          //!
DECLARE_SOA_COLUMN(Mother0, mother0, int);          //!
DECLARE_SOA_COLUMN(Mother1, mother1, int);          //!
DECLARE_SOA_COLUMN(Daughter0, daughter0, int);      //!
DECLARE_SOA_COLUMN(Daughter1, daughter1, int);      //!
DECLARE_SOA_COLUMN(Weight, weight, float);          //!
DECLARE_SOA_COLUMN(Px, px, float);                  //!
DECLARE_SOA_COLUMN(Py, py, float);                  //!
DECLARE_SOA_COLUMN(Pz, pz, float);                  //!
DECLARE_SOA_COLUMN(E, e, float);                    //!
DECLARE_SOA_COLUMN(Vx, vx, float);                  //!
DECLARE_SOA_COLUMN(Vy, vy, float);                  //!
DECLARE_SOA_COLUMN(Vz, vz, float);                  //!
DECLARE_SOA_COLUMN(Vt, vt, float);                  //!
DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi,                //!
                           [](float px, float py) -> float { return static_cast<float>(M_PI) + std::atan2(-py, -px); });
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, //!
                           [](float px, float py, float pz) -> float { return 0.5f * std::log((std::sqrt(px * px + py * py + pz * pz) + pz) / (std::sqrt(px * px + py * py + pz * pz) - pz)); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, //!
                           [](float px, float py) -> float { return std::sqrt(px * px + py * py); });
DECLARE_SOA_DYNAMIC_COLUMN(ProducedByGenerator, producedByGenerator, //!
                           [](uint8_t flags) -> bool { return (flags & 0x1) == 0x0; });
} // namespace mcparticle

DECLARE_SOA_TABLE(McParticles, "AOD", "MCPARTICLE", //!
                  o2::soa::Index<>, mcparticle::McCollisionId,
                  mcparticle::PdgCode, mcparticle::StatusCode, mcparticle::Flags,
                  mcparticle::Mother0, mcparticle::Mother1,
                  mcparticle::Daughter0, mcparticle::Daughter1, mcparticle::Weight,
                  mcparticle::Px, mcparticle::Py, mcparticle::Pz, mcparticle::E,
                  mcparticle::Vx, mcparticle::Vy, mcparticle::Vz, mcparticle::Vt,
                  mcparticle::Phi<mcparticle::Px, mcparticle::Py>,
                  mcparticle::Eta<mcparticle::Px, mcparticle::Py, mcparticle::Pz>,
                  mcparticle::Pt<mcparticle::Px, mcparticle::Py>,
                  mcparticle::ProducedByGenerator<mcparticle::Flags>);
using McParticle = McParticles::iterator;

namespace mctracklabel
{
DECLARE_SOA_INDEX_COLUMN(McParticle, mcParticle); //!
DECLARE_SOA_COLUMN(McMask, mcMask, uint16_t);     //!
/// Bit mask to indicate detector mismatches (bit ON means mismatch)
/// Bit 0-6: mismatch at ITS layer
/// Bit 7-9: # of TPC mismatches in the ranges 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, >64
/// Bit 10: TRD, bit 11: TOF, bit 15: indicates negative label
} // namespace mctracklabel

DECLARE_SOA_TABLE(McTrackLabels, "AOD", "MCTRACKLABEL", //!
                  mctracklabel::McParticleId, mctracklabel::McMask);
using McTrackLabel = McTrackLabels::iterator;

namespace mccalolabel
{
DECLARE_SOA_INDEX_COLUMN(McParticle, mcParticle); //!
DECLARE_SOA_COLUMN(McMask, mcMask, uint16_t);     //!
/// Bit mask to indicate detector mismatches (bit ON means mismatch)
/// Bit 15: indicates negative label
} // namespace mccalolabel

DECLARE_SOA_TABLE(McCaloLabels, "AOD", "MCCALOLABEL", //!
                  mccalolabel::McParticleId, mccalolabel::McMask);
using McCaloLabel = McCaloLabels::iterator;

namespace mccollisionlabel
{
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision); //!
DECLARE_SOA_COLUMN(McMask, mcMask, uint16_t);       //!
/// Bit mask to indicate collision mismatches (bit ON means mismatch)
/// Bit 15: indicates negative label
} // namespace mccollisionlabel

DECLARE_SOA_TABLE(McCollisionLabels, "AOD", "MCCOLLISLABEL", //!
                  mccollisionlabel::McCollisionId, mccollisionlabel::McMask);
using McCollisionLabel = McCollisionLabels::iterator;

// --- Matching between collisions and other tables through BC ---

namespace indices
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision); //!
DECLARE_SOA_INDEX_COLUMN(BC, bc);               //!
DECLARE_SOA_INDEX_COLUMN(Zdc, zdc);             //!
DECLARE_SOA_INDEX_COLUMN(FV0A, fv0a);           //!
DECLARE_SOA_INDEX_COLUMN(FV0C, fv0c);           //!
DECLARE_SOA_INDEX_COLUMN(FT0, ft0);             //!
DECLARE_SOA_INDEX_COLUMN(FDD, fdd);             //!
} // namespace indices

// First entry: Collision
#define INDEX_LIST_RUN2 indices::CollisionId, indices::ZdcId, indices::BCId, indices::FT0Id, indices::FV0AId, indices::FV0CId, indices::FDDId
DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(Run2MatchedExclusive, BCs, "MA_RN2_EX", INDEX_LIST_RUN2); //!
DECLARE_SOA_INDEX_TABLE(Run2MatchedSparse, BCs, "MA_RN2_SP", INDEX_LIST_RUN2);              //!

#define INDEX_LIST_RUN3 indices::CollisionId, indices::ZdcId, indices::BCId, indices::FT0Id, indices::FV0AId, indices::FDDId
DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(Run3MatchedExclusive, BCs, "MA_RN3_EX", INDEX_LIST_RUN3); //!
DECLARE_SOA_INDEX_TABLE(Run3MatchedSparse, BCs, "MA_RN3_SP", INDEX_LIST_RUN3);              //!

// First entry: BC
DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(MatchedBCCollisionsExclusive, BCs, "MA_BCCOL_EX", //!
                                  indices::BCId, indices::CollisionId);
DECLARE_SOA_INDEX_TABLE(MatchedBCCollisionsSparse, BCs, "MA_BCCOL_SP", //!
                        indices::BCId, indices::CollisionId);

DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(Run3MatchedToBCExclusive, BCs, "MA_RN3_BC_EX", //!
                                  indices::BCId, indices::ZdcId, indices::FT0Id, indices::FV0AId, indices::FDDId);
DECLARE_SOA_INDEX_TABLE(Run3MatchedToBCSparse, BCs, "MA_RN3_BC_SP", //!
                        indices::BCId, indices::ZdcId, indices::FT0Id, indices::FV0AId, indices::FDDId);

DECLARE_SOA_INDEX_TABLE(Run2MatchedToBCSparse, BCs, "MA_RN2_BC_SP", //!
                        indices::BCId, indices::ZdcId, indices::FT0Id, indices::FV0AId, indices::FV0CId, indices::FDDId);

// Joins with collisions (only for sparse ones)
// NOTE: index table needs to be always last argument
using CollisionMatchedRun2Sparse = soa::Join<Collisions, Run2MatchedSparse>::iterator;
using CollisionMatchedRun3Sparse = soa::Join<Collisions, Run3MatchedSparse>::iterator;

} // namespace aod

} // namespace o2
#endif // O2_FRAMEWORK_ANALYSISDATAMODEL_H_
