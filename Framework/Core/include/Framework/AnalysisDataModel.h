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
DECLARE_SOA_COLUMN(RunNumber, runNumber, int);
DECLARE_SOA_COLUMN(GlobalBC, globalBC, uint64_t);
DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint64_t);
} // namespace bc

DECLARE_SOA_TABLE(BCs, "AOD", "BC", o2::soa::Index<>,
                  bc::RunNumber, bc::GlobalBC,
                  bc::TriggerMask);
using BC = BCs::iterator;

namespace collision
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(PosX, posX, float);
DECLARE_SOA_COLUMN(PosY, posY, float);
DECLARE_SOA_COLUMN(PosZ, posZ, float);
DECLARE_SOA_COLUMN(CovXX, covXX, float);
DECLARE_SOA_COLUMN(CovXY, covXY, float);
DECLARE_SOA_COLUMN(CovXZ, covXZ, float);
DECLARE_SOA_COLUMN(CovYY, covYY, float);
DECLARE_SOA_COLUMN(CovYZ, covYZ, float);
DECLARE_SOA_COLUMN(CovZZ, covZZ, float);
DECLARE_SOA_COLUMN(Chi2, chi2, float);
DECLARE_SOA_COLUMN(NumContrib, numContrib, uint32_t);
DECLARE_SOA_COLUMN(CollisionTime, collisionTime, float);
DECLARE_SOA_COLUMN(CollisionTimeRes, collisionTimeRes, float);
DECLARE_SOA_COLUMN(CollisionTimeMask, collisionTimeMask, uint8_t); // TODO put nature of CollisionTimeRes here, e.g. MSB 0 = exact range / 1 = Gaussian uncertainty
} // namespace collision

DECLARE_SOA_TABLE(Collisions, "AOD", "COLLISION", o2::soa::Index<>, collision::BCId, collision::PosX, collision::PosY, collision::PosZ, collision::CovXX, collision::CovXY, collision::CovXZ, collision::CovYY, collision::CovYZ, collision::CovZZ, collision::Chi2, collision::NumContrib, collision::CollisionTime, collision::CollisionTimeRes, collision::CollisionTimeMask);

using Collision = Collisions::iterator;

// NOTE Relation between Collisions and BC table
// (important for pp in case of ambigous assignment)
// A collision entry points to the entry in the BC table based on the calculated BC from the collision time
// To study other compatible triggers with the collision time, use this helper (not yet implemented :)):
// auto compatibleBCs = getCompatibleBCs(collision, BCs, /* sigma */ 3);

namespace track
{
// TRACKPAR TABLE definition
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(TrackType, trackType, uint8_t); // TODO change to TrackTypeEnum when enums are supported
DECLARE_SOA_COLUMN(X, x, float);
DECLARE_SOA_COLUMN(Alpha, alpha, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(Z, z, float);
DECLARE_SOA_COLUMN(Snp, snp, float);
DECLARE_SOA_COLUMN(Tgl, tgl, float);
DECLARE_SOA_COLUMN(Signed1Pt, signed1Pt, float);
DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi, [](float snp, float alpha) -> float { return asinf(snp) + alpha + static_cast<float>(M_PI); });
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, [](float tgl) -> float { return std::log(std::tan(0.25f * static_cast<float>(M_PI) - 0.5f * std::atan(tgl))); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, [](float signed1Pt) -> float { return std::abs(1.0f / signed1Pt); });
DECLARE_SOA_DYNAMIC_COLUMN(Charge, charge, [](float signed1Pt) -> short { return (signed1Pt > 0) ? 1 : -1; });
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float signed1Pt, float snp, float alpha) -> float {
  auto pt = 1.f / std::abs(signed1Pt);
  float cs, sn;
  utils::sincosf(alpha, sn, cs);
  auto r = std::sqrt((1.f - snp) * (1.f + snp));
  return pt * (r * cs - snp * sn);
});
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float signed1Pt, float snp, float alpha) -> float {
  auto pt = 1.f / std::abs(signed1Pt);
  float cs, sn;
  utils::sincosf(alpha, sn, cs);
  auto r = std::sqrt((1.f - snp) * (1.f + snp));
  return pt * (snp * cs + r * sn);
});
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float signed1Pt, float tgl) -> float {
  auto pt = 1.f / std::abs(signed1Pt);
  return pt * tgl;
});
DECLARE_SOA_DYNAMIC_COLUMN(P, p, [](float signed1Pt, float tgl) -> float {
  return std::sqrt(1.f + tgl * tgl) / std::abs(signed1Pt);
});
DECLARE_SOA_DYNAMIC_COLUMN(P2, p2, [](float signed1Pt, float tgl) -> float {
  return (1.f + tgl * tgl) / (signed1Pt * signed1Pt);
});

// TRACKPARCOV TABLE definition
DECLARE_SOA_COLUMN(CYY, cYY, float);
DECLARE_SOA_COLUMN(CZY, cZY, float);
DECLARE_SOA_COLUMN(CZZ, cZZ, float);
DECLARE_SOA_COLUMN(CSnpY, cSnpY, float);
DECLARE_SOA_COLUMN(CSnpZ, cSnpZ, float);
DECLARE_SOA_COLUMN(CSnpSnp, cSnpSnp, float);
DECLARE_SOA_COLUMN(CTglY, cTglY, float);
DECLARE_SOA_COLUMN(CTglZ, cTglZ, float);
DECLARE_SOA_COLUMN(CTglSnp, cTglSnp, float);
DECLARE_SOA_COLUMN(CTglTgl, cTglTgl, float);
DECLARE_SOA_COLUMN(C1PtY, c1PtY, float);
DECLARE_SOA_COLUMN(C1PtZ, c1PtZ, float);
DECLARE_SOA_COLUMN(C1PtSnp, c1PtSnp, float);
DECLARE_SOA_COLUMN(C1PtTgl, c1PtTgl, float);
DECLARE_SOA_COLUMN(C1Pt21Pt2, c1Pt21Pt2, float);

// TRACKEXTRA TABLE definition
DECLARE_SOA_COLUMN(TPCInnerParam, tpcInnerParam, float);
DECLARE_SOA_COLUMN(Flags, flags, uint64_t);
DECLARE_SOA_COLUMN(ITSClusterMap, itsClusterMap, uint8_t);
DECLARE_SOA_COLUMN(TPCNClsFindable, tpcNClsFindable, uint8_t);
DECLARE_SOA_COLUMN(TPCNClsFindableMinusFound, tpcNClsFindableMinusFound, int8_t);
DECLARE_SOA_COLUMN(TPCNClsFindableMinusCrossedRows, tpcNClsFindableMinusCrossedRows, int8_t);
DECLARE_SOA_COLUMN(TPCNClsShared, tpcNClsShared, uint8_t);
DECLARE_SOA_COLUMN(TRDNTracklets, trdNTracklets, uint8_t);
DECLARE_SOA_COLUMN(ITSChi2NCl, itsChi2NCl, float);
DECLARE_SOA_COLUMN(TPCChi2NCl, tpcChi2NCl, float);
DECLARE_SOA_COLUMN(TRDChi2, trdChi2, float);
DECLARE_SOA_COLUMN(TOFChi2, tofChi2, float);
DECLARE_SOA_COLUMN(TPCSignal, tpcSignal, float);
DECLARE_SOA_COLUMN(TRDSignal, trdSignal, float);
DECLARE_SOA_COLUMN(TOFSignal, tofSignal, float);
DECLARE_SOA_COLUMN(Length, length, float);
DECLARE_SOA_DYNAMIC_COLUMN(TPCNClsFound, tpcNClsFound, [](uint8_t tpcNClsFindable, uint8_t tpcNClsFindableMinusFound) -> int16_t { return tpcNClsFindable - tpcNClsFindableMinusFound; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCNClsCrossedRows, tpcNClsCrossedRows, [](uint8_t tpcNClsFindable, uint8_t TPCNClsFindableMinusCrossedRows) -> int16_t { return tpcNClsFindable - TPCNClsFindableMinusCrossedRows; });
DECLARE_SOA_DYNAMIC_COLUMN(ITSNCls, itsNCls, [](uint8_t itsClusterMap) -> uint8_t {
  uint8_t itsNcls = 0;
  constexpr uint8_t bit = 1;
  for (int layer = 0; layer < 7; layer++)
    if (itsClusterMap & (bit << layer))
      itsNcls++;
  return itsNcls;
});

DECLARE_SOA_DYNAMIC_COLUMN(TPCCrossedRowsOverFindableCls, tpcCrossedRowsOverFindableCls,
                           [](uint8_t tpcNClsFindable, uint8_t tpcNClsFindableMinusCrossedRows) -> float {
                             // FIXME: use int16 tpcNClsCrossedRows from dynamic column as argument
                             int16_t tpcNClsCrossedRows = tpcNClsFindable - tpcNClsFindableMinusCrossedRows;
                             return (float)tpcNClsCrossedRows / (float)tpcNClsFindable;
                             ;
                           });

DECLARE_SOA_DYNAMIC_COLUMN(TPCFractionSharedCls, tpcFractionSharedCls, [](uint8_t tpcNClsShared, uint8_t tpcNClsFindable, uint8_t tpcNClsFindableMinusFound) -> float {
  // FIXME: use tpcNClsFound from dynamic column as argument
  int16_t tpcNClsFound = tpcNClsFindable - tpcNClsFindableMinusFound;
  return (float)tpcNClsShared / (float)tpcNClsFound;
});
} // namespace track

DECLARE_SOA_TABLE(Tracks, "AOD", "TRACKPAR",
                  o2::soa::Index<>, track::CollisionId, track::TrackType,
                  track::X, track::Alpha,
                  track::Y, track::Z, track::Snp, track::Tgl,
                  track::Signed1Pt,
                  track::Phi<track::Snp, track::Alpha>,
                  track::Eta<track::Tgl>,
                  track::Pt<track::Signed1Pt>,
                  track::Px<track::Signed1Pt, track::Snp, track::Alpha>,
                  track::Py<track::Signed1Pt, track::Snp, track::Alpha>,
                  track::Pz<track::Signed1Pt, track::Tgl>,
                  track::P<track::Signed1Pt, track::Tgl>,
                  track::P2<track::Signed1Pt, track::Tgl>,
                  track::Charge<track::Signed1Pt>);

DECLARE_SOA_TABLE(TracksCov, "AOD", "TRACKPARCOV",
                  track::CYY, track::CZY, track::CZZ, track::CSnpY,
                  track::CSnpZ, track::CSnpSnp, track::CTglY,
                  track::CTglZ, track::CTglSnp, track::CTglTgl,
                  track::C1PtY, track::C1PtZ, track::C1PtSnp, track::C1PtTgl,
                  track::C1Pt21Pt2);

DECLARE_SOA_TABLE(TracksExtra, "AOD", "TRACKEXTRA",
                  track::TPCInnerParam, track::Flags, track::ITSClusterMap,
                  track::TPCNClsFindable, track::TPCNClsFindableMinusFound, track::TPCNClsFindableMinusCrossedRows,
                  track::TPCNClsShared, track::TRDNTracklets, track::ITSChi2NCl,
                  track::TPCChi2NCl, track::TRDChi2, track::TOFChi2,
                  track::TPCSignal, track::TRDSignal, track::TOFSignal, track::Length,
                  track::TPCNClsFound<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                  track::TPCNClsCrossedRows<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                  track::ITSNCls<track::ITSClusterMap>,
                  track::TPCCrossedRowsOverFindableCls<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>,
                  track::TPCFractionSharedCls<track::TPCNClsShared, track::TPCNClsFindable, track::TPCNClsFindableMinusFound>);

using Track = Tracks::iterator;
using TrackCov = TracksCov::iterator;
using TrackExtra = TracksExtra::iterator;

namespace unassignedtracks
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_INDEX_COLUMN(Track, track);
} // namespace unassignedtracks

DECLARE_SOA_TABLE(UnassignedTracks, "AOD", "UNASSIGNEDTRACK",
                  unassignedtracks::CollisionId, unassignedtracks::TrackId);

using UnassignedTrack = UnassignedTracks::iterator;

namespace calo
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(CellNumber, cellNumber, int16_t);
DECLARE_SOA_COLUMN(Amplitude, amplitude, float);
DECLARE_SOA_COLUMN(Time, time, float);
DECLARE_SOA_COLUMN(CellType, cellType, int8_t);
DECLARE_SOA_COLUMN(CaloType, caloType, int8_t);
} // namespace calo

DECLARE_SOA_TABLE(Calos, "AOD", "CALO", calo::BCId,
                  calo::CellNumber, calo::Amplitude, calo::Time,
                  calo::CellType, calo::CaloType);
using Calo = Calos::iterator;

namespace calotrigger
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(FastOrAbsId, fastOrAbsId, int32_t);
DECLARE_SOA_COLUMN(L0Amplitude, l0Amplitude, float);
DECLARE_SOA_COLUMN(L0Time, l0Time, float);
DECLARE_SOA_COLUMN(L1TimeSum, l1TimeSum, int32_t);
DECLARE_SOA_COLUMN(NL0Times, nl0Times, int8_t);
DECLARE_SOA_COLUMN(TriggerBits, triggerBits, int32_t);
DECLARE_SOA_COLUMN(CaloType, caloType, int8_t);
} // namespace calotrigger

DECLARE_SOA_TABLE(CaloTriggers, "AOD", "CALOTRIGGER",
                  calotrigger::BCId, calotrigger::FastOrAbsId,
                  calotrigger::L0Amplitude, calotrigger::L0Time,
                  calotrigger::L1TimeSum, calotrigger::NL0Times,
                  calotrigger::TriggerBits, calotrigger::CaloType);
using CaloTrigger = CaloTriggers::iterator;

namespace muon
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(InverseBendingMomentum, inverseBendingMomentum, float);
DECLARE_SOA_COLUMN(ThetaX, thetaX, float);
DECLARE_SOA_COLUMN(ThetaY, thetaY, float);
DECLARE_SOA_COLUMN(ZMu, zMu, float);
DECLARE_SOA_COLUMN(BendingCoor, bendingCoor, float);
DECLARE_SOA_COLUMN(NonBendingCoor, nonBendingCoor, float);
// FIXME: need to implement array columns...
// DECLARE_SOA_COLUMN(Covariances, covariances, float[], "fCovariances");
DECLARE_SOA_COLUMN(Chi2, chi2, float);
DECLARE_SOA_COLUMN(Chi2MatchTrigger, chi2MatchTrigger, float);
} // namespace muon

DECLARE_SOA_TABLE(Muons, "AOD", "MUON",
                  muon::BCId, muon::InverseBendingMomentum,
                  muon::ThetaX, muon::ThetaY, muon::ZMu,
                  muon::BendingCoor, muon::NonBendingCoor,
                  muon::Chi2, muon::Chi2MatchTrigger);
using Muon = Muons::iterator;

// NOTE for now muon tracks are uniquely assigned to a BC / GlobalBC assuming they contain an MID hit. Discussion on tracks without MID hit is ongoing.

namespace muoncluster
{
DECLARE_SOA_INDEX_COLUMN_FULL(Track, track, int, Muons, "fMuonsID"); // points to a muon track in the Muon table
DECLARE_SOA_COLUMN(X, x, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(Z, z, float);
DECLARE_SOA_COLUMN(ErrX, errX, float);
DECLARE_SOA_COLUMN(ErrY, errY, float);
DECLARE_SOA_COLUMN(Charge, charge, float);
DECLARE_SOA_COLUMN(Chi2, chi2, float);
} // namespace muoncluster

DECLARE_SOA_TABLE(MuonClusters, "AOD", "MUONCLUSTER",
                  muoncluster::TrackId,
                  muoncluster::X, muoncluster::Y, muoncluster::Z,
                  muoncluster::ErrX, muoncluster::ErrY,
                  muoncluster::Charge, muoncluster::Chi2);

using MuonCluster = MuonClusters::iterator;

namespace zdc
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(EnergyZEM1, energyZEM1, float);
DECLARE_SOA_COLUMN(EnergyZEM2, energyZEM2, float);
DECLARE_SOA_COLUMN(EnergyCommonZNA, energyCommonZNA, float);
DECLARE_SOA_COLUMN(EnergyCommonZNC, energyCommonZNC, float);
DECLARE_SOA_COLUMN(EnergyCommonZPA, energyCommonZPA, float);
DECLARE_SOA_COLUMN(EnergyCommonZPC, energyCommonZPC, float);
DECLARE_SOA_COLUMN(EnergySectorZNA, energySectorZNA, float[4]);
DECLARE_SOA_COLUMN(EnergySectorZNC, energySectorZNC, float[4]);
DECLARE_SOA_COLUMN(EnergySectorZPA, energySectorZPA, float[4]);
DECLARE_SOA_COLUMN(EnergySectorZPC, energySectorZPC, float[4]);
DECLARE_SOA_COLUMN(TimeZEM1, timeZEM1, float);
DECLARE_SOA_COLUMN(TimeZEM2, timeZEM2, float);
DECLARE_SOA_COLUMN(TimeZNA, timeZNA, float);
DECLARE_SOA_COLUMN(TimeZNC, timeZNC, float);
DECLARE_SOA_COLUMN(TimeZPA, timeZPA, float);
DECLARE_SOA_COLUMN(TimeZPC, timeZPC, float);
} // namespace zdc

DECLARE_SOA_TABLE(Zdcs, "AOD", "ZDC", zdc::BCId, zdc::EnergyZEM1, zdc::EnergyZEM2,
                  zdc::EnergyCommonZNA, zdc::EnergyCommonZNC, zdc::EnergyCommonZPA, zdc::EnergyCommonZPC,
                  zdc::EnergySectorZNA, zdc::EnergySectorZNC, zdc::EnergySectorZPA, zdc::EnergySectorZPC,
                  zdc::TimeZEM1, zdc::TimeZEM2, zdc::TimeZNA, zdc::TimeZNC, zdc::TimeZPA, zdc::TimeZPC);
using Zdc = Zdcs::iterator;

namespace ft0
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(Amplitude, amplitude, float[208]);
DECLARE_SOA_COLUMN(TimeA, timeA, float);
DECLARE_SOA_COLUMN(TimeC, timeC, float);
DECLARE_SOA_COLUMN(BCSignal, triggerSignal, uint8_t);
} // namespace ft0

DECLARE_SOA_TABLE(FT0s, "AOD", "FT0", ft0::BCId,
                  ft0::Amplitude, ft0::TimeA, ft0::TimeC,
                  ft0::BCSignal);
using FT0 = FT0s::iterator;

namespace fv0
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(Amplitude, amplitude, float[48]);
DECLARE_SOA_COLUMN(TimeA, timeA, float);
DECLARE_SOA_COLUMN(BCSignal, triggerSignal, uint8_t);
} // namespace fv0

DECLARE_SOA_TABLE(FV0s, "AOD", "FV0", fv0::BCId,
                  fv0::Amplitude, fv0::TimeA, fv0::BCSignal);
using FV0 = FV0s::iterator;

namespace fdd
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(Amplitude, amplitude, float[8]);
DECLARE_SOA_COLUMN(TimeA, timeA, float);
DECLARE_SOA_COLUMN(TimeC, timeC, float);
DECLARE_SOA_COLUMN(BCSignal, triggerSignal, uint8_t);
} // namespace fdd

DECLARE_SOA_TABLE(FDDs, "AOD", "FDD", fdd::BCId,
                  fdd::Amplitude, fdd::TimeA, fdd::TimeC,
                  fdd::BCSignal);
using FDD = FDDs::iterator;

namespace v0
{
DECLARE_SOA_INDEX_COLUMN_FULL(PosTrack, posTrack, int, Tracks, "fPosTrackID");
DECLARE_SOA_INDEX_COLUMN_FULL(NegTrack, negTrack, int, Tracks, "fNegTrackID");
} // namespace v0

DECLARE_SOA_TABLE(V0s, "AOD", "V0", v0::PosTrackId, v0::NegTrackId);
using V0 = V0s::iterator;

namespace cascade
{
DECLARE_SOA_INDEX_COLUMN(V0, v0);
DECLARE_SOA_INDEX_COLUMN_FULL(Bachelor, bachelor, int, Tracks, "fTracksID");
} // namespace cascade

DECLARE_SOA_TABLE(Cascades, "AOD", "CASCADE", cascade::V0Id, cascade::BachelorId);
using Casecade = Cascades::iterator;

// ---- LEGACY tables ----

namespace run2v0
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(Adc, adc, float[64]);
DECLARE_SOA_COLUMN(Time, time, float[64]);
DECLARE_SOA_COLUMN(Width, width, float[64]);
DECLARE_SOA_COLUMN(MultA, multA, float);
DECLARE_SOA_COLUMN(MultC, multC, float);
DECLARE_SOA_COLUMN(TimeA, timeA, float);
DECLARE_SOA_COLUMN(TimeC, timeC, float);
DECLARE_SOA_COLUMN(BBFlag, bbFlag, uint64_t);
DECLARE_SOA_COLUMN(BGFlag, bgFlag, uint64_t);
} // namespace run2v0

DECLARE_SOA_TABLE(Run2V0s, "AOD", "RUN2V0", run2v0::BCId,
                  run2v0::Adc, run2v0::Time, run2v0::Width,
                  run2v0::MultA, run2v0::MultC,
                  run2v0::TimeA, run2v0::TimeC,
                  run2v0::BBFlag, run2v0::BGFlag);
using Run2V0 = Run2V0s::iterator;

// ---- MC tables ----

namespace mccollision
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_COLUMN(GeneratorsID, generatorsID, short);
DECLARE_SOA_COLUMN(X, x, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(Z, z, float);
DECLARE_SOA_COLUMN(T, t, float);
DECLARE_SOA_COLUMN(Weight, weight, float);
} // namespace mccollision

DECLARE_SOA_TABLE(McCollisions, "AOD", "MCCOLLISION", o2::soa::Index<>, mccollision::BCId,
                  mccollision::GeneratorsID,
                  mccollision::X, mccollision::Y, mccollision::Z, mccollision::T, mccollision::Weight);
using McCollision = McCollisions::iterator;

namespace mcparticle
{
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision);
DECLARE_SOA_COLUMN(PdgCode, pdgCode, int);
DECLARE_SOA_COLUMN(StatusCode, statusCode, int);
DECLARE_SOA_COLUMN(Mother, mother, int[2]);
DECLARE_SOA_COLUMN(Daughter, daughter, int[2]);
DECLARE_SOA_COLUMN(Weight, weight, float);
DECLARE_SOA_COLUMN(Px, px, float);
DECLARE_SOA_COLUMN(Py, py, float);
DECLARE_SOA_COLUMN(Pz, pz, float);
DECLARE_SOA_COLUMN(E, e, float);
DECLARE_SOA_COLUMN(Vx, vx, float);
DECLARE_SOA_COLUMN(Vy, vy, float);
DECLARE_SOA_COLUMN(Vz, vz, float);
DECLARE_SOA_COLUMN(Vt, vt, float);
DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi, [](float px, float py) -> float { return static_cast<float>(M_PI) + std::atan2(-py, -px); });
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, [](float px, float py, float pz) -> float { return 0.5f * std::log((std::sqrt(px * px + py * py + pz * pz) + pz) / (std::sqrt(px * px + py * py + pz * pz) - pz)); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, [](float px, float py) -> float { return std::sqrt(px * px + py * py); });
} // namespace mcparticle

DECLARE_SOA_TABLE(McParticles, "AOD", "MCPARTICLE",
                  o2::soa::Index<>, mcparticle::McCollisionId,
                  mcparticle::PdgCode, mcparticle::StatusCode,
                  mcparticle::Mother, mcparticle::Daughter, mcparticle::Weight,
                  mcparticle::Px, mcparticle::Py, mcparticle::Pz, mcparticle::E,
                  mcparticle::Vx, mcparticle::Vy, mcparticle::Vz, mcparticle::Vt,
                  mcparticle::Phi<mcparticle::Px, mcparticle::Py>,
                  mcparticle::Eta<mcparticle::Px, mcparticle::Py, mcparticle::Pz>,
                  mcparticle::Pt<mcparticle::Px, mcparticle::Py>);
using McParticle = McParticles::iterator;

namespace mctracklabel
{
DECLARE_SOA_INDEX_COLUMN_FULL(Label, label, uint32_t, McParticles, "fLabel");
DECLARE_SOA_COLUMN(LabelMask, labelMask, uint16_t);
/// Bit mask to indicate detector mismatches (bit ON means mismatch)
/// Bit 0-6: mismatch at ITS layer
/// Bit 7-9: # of TPC mismatches in the ranges 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, >64
/// Bit 10: TRD, bit 11: TOF, bit 15: indicates negative label
} // namespace mctracklabel

DECLARE_SOA_TABLE(McTrackLabels, "AOD", "MCTRACKLABEL",
                  mctracklabel::LabelId, mctracklabel::LabelMask);
using McTrackLabel = McTrackLabels::iterator;

namespace mccalolabel
{
DECLARE_SOA_INDEX_COLUMN_FULL(Label, label, uint32_t, McParticles, "fLabel");
DECLARE_SOA_COLUMN(LabelMask, labelMask, uint16_t);
/// Bit mask to indicate detector mismatches (bit ON means mismatch)
/// Bit 15: indicates negative label
} // namespace mccalolabel

DECLARE_SOA_TABLE(McCaloLabels, "AOD", "MCCALOLABEL",
                  mccalolabel::LabelId, mccalolabel::LabelMask);
using McCaloLabel = McCaloLabels::iterator;

namespace mccollisionlabel
{
DECLARE_SOA_INDEX_COLUMN_FULL(Label, label, uint32_t, McCollisions, "fLabel");
DECLARE_SOA_COLUMN(LabelMask, labelMask, uint16_t);
/// Bit mask to indicate collision mismatches (bit ON means mismatch)
/// Bit 15: indicates negative label
} // namespace mccollisionlabel

DECLARE_SOA_TABLE(McCollisionLabels, "AOD", "MCCOLLISLABEL",
                  mccollisionlabel::LabelId, mccollisionlabel::LabelMask);
using McCollisionLabel = McCollisionLabels::iterator;

} // namespace aod

} // namespace o2
#endif // O2_FRAMEWORK_ANALYSISDATAMODEL_H_
