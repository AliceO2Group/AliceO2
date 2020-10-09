// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFSecondaryVertex.h
/// \brief Definitions of tables of heavy-flavour decay candidates.
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#ifndef O2_ANALYSIS_HFSECONDARYVERTEX_H_
#define O2_ANALYSIS_HFSECONDARYVERTEX_H_

#include "Framework/AnalysisDataModel.h"
#include "Analysis/RecoDecay.h"
#include "PID/PIDResponse.h"

namespace o2::aod
{
namespace hf_seltrack
{
DECLARE_SOA_COLUMN(IsSel2Prong, isSel2Prong, int);
DECLARE_SOA_COLUMN(IsSel3Prong, isSel3Prong, int);
DECLARE_SOA_COLUMN(DCAPrim0, dcaPrim0, float);
DECLARE_SOA_COLUMN(DCAPrim1, dcaPrim1, float);
} // namespace hf_seltrack
DECLARE_SOA_TABLE(HFSelTrack, "AOD", "SELTRACK",
                  hf_seltrack::IsSel2Prong,
                  hf_seltrack::IsSel3Prong,
                  hf_seltrack::DCAPrim0,
                  hf_seltrack::DCAPrim1);

using BigTracks = soa::Join<Tracks, TracksCov, TracksExtra, HFSelTrack, pidRespTPC, pidRespTOF>;

// FIXME: this is a workaround until we get the index columns to work with joins.

namespace hf_track_index
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_INDEX_COLUMN_FULL(Index0, index0, int, BigTracks, "fIndex0");
DECLARE_SOA_INDEX_COLUMN_FULL(Index1, index1, int, BigTracks, "fIndex1");
DECLARE_SOA_INDEX_COLUMN_FULL(Index2, index2, int, BigTracks, "fIndex2");
DECLARE_SOA_INDEX_COLUMN_FULL(Index3, index3, int, BigTracks, "fIndex3");
DECLARE_SOA_COLUMN(HFflag, hfflag, int);
} // namespace hf_track_index

DECLARE_SOA_TABLE(HfTrackIndexProng2, "AOD", "HFTRACKIDXP2",
                  hf_track_index::CollisionId,
                  hf_track_index::Index0Id,
                  hf_track_index::Index1Id,
                  hf_track_index::HFflag);

DECLARE_SOA_TABLE(HfTrackIndexProng3, "AOD", "HFTRACKIDXP3",
                  hf_track_index::CollisionId,
                  hf_track_index::Index0Id,
                  hf_track_index::Index1Id,
                  hf_track_index::Index2Id,
                  hf_track_index::HFflag);

// general decay properties
namespace hf_cand
{
// secondary vertex
DECLARE_SOA_COLUMN(XSecondaryVertex, xSecondaryVertex, float);
DECLARE_SOA_COLUMN(YSecondaryVertex, ySecondaryVertex, float);
DECLARE_SOA_COLUMN(ZSecondaryVertex, zSecondaryVertex, float);
DECLARE_SOA_DYNAMIC_COLUMN(RSecondaryVertex, rSecondaryVertex, [](float xVtxS, float yVtxS) { return RecoDecay::sqrtSumOfSquares(xVtxS, yVtxS); });
DECLARE_SOA_COLUMN(Chi2PCA, chi2PCA, float); // sum of (non-weighted) distances of the secondary vertex to its prongs
// prong properties
DECLARE_SOA_COLUMN(PxProng0, pxProng0, float);
DECLARE_SOA_COLUMN(PyProng0, pyProng0, float);
DECLARE_SOA_COLUMN(PzProng0, pzProng0, float);
DECLARE_SOA_DYNAMIC_COLUMN(PtProng0, ptProng0, [](float px, float py) { return RecoDecay::Pt(px, py); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt2Prong0, pt2Prong0, [](float px, float py) { return RecoDecay::Pt2(px, py); });
DECLARE_SOA_DYNAMIC_COLUMN(PVectorProng0, pVectorProng0, [](float px, float py, float pz) { return array{px, py, pz}; });
DECLARE_SOA_COLUMN(ImpactParameter0, impactParameter0, float);
DECLARE_SOA_COLUMN(ErrorImpactParameter0, errorImpactParameter0, float);
DECLARE_SOA_DYNAMIC_COLUMN(ImpactParameterNormalised0, impactParameterNormalised0, [](float dca, float err) { return dca / err; });
DECLARE_SOA_COLUMN(PxProng1, pxProng1, float);
DECLARE_SOA_COLUMN(PyProng1, pyProng1, float);
DECLARE_SOA_COLUMN(PzProng1, pzProng1, float);
DECLARE_SOA_DYNAMIC_COLUMN(PtProng1, ptProng1, [](float px, float py) { return RecoDecay::Pt(px, py); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt2Prong1, pt2Prong1, [](float px, float py) { return RecoDecay::Pt2(px, py); });
DECLARE_SOA_DYNAMIC_COLUMN(PVectorProng1, pVectorProng1, [](float px, float py, float pz) { return array{px, py, pz}; });
DECLARE_SOA_COLUMN(ImpactParameter1, impactParameter1, float);
DECLARE_SOA_COLUMN(ErrorImpactParameter1, errorImpactParameter1, float);
DECLARE_SOA_DYNAMIC_COLUMN(ImpactParameterNormalised1, impactParameterNormalised1, [](float dca, float err) { return dca / err; });
DECLARE_SOA_COLUMN(PxProng2, pxProng2, float);
DECLARE_SOA_COLUMN(PyProng2, pyProng2, float);
DECLARE_SOA_COLUMN(PzProng2, pzProng2, float);
DECLARE_SOA_DYNAMIC_COLUMN(PtProng2, ptProng2, [](float px, float py) { return RecoDecay::Pt(px, py); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt2Prong2, pt2Prong2, [](float px, float py) { return RecoDecay::Pt2(px, py); });
DECLARE_SOA_DYNAMIC_COLUMN(PVectorProng2, pVectorProng2, [](float px, float py, float pz) { return array{px, py, pz}; });
DECLARE_SOA_COLUMN(ImpactParameter2, impactParameter2, float);
DECLARE_SOA_COLUMN(ErrorImpactParameter2, errorImpactParameter2, float);
DECLARE_SOA_DYNAMIC_COLUMN(ImpactParameterNormalised2, impactParameterNormalised2, [](float dca, float err) { return dca / err; });
// candidate properties
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, [](float px, float py) { return RecoDecay::Pt(px, py); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt2, pt2, [](float px, float py) { return RecoDecay::Pt2(px, py); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, [](float px, float py, float pz) { return RecoDecay::P(px, py, pz); });
DECLARE_SOA_DYNAMIC_COLUMN(P2, p2, [](float px, float py, float pz) { return RecoDecay::P2(px, py, pz); });
DECLARE_SOA_DYNAMIC_COLUMN(PVector, pVector, [](float px, float py, float pz) { return array{px, py, pz}; });
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, [](float px, float py, float pz) { return RecoDecay::Eta(array{px, py, pz}); });
DECLARE_SOA_DYNAMIC_COLUMN(Y, y, [](float px, float py, float pz, double m) { return RecoDecay::Y(array{px, py, pz}, m); });
DECLARE_SOA_DYNAMIC_COLUMN(E, e, [](float px, float py, float pz, double m) { return RecoDecay::E(px, py, pz, m); });
DECLARE_SOA_DYNAMIC_COLUMN(E2, e2, [](float px, float py, float pz, double m) { return RecoDecay::E2(px, py, pz, m); });
DECLARE_SOA_DYNAMIC_COLUMN(DecayLength, decayLength, [](float xVtxP, float yVtxP, float zVtxP, float xVtxS, float yVtxS, float zVtxS) { return RecoDecay::distance(array{xVtxP, yVtxP, zVtxP}, array{xVtxS, yVtxS, zVtxS}); });
DECLARE_SOA_DYNAMIC_COLUMN(DecayLengthXY, decayLengthXY, [](float xVtxP, float yVtxP, float xVtxS, float yVtxS) { return RecoDecay::distanceXY(array{xVtxP, yVtxP}, array{xVtxS, yVtxS}); });
DECLARE_SOA_DYNAMIC_COLUMN(DecayLengthNormalised, decayLengthNormalised, [](float xVtxP, float yVtxP, float zVtxP, float xVtxS, float yVtxS, float zVtxS, float err) { return RecoDecay::distance(array{xVtxP, yVtxP, zVtxP}, array{xVtxS, yVtxS, zVtxS}) / err; });
DECLARE_SOA_DYNAMIC_COLUMN(DecayLengthXYNormalised, decayLengthXYNormalised, [](float xVtxP, float yVtxP, float xVtxS, float yVtxS, float err) { return RecoDecay::distanceXY(array{xVtxP, yVtxP}, array{xVtxS, yVtxS}) / err; });
DECLARE_SOA_COLUMN(ErrorDecayLength, errorDecayLength, float);
DECLARE_SOA_COLUMN(ErrorDecayLengthXY, errorDecayLengthXY, float);
DECLARE_SOA_DYNAMIC_COLUMN(CPA, cpa, [](float xVtxP, float yVtxP, float zVtxP, float xVtxS, float yVtxS, float zVtxS, float px, float py, float pz) { return RecoDecay::CPA(array{xVtxP, yVtxP, zVtxP}, array{xVtxS, yVtxS, zVtxS}, array{px, py, pz}); });
DECLARE_SOA_DYNAMIC_COLUMN(CPAXY, cpaXY, [](float xVtxP, float yVtxP, float xVtxS, float yVtxS, float px, float py) { return RecoDecay::CPAXY(array{xVtxP, yVtxP}, array{xVtxS, yVtxS}, array{px, py}); });
DECLARE_SOA_DYNAMIC_COLUMN(Ct, ct, [](float xVtxP, float yVtxP, float zVtxP, float xVtxS, float yVtxS, float zVtxS, float px, float py, float pz, double m) { return RecoDecay::Ct(array{px, py, pz}, RecoDecay::distance(array{xVtxP, yVtxP, zVtxP}, array{xVtxS, yVtxS, zVtxS}), m); });
} // namespace hf_cand

// specific 2-prong decay properties
namespace hf_cand_prong2
{
DECLARE_SOA_EXPRESSION_COLUMN(Px, px, float, 1.f * aod::hf_cand::pxProng0 + 1.f * aod::hf_cand::pxProng1);
DECLARE_SOA_EXPRESSION_COLUMN(Py, py, float, 1.f * aod::hf_cand::pyProng0 + 1.f * aod::hf_cand::pyProng1);
DECLARE_SOA_EXPRESSION_COLUMN(Pz, pz, float, 1.f * aod::hf_cand::pzProng0 + 1.f * aod::hf_cand::pzProng1);
DECLARE_SOA_DYNAMIC_COLUMN(ImpactParameterProduct, impactParameterProduct, [](float dca1, float dca2) { return dca1 * dca2; });
DECLARE_SOA_DYNAMIC_COLUMN(M, m, [](float px0, float py0, float pz0, float px1, float py1, float pz1, const array<double, 2>& m) { return RecoDecay::M(array{array{px0, py0, pz0}, array{px1, py1, pz1}}, m); });
DECLARE_SOA_DYNAMIC_COLUMN(M2, m2, [](float px0, float py0, float pz0, float px1, float py1, float pz1, const array<double, 2>& m) { return RecoDecay::M2(array{array{px0, py0, pz0}, array{px1, py1, pz1}}, m); });
DECLARE_SOA_DYNAMIC_COLUMN(CosThetaStar, cosThetaStar, [](float px0, float py0, float pz0, float px1, float py1, float pz1, const array<double, 2>& m, double mTot, int iProng) { return RecoDecay::CosThetaStar(array{array{px0, py0, pz0}, array{px1, py1, pz1}}, m, mTot, iProng); });

// functions for specific particles

// D0(bar) → π± K∓

template <typename T>
auto CtD0(const T& candidate)
{
  return candidate.ct(RecoDecay::getMassPDG(421));
}

template <typename T>
auto YD0(const T& candidate)
{
  return candidate.y(RecoDecay::getMassPDG(421));
}

template <typename T>
auto ED0(const T& candidate)
{
  return candidate.e(RecoDecay::getMassPDG(421));
}

template <typename T>
auto InvMassD0(const T& candidate)
{
  return candidate.m(array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kKPlus)});
}

template <typename T>
auto InvMassD0bar(const T& candidate)
{
  return candidate.m(array{RecoDecay::getMassPDG(kKPlus), RecoDecay::getMassPDG(kPiPlus)});
}

template <typename T>
auto CosThetaStarD0(const T& candidate)
{
  return candidate.cosThetaStar(array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kKPlus)}, RecoDecay::getMassPDG(421), 1);
}

template <typename T>
auto CosThetaStarD0bar(const T& candidate)
{
  return candidate.cosThetaStar(array{RecoDecay::getMassPDG(kKPlus), RecoDecay::getMassPDG(kPiPlus)}, RecoDecay::getMassPDG(421), 0);
}
} // namespace hf_cand_prong2

DECLARE_SOA_TABLE(HfCandBase, "AOD", "HFCANDBASE",
                  collision::PosX, collision::PosY, collision::PosZ,
                  hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex,
                  hf_cand::ErrorDecayLength, hf_cand::ErrorDecayLengthXY,
                  hf_cand::Chi2PCA,
                  /* dynamic columns */
                  hf_cand::RSecondaryVertex<hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex>,
                  hf_cand::ImpactParameterNormalised0<hf_cand::ImpactParameter0, hf_cand::ErrorImpactParameter0>,
                  hf_cand::ImpactParameterNormalised1<hf_cand::ImpactParameter1, hf_cand::ErrorImpactParameter1>,
                  hf_cand::PtProng0<hf_cand::PxProng0, hf_cand::PyProng0>,
                  hf_cand::Pt2Prong0<hf_cand::PxProng0, hf_cand::PyProng0>,
                  hf_cand::PVectorProng0<hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0>,
                  hf_cand::PtProng1<hf_cand::PxProng1, hf_cand::PyProng1>,
                  hf_cand::Pt2Prong1<hf_cand::PxProng1, hf_cand::PyProng1>,
                  hf_cand::PVectorProng1<hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1>,
                  hf_cand::DecayLength<collision::PosX, collision::PosY, collision::PosZ, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex>,
                  hf_cand::DecayLengthXY<collision::PosX, collision::PosY, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex>,
                  hf_cand::DecayLengthNormalised<collision::PosX, collision::PosY, collision::PosZ, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex, hf_cand::ErrorDecayLength>,
                  hf_cand::DecayLengthXYNormalised<collision::PosX, collision::PosY, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ErrorDecayLengthXY>, //);

                  //DECLARE_SOA_TABLE(HfCandProng2Base, "AOD", "HFCANDPRG2BASE", // TODO split table
                  hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0,
                  hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1,
                  hf_cand::ImpactParameter0, hf_cand::ImpactParameter1,
                  hf_cand::ErrorImpactParameter0, hf_cand::ErrorImpactParameter1,
                  hf_track_index::Index0Id, hf_track_index::Index1Id,
                  /* dynamic columns */
                  hf_cand_prong2::ImpactParameterProduct<hf_cand::ImpactParameter0, hf_cand::ImpactParameter1>,
                  hf_cand_prong2::M<hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0, hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1>,
                  hf_cand_prong2::M2<hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0, hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1>,
                  hf_cand_prong2::CosThetaStar<hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0, hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1>,
                  /* dynamic columns that use candidate momentum components */
                  hf_cand::Pt<hf_cand_prong2::Px, hf_cand_prong2::Py>,
                  hf_cand::Pt2<hf_cand_prong2::Px, hf_cand_prong2::Py>,
                  hf_cand::P<hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>,
                  hf_cand::P2<hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>,
                  hf_cand::PVector<hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>,
                  hf_cand::CPA<collision::PosX, collision::PosY, collision::PosZ, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex, hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>,
                  hf_cand::CPAXY<collision::PosX, collision::PosY, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand_prong2::Px, hf_cand_prong2::Py>,
                  hf_cand::Ct<collision::PosX, collision::PosY, collision::PosZ, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex, hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>,
                  hf_cand::Eta<hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>,
                  hf_cand::Y<hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>,
                  hf_cand::E<hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>,
                  hf_cand::E2<hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz>);

//using HfCandProng2Origin = soa::Join<HfCandBase, HfCandProng2Base>; // TODO split table
using HfCandProng2Origin = HfCandBase;

// extended table with expression columns that can be used as arguments of dynamic columns
DECLARE_SOA_EXTENDED_TABLE_USER(HfCandProng2Ext, HfCandProng2Origin, "HFCANDPRG2EXT",
                                hf_cand_prong2::Px, hf_cand_prong2::Py, hf_cand_prong2::Pz);

using HfCandProng2 = HfCandProng2Ext;

// specific 3-prong decay properties
namespace hf_cand_prong3
{
DECLARE_SOA_EXPRESSION_COLUMN(Px, px, float, 1.f * aod::hf_cand::pxProng0 + 1.f * aod::hf_cand::pxProng1 + 1.f * aod::hf_cand::pxProng2);
DECLARE_SOA_EXPRESSION_COLUMN(Py, py, float, 1.f * aod::hf_cand::pyProng0 + 1.f * aod::hf_cand::pyProng1 + 1.f * aod::hf_cand::pyProng2);
DECLARE_SOA_EXPRESSION_COLUMN(Pz, pz, float, 1.f * aod::hf_cand::pzProng0 + 1.f * aod::hf_cand::pzProng1 + 1.f * aod::hf_cand::pzProng2);
DECLARE_SOA_DYNAMIC_COLUMN(M, m, [](float px0, float py0, float pz0, float px1, float py1, float pz1, float px2, float py2, float pz2, const array<double, 3>& m) { return RecoDecay::M(array{array{px0, py0, pz0}, array{px1, py1, pz1}, array{px2, py2, pz2}}, m); });
DECLARE_SOA_DYNAMIC_COLUMN(M2, m2, [](float px0, float py0, float pz0, float px1, float py1, float pz1, float px2, float py2, float pz2, const array<double, 3>& m) { return RecoDecay::M2(array{array{px0, py0, pz0}, array{px1, py1, pz1}, array{px2, py2, pz2}}, m); });

// functions for specific particles

// D± → π± K∓ π±

template <typename T>
auto CtDPlus(const T& candidate)
{
  return candidate.ct(RecoDecay::getMassPDG(411));
}

template <typename T>
auto YDPlus(const T& candidate)
{
  return candidate.y(RecoDecay::getMassPDG(411));
}

template <typename T>
auto EDPlus(const T& candidate)
{
  return candidate.e(RecoDecay::getMassPDG(411));
}

template <typename T>
auto InvMassDPlus(const T& candidate)
{
  return candidate.m(array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kKPlus), RecoDecay::getMassPDG(kPiPlus)});
}
} // namespace hf_cand_prong3

DECLARE_SOA_TABLE(HfCandBase3, "AOD", "HFCANDBASE3",
                  collision::PosX, collision::PosY, collision::PosZ,
                  hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex,
                  hf_cand::ErrorDecayLength, hf_cand::ErrorDecayLengthXY,
                  hf_cand::Chi2PCA,
                  /* dynamic columns */
                  hf_cand::RSecondaryVertex<hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex>,
                  hf_cand::ImpactParameterNormalised0<hf_cand::ImpactParameter0, hf_cand::ErrorImpactParameter0>,
                  hf_cand::ImpactParameterNormalised1<hf_cand::ImpactParameter1, hf_cand::ErrorImpactParameter1>,
                  hf_cand::PtProng0<hf_cand::PxProng0, hf_cand::PyProng0>,
                  hf_cand::Pt2Prong0<hf_cand::PxProng0, hf_cand::PyProng0>,
                  hf_cand::PVectorProng0<hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0>,
                  hf_cand::PtProng1<hf_cand::PxProng1, hf_cand::PyProng1>,
                  hf_cand::Pt2Prong1<hf_cand::PxProng1, hf_cand::PyProng1>,
                  hf_cand::PVectorProng1<hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1>,
                  hf_cand::DecayLength<collision::PosX, collision::PosY, collision::PosZ, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex>,
                  hf_cand::DecayLengthXY<collision::PosX, collision::PosY, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex>,
                  hf_cand::DecayLengthNormalised<collision::PosX, collision::PosY, collision::PosZ, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex, hf_cand::ErrorDecayLength>,
                  hf_cand::DecayLengthXYNormalised<collision::PosX, collision::PosY, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ErrorDecayLengthXY>, //);

                  //DECLARE_SOA_TABLE(HfCandProng3Base, "AOD", "HFCANDPRG3BASE", // TODO split table
                  hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0,
                  hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1,
                  hf_cand::PxProng2, hf_cand::PyProng2, hf_cand::PzProng2,
                  hf_cand::ImpactParameter0, hf_cand::ImpactParameter1, hf_cand::ImpactParameter2,
                  hf_cand::ErrorImpactParameter0, hf_cand::ErrorImpactParameter1, hf_cand::ErrorImpactParameter2,
                  hf_track_index::Index0Id, hf_track_index::Index1Id, hf_track_index::Index2Id,
                  /* dynamic columns */
                  hf_cand_prong3::M<hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0, hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1, hf_cand::PxProng2, hf_cand::PyProng2, hf_cand::PzProng2>,
                  hf_cand_prong3::M2<hf_cand::PxProng0, hf_cand::PyProng0, hf_cand::PzProng0, hf_cand::PxProng1, hf_cand::PyProng1, hf_cand::PzProng1, hf_cand::PxProng2, hf_cand::PyProng2, hf_cand::PzProng2>,
                  hf_cand::ImpactParameterNormalised2<hf_cand::ImpactParameter2, hf_cand::ErrorImpactParameter2>,
                  hf_cand::PtProng2<hf_cand::PxProng2, hf_cand::PyProng2>,
                  hf_cand::Pt2Prong2<hf_cand::PxProng2, hf_cand::PyProng2>,
                  hf_cand::PVectorProng2<hf_cand::PxProng2, hf_cand::PyProng2, hf_cand::PzProng2>,
                  /* dynamic columns that use candidate momentum components */
                  hf_cand::Pt<hf_cand_prong3::Px, hf_cand_prong3::Py>,
                  hf_cand::Pt2<hf_cand_prong3::Px, hf_cand_prong3::Py>,
                  hf_cand::P<hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>,
                  hf_cand::P2<hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>,
                  hf_cand::PVector<hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>,
                  hf_cand::CPA<collision::PosX, collision::PosY, collision::PosZ, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex, hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>,
                  hf_cand::CPAXY<collision::PosX, collision::PosY, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand_prong3::Px, hf_cand_prong3::Py>,
                  hf_cand::Ct<collision::PosX, collision::PosY, collision::PosZ, hf_cand::XSecondaryVertex, hf_cand::YSecondaryVertex, hf_cand::ZSecondaryVertex, hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>,
                  hf_cand::Eta<hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>,
                  hf_cand::Y<hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>,
                  hf_cand::E<hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>,
                  hf_cand::E2<hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz>);

//using HfCandProng3Origin = soa::Join<HfCandBase3, HfCandProng3Base>; // TODO split table
using HfCandProng3Origin = HfCandBase3;

// extended table with expression columns that can be used as arguments of dynamic columns
DECLARE_SOA_EXTENDED_TABLE_USER(HfCandProng3Ext, HfCandProng3Origin, "HFCANDPRG3EXT",
                                hf_cand_prong3::Px, hf_cand_prong3::Py, hf_cand_prong3::Pz);

using HfCandProng3 = HfCandProng3Ext;

} // namespace o2::aod

#endif // O2_ANALYSIS_HFSECONDARYVERTEX_H_
