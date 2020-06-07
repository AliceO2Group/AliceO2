// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_SECONDARYVERTEX_H_
#define O2_ANALYSIS_SECONDARYVERTEX_H_

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace secvtx2prong
{
// FIXME: this is a workaround until we get the index columns to work with joins.
using BigTracks = soa::Join<Tracks, TracksCov, TracksExtra>;

DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(Posdecayx, posdecayx, float);
DECLARE_SOA_COLUMN(Posdecayy, posdecayy, float);
DECLARE_SOA_COLUMN(Posdecayz, posdecayz, float);
DECLARE_SOA_INDEX_COLUMN_FULL(Index0, index0, int, BigTracks, "fIndex0");
DECLARE_SOA_COLUMN(Px0, px0, float);
DECLARE_SOA_COLUMN(Py0, py0, float);
DECLARE_SOA_COLUMN(Pz0, pz0, float);
DECLARE_SOA_COLUMN(Y0, y0, float);
DECLARE_SOA_INDEX_COLUMN_FULL(Index1, index1, int, BigTracks, "fIndex1");
DECLARE_SOA_COLUMN(Px1, px1, float);
DECLARE_SOA_COLUMN(Py1, py1, float);
DECLARE_SOA_COLUMN(Pz1, pz1, float);
DECLARE_SOA_COLUMN(Y1, y1, float);
DECLARE_SOA_COLUMN(Mass, mass, float);
DECLARE_SOA_COLUMN(Massbar, massbar, float);
DECLARE_SOA_DYNAMIC_COLUMN(DecaylengthXY, decaylengthXY,
                           [](float xvtxd, float yvtxd, float xvtxp, float yvtxp) { return sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp)); });
DECLARE_SOA_DYNAMIC_COLUMN(Decaylength, decaylength,
                           [](float xvtxd, float yvtxd, float zvtxd, float xvtxp,
                              float yvtxp, float zvtxp) { return sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp) + (zvtxd - zvtxp) * (zvtxd - zvtxp)); });
//old way of doing it
//DECLARE_SOA_COLUMN(Decaylength, decaylength, float);
//DECLARE_SOA_COLUMN(DecaylengthXY, decaylengthXY, float);

} // namespace secvtx2prong
namespace secvtx3prong
{
// FIXME: this is a workaround until we get the index columns to work with joins.
using BigTracks = soa::Join<Tracks, TracksCov, TracksExtra>;

DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(Posdecayx, posdecayx, float);
DECLARE_SOA_COLUMN(Posdecayy, posdecayy, float);
DECLARE_SOA_COLUMN(Posdecayz, posdecayz, float);
DECLARE_SOA_INDEX_COLUMN_FULL(Index0, index0, int, BigTracks, "fIndex0");
DECLARE_SOA_COLUMN(Px0, px0, float);
DECLARE_SOA_COLUMN(Py0, py0, float);
DECLARE_SOA_COLUMN(Pz0, pz0, float);
DECLARE_SOA_COLUMN(Y0, y0, float);
DECLARE_SOA_INDEX_COLUMN_FULL(Index1, index1, int, BigTracks, "fIndex1");
DECLARE_SOA_COLUMN(Px1, px1, float);
DECLARE_SOA_COLUMN(Py1, py1, float);
DECLARE_SOA_COLUMN(Pz1, pz1, float);
DECLARE_SOA_COLUMN(Y1, y1, float);
DECLARE_SOA_INDEX_COLUMN_FULL(Index2, index2, int, BigTracks, "fIndex2");
DECLARE_SOA_COLUMN(Px2, px2, float);
DECLARE_SOA_COLUMN(Py2, py2, float);
DECLARE_SOA_COLUMN(Pz2, pz2, float);
DECLARE_SOA_COLUMN(Y2, y2, float);
DECLARE_SOA_COLUMN(Mass, mass, float);
DECLARE_SOA_COLUMN(Massbar, massbar, float);
DECLARE_SOA_DYNAMIC_COLUMN(DecaylengthXY, decaylengthXY,
                           [](float xvtxd, float yvtxd, float xvtxp, float yvtxp) { return sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp)); });
DECLARE_SOA_DYNAMIC_COLUMN(Decaylength, decaylength,
                           [](float xvtxd, float yvtxd, float zvtxd, float xvtxp,
                              float yvtxp, float zvtxp) { return sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp) + (zvtxd - zvtxp) * (zvtxd - zvtxp)); });
//old way of doing it
//DECLARE_SOA_COLUMN(Decaylength, decaylength, float);
//DECLARE_SOA_COLUMN(DecaylengthXY, decaylengthXY, float);

} // namespace secvtx3prong
namespace cand2prong
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(MassD0, massD0, float);
DECLARE_SOA_COLUMN(MassD0bar, massD0bar, float);
} // namespace cand2prong

DECLARE_SOA_TABLE(SecVtx2Prong, "AOD", "VTX2PRONG",
                  secvtx2prong::CollisionId,
                  collision::PosX, collision::PosY, collision::PosZ,
                  secvtx2prong::Posdecayx, secvtx2prong::Posdecayy, secvtx2prong::Posdecayz,
                  secvtx2prong::Index0Id,
                  secvtx2prong::Px0, secvtx2prong::Py0, secvtx2prong::Pz0, secvtx2prong::Y0,
                  secvtx2prong::Index1Id,
                  secvtx2prong::Px1, secvtx2prong::Py1, secvtx2prong::Pz1, secvtx2prong::Y1,
                  secvtx2prong::Mass, secvtx2prong::Massbar,
                  secvtx2prong::DecaylengthXY<secvtx2prong::Posdecayx, secvtx2prong::Posdecayy,
                                              collision::PosX, collision::PosY>,
                  secvtx2prong::Decaylength<secvtx2prong::Posdecayx, secvtx2prong::Posdecayy,
                                            secvtx2prong::Posdecayz, collision::PosX,
                                            collision::PosY, collision::PosZ>);
DECLARE_SOA_TABLE(SecVtx3Prong, "AOD", "VTX3PRONG",
                  secvtx3prong::CollisionId,
                  collision::PosX, collision::PosY, collision::PosZ,
                  secvtx3prong::Posdecayx, secvtx3prong::Posdecayy, secvtx3prong::Posdecayz,
                  secvtx3prong::Index0Id,
                  secvtx3prong::Px0, secvtx3prong::Py0, secvtx3prong::Pz0, secvtx3prong::Y0,
                  secvtx3prong::Index1Id,
                  secvtx3prong::Px1, secvtx3prong::Py1, secvtx3prong::Pz1, secvtx3prong::Y1,
                  secvtx3prong::Index2Id,
                  secvtx3prong::Px2, secvtx3prong::Py2, secvtx3prong::Pz2, secvtx3prong::Y2,
                  secvtx3prong::Mass, secvtx3prong::Massbar,
                  secvtx3prong::DecaylengthXY<secvtx3prong::Posdecayx, secvtx3prong::Posdecayy,
                                              collision::PosX, collision::PosY>,
                  secvtx3prong::Decaylength<secvtx3prong::Posdecayx, secvtx3prong::Posdecayy,
                                            secvtx3prong::Posdecayz, collision::PosX,
                                            collision::PosY, collision::PosZ>);

DECLARE_SOA_TABLE(Cand2Prong, "AOD", "CANDDZERO",
                  //cand2prong::CollisionId,
                  cand2prong::MassD0, cand2prong::MassD0bar);
} // namespace o2::aod

//FIXME: this functions will need to become dynamic columns
//once we will be able to build dynamic columns starting from
//columns that belongs to different tables

float energy(float px, float py, float pz, float mass)
{
  float en_ = sqrtf(mass * mass + px * px + py * py + pz * pz);
  return en_;
};

float invmass2prongs(float px0, float py0, float pz0, float mass0,
                     float px1, float py1, float pz1, float mass1)
{

  float energy0_ = energy(px0, py0, pz0, mass0);
  float energy1_ = energy(px1, py1, pz1, mass1);
  float energytot = energy0_ + energy1_;

  float psum2 = (px0 + px1) * (px0 + px1) +
                (py0 + py1) * (py0 + py1) +
                (pz0 + pz1) * (pz0 + pz1);
  float mass = sqrtf(energytot * energytot - psum2);
  return mass;
};

#endif // O2_ANALYSIS_SECONDARYVERTEX_H_
