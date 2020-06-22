// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_SECONDARYVERTEXHF_H_
#define O2_ANALYSIS_SECONDARYVERTEXHF_H_

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace hftrackindexprong2
{
// FIXME: this is a workaround until we get the index columns to work with joins.
using BigTracks = soa::Join<Tracks, TracksCov, TracksExtra>;

DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_INDEX_COLUMN_FULL(Index0, index0, int, BigTracks, "fIndex0");
DECLARE_SOA_INDEX_COLUMN_FULL(Index1, index1, int, BigTracks, "fIndex1");
DECLARE_SOA_COLUMN(HFflag, hfflag, float);
} // namespace hftrackindexprong2

namespace hftrackindexprong3
{
// FIXME: this is a workaround until we get the index columns to work with joins.
using BigTracks = soa::Join<Tracks, TracksCov, TracksExtra>;

DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_INDEX_COLUMN_FULL(Index0, index0, int, BigTracks, "fIndex0");
DECLARE_SOA_INDEX_COLUMN_FULL(Index1, index1, int, BigTracks, "fIndex1");
DECLARE_SOA_INDEX_COLUMN_FULL(Index2, index2, int, BigTracks, "fIndex2");
DECLARE_SOA_COLUMN(HFflag, hfflag, float);
} // namespace hftrackindexprong3

DECLARE_SOA_TABLE(HfTrackIndexProng2, "AOD", "HFTRACKIDXP2",
                  hftrackindexprong2::CollisionId,
                  hftrackindexprong2::Index0Id,
                  hftrackindexprong2::Index1Id,
                  hftrackindexprong2::HFflag);

DECLARE_SOA_TABLE(HfTrackIndexProng3, "AOD", "HFTRACKIDXP3",
                  hftrackindexprong3::CollisionId,
                  hftrackindexprong3::Index0Id,
                  hftrackindexprong3::Index1Id,
                  hftrackindexprong3::Index2Id,
                  hftrackindexprong3::HFflag);
} // namespace o2::aod

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

float invmass3prongs2(float px0, float py0, float pz0, float mass0,
                      float px1, float py1, float pz1, float mass1,
                      float px2, float py2, float pz2, float mass2)
{
  float energy0_ = energy(px0, py0, pz0, mass0);
  float energy1_ = energy(px1, py1, pz1, mass1);
  float energy2_ = energy(px2, py2, pz2, mass2);
  float energytot = energy0_ + energy1_ + energy2_;

  float psum2 = (px0 + px1 + px2) * (px0 + px1 + px2) +
                (py0 + py1 + py2) * (py0 + py1 + py2) +
                (pz0 + pz1 + pz2) * (pz0 + pz1 + pz2);
  return energytot * energytot - psum2;
};

#endif // O2_ANALYSIS_SECONDARYVERTEXHF_H_
