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

DECLARE_SOA_TABLE(Cand2Prong, "AOD", "CANDDZERO",
                  //cand2prong::CollisionId,
                  cand2prong::MassD0, cand2prong::MassD0bar);
} // namespace o2::aod

#endif // O2_ANALYSIS_SECONDARYVERTEX_H_
