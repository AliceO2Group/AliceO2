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
#include "Analysis/RecoDecay.h"

namespace o2::aod
{
namespace hftrackindexprong2
{
// FIXME: this is a workaround until we get the index columns to work with joins.
using BigTracks = soa::Join<Tracks, TracksCov, TracksExtra>;

DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_INDEX_COLUMN_FULL(Index0, index0, int, BigTracks, "fIndex0");
DECLARE_SOA_INDEX_COLUMN_FULL(Index1, index1, int, BigTracks, "fIndex1");
DECLARE_SOA_COLUMN(HFflag, hfflag, int);
} // namespace hftrackindexprong2

namespace hftrackindexprong3
{
// FIXME: this is a workaround until we get the index columns to work with joins.
using BigTracks = soa::Join<Tracks, TracksCov, TracksExtra>;

DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_INDEX_COLUMN_FULL(Index0, index0, int, BigTracks, "fIndex0");
DECLARE_SOA_INDEX_COLUMN_FULL(Index1, index1, int, BigTracks, "fIndex1");
DECLARE_SOA_INDEX_COLUMN_FULL(Index2, index2, int, BigTracks, "fIndex2");
DECLARE_SOA_COLUMN(HFflag, hfflag, int);
} // namespace hftrackindexprong3

namespace hfcandprong2
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(PxProng0, pxprong0, float);
DECLARE_SOA_COLUMN(PyProng0, pyprong0, float);
DECLARE_SOA_COLUMN(PzProng0, pzprong0, float);
DECLARE_SOA_COLUMN(PxProng1, pxprong1, float);
DECLARE_SOA_COLUMN(PyProng1, pyprong1, float);
DECLARE_SOA_COLUMN(PzProng1, pzprong1, float);
DECLARE_SOA_COLUMN(DecayVtxX, decayvtxx, float);
DECLARE_SOA_COLUMN(DecayVtxY, decayvtxy, float);
DECLARE_SOA_COLUMN(DecayVtxZ, decayvtxz, float);
DECLARE_SOA_COLUMN(MassD0, massD0, float);
DECLARE_SOA_COLUMN(MassD0bar, massD0bar, float);
DECLARE_SOA_DYNAMIC_COLUMN(PtProng0, ptprong0,
                           [](float px0, float py0) { return pttrack(px0, py0); });
DECLARE_SOA_DYNAMIC_COLUMN(PtProng1, ptprong1,
                           [](float px1, float py1) { return pttrack(px1, py1); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt,
                           [](float px0, float py0, float px1, float py1) { return ptcand2prong(px0, py0, px1, py1); });
DECLARE_SOA_DYNAMIC_COLUMN(DecaylengthXY, decaylengthxy,
                           [](float xvtxd, float yvtxd, float xvtxp, float yvtxp) { return declengthxy(xvtxd, yvtxd, xvtxp, yvtxp); });
DECLARE_SOA_DYNAMIC_COLUMN(Decaylength, decaylength,
                           [](float xvtxd, float yvtxd, float zvtxd, float xvtxp,
                              float yvtxp, float zvtxp) { return declength(xvtxd, yvtxd, zvtxd, xvtxp, yvtxp, zvtxp); });
} // namespace hfcandprong2

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

DECLARE_SOA_TABLE(HfCandProng2, "AOD", "HFCANDPRONG2",
                  collision::PosX, collision::PosY, collision::PosZ,
                  hfcandprong2::PxProng0, hfcandprong2::PyProng0, hfcandprong2::PzProng0,
                  hfcandprong2::PxProng1, hfcandprong2::PyProng1, hfcandprong2::PzProng1,
                  hfcandprong2::DecayVtxX, hfcandprong2::DecayVtxY, hfcandprong2::DecayVtxZ,
                  hfcandprong2::MassD0, hfcandprong2::MassD0bar,
                  hfcandprong2::PtProng0<hfcandprong2::PxProng0, hfcandprong2::PyProng0>,
                  hfcandprong2::PtProng1<hfcandprong2::PxProng1, hfcandprong2::PyProng1>,
                  hfcandprong2::Pt<hfcandprong2::PxProng0, hfcandprong2::PyProng0,
                                   hfcandprong2::PxProng1, hfcandprong2::PyProng1>,
                  hfcandprong2::DecaylengthXY<hfcandprong2::DecayVtxX, hfcandprong2::DecayVtxY,
                                              collision::PosX, collision::PosY>,
                  hfcandprong2::Decaylength<hfcandprong2::DecayVtxX, hfcandprong2::DecayVtxY,
                                            hfcandprong2::DecayVtxZ, collision::PosX,
                                            collision::PosY, collision::PosZ>);
} // namespace o2::aod

#endif // O2_ANALYSIS_SECONDARYVERTEXHF_H_
