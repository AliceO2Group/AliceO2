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
DECLARE_SOA_DYNAMIC_COLUMN(Value, value,
                           [](int val) { return val; });
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
DECLARE_SOA_COLUMN(MassD0, massD0, float);
DECLARE_SOA_COLUMN(MassD0bar, massD0bar, float);
} // namespace hfcandprong2

DECLARE_SOA_TABLE(HfTrackIndexProng2, "AOD", "HFTRACKIDXP2",
                  hftrackindexprong2::CollisionId,
                  hftrackindexprong2::Index0Id,
                  hftrackindexprong2::Index1Id,
                  hftrackindexprong2::HFflag,
                  hftrackindexprong2::Value<hftrackindexprong2::Index0Id>);

DECLARE_SOA_TABLE(HfTrackIndexProng3, "AOD", "HFTRACKIDXP3",
                  hftrackindexprong3::CollisionId,
                  hftrackindexprong3::Index0Id,
                  hftrackindexprong3::Index1Id,
                  hftrackindexprong3::Index2Id,
                  hftrackindexprong3::HFflag);

DECLARE_SOA_TABLE(HfCandProng2, "AOD", "HFCANDPRONG2",
                  //hfcandprong2::CollisionId,
                  hfcandprong2::MassD0, hfcandprong2::MassD0bar);
} // namespace o2::aod


#endif // O2_ANALYSIS_SECONDARYVERTEXHF_H_
