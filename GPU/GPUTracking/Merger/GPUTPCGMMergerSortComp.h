// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMMergerSortComp.h
/// \brief Orderings of the merger's sorts, for the kernels and for the backends that run these sorts themselves

#ifndef GPUTPCGMMERGERSORTCOMP_H
#define GPUTPCGMMERGERSORTCOMP_H

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergerTypes.h"
#include "GPUTPCGMMerger.h"

namespace o2::gpu::internal
{
struct MergeLooperParam {
  float refz;
  float x;
  float y;
  uint32_t id;
};

struct MergeBorderTracks_compMax {
  GPUd() bool operator()(const gputpcgmmergertypes::GPUTPCGMBorderRange& a, const gputpcgmmergertypes::GPUTPCGMBorderRange& b)
  {
    return GPUCA_DETERMINISTIC_CODE((a.fMax != b.fMax) ? (a.fMax < b.fMax) : (a.fId < b.fId), a.fMax < b.fMax);
  }
};
struct MergeBorderTracks_compMin {
  GPUd() bool operator()(const gputpcgmmergertypes::GPUTPCGMBorderRange& a, const gputpcgmmergertypes::GPUTPCGMBorderRange& b)
  {
    return GPUCA_DETERMINISTIC_CODE((a.fMin != b.fMin) ? (a.fMin < b.fMin) : (a.fId < b.fId), a.fMin < b.fMin);
  }
};

struct GPUTPCGMMergerSortTracks_comp {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUhd() GPUTPCGMMergerSortTracks_comp(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const int32_t aa, const int32_t bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    if (a.OK() != b.OK()) {
      return a.OK();
    }
    if (a.CCE() != b.CCE()) {
      return a.CCE() > b.CCE();
    }
    GPUCA_DETERMINISTIC_CODE( // clang-format off
      if (a.NClusters() != b.NClusters()) {
        return a.NClusters() > b.NClusters();
      } if (CAMath::Abs(a.GetParam().GetQPt()) != CAMath::Abs(b.GetParam().GetQPt())) {
        return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
      } if (a.GetParam().GetY() != b.GetParam().GetY()) {
        return a.GetParam().GetY() > b.GetParam().GetY();
      }
      return aa > bb;
    , // !GPUCA_DETERMINISTIC_CODE
      return a.NClusters() > b.NClusters();
    ) // clang-format on
  }
};

struct GPUTPCGMMergerSortTracksQPt_comp {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUhd() GPUTPCGMMergerSortTracksQPt_comp(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const int32_t aa, const int32_t bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    GPUCA_DETERMINISTIC_CODE( // clang-format off
      if (CAMath::Abs(a.GetParam().GetQPt()) != CAMath::Abs(b.GetParam().GetQPt())) {
        return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
      } if (a.GetParam().GetY() != b.GetParam().GetY()) {
        return a.GetParam().GetY() > b.GetParam().GetY();
      }
      return a.GetParam().GetZ() > b.GetParam().GetZ();
    , // !GPUCA_DETERMINISTIC_CODE
      return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
    ) // clang-format on
  }
};

struct GPUTPCGMMergerMergeLoopers_comp {
  GPUd() bool operator()(const MergeLooperParam& a, const MergeLooperParam& b)
  {
    return GPUCA_DETERMINISTIC_CODE(CAMath::Abs(a.refz) != CAMath::Abs(b.refz) ? CAMath::Abs(a.refz) < CAMath::Abs(b.refz) : a.id < b.id, CAMath::Abs(a.refz) < CAMath::Abs(b.refz));
  }
};

struct GPUTPCGMO2OutputSort_comp {
  GPUd() bool operator()(const GPUTPCGMMerger::tmpSort& a, const GPUTPCGMMerger::tmpSort& b)
  {
    return GPUCA_DETERMINISTIC_CODE(a.y != b.y ? a.y > b.y : a.x > b.x, a.y > b.y);
  }
};

} // namespace o2::gpu::internal

#endif
