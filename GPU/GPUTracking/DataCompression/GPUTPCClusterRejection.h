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

/// \file GPUTPCClusterRejection.h
/// \author David Rohr

#ifndef GPUTPCCLUSTERREJECTION_H
#define GPUTPCCLUSTERREJECTION_H

#include "GPUTPCGMMergerTypes.h"
#include "GPUCommonMath.h"

namespace o2::gpu
{
struct GPUTPCClusterRejection {
  template <class T, class S>
  GPUdi() static bool IsTrackRejected(const T& trk, const S& param)
  {
    return CAMath::Abs(trk.GetParam().GetQPt() * param.qptB5Scaler) > param.rec.tpc.rejectQPtB5 || trk.MergedLooper();
  }

  template <bool C, class T = void, class S = void>
  GPUdi() static constexpr bool GetRejectionStatus(int32_t attach, bool& physics, T* counts = nullptr, S* mev200 = nullptr)
  {
    (void)counts; // FIXME: Avoid incorrect -Wunused-but-set-parameter warning
    (void)mev200;
    bool retVal = false;
    if (attach == 0) {
      retVal = false;
    } else if ((attach & gputpcgmmergertypes::attachGoodLeg) == 0) {
      if constexpr (C) {
        counts->nLoopers++;
      }
      retVal = true;
    } else if (attach & gputpcgmmergertypes::attachHighIncl) {
      if constexpr (C) {
        counts->nHighIncl++;
      }
      retVal = true;
    } else if (attach & gputpcgmmergertypes::attachTube) {
      if constexpr (C) {
        if (*mev200) {
          counts->nTube200++;
        } else {
          counts->nTube++;
        }
      }
      retVal = false;
    } else if ((attach & gputpcgmmergertypes::attachGood) == 0) {
      if constexpr (C) {
        counts->nRejected++;
      }
      retVal = false;
    } else {
      physics = true;
      retVal = false;
    }

    if (attach & gputpcgmmergertypes::attachProtect) {
      retVal = false;
    }
    return retVal;
  }

  GPUdi() static constexpr bool GetIsRejected(int32_t attach)
  {
    bool physics = false;
    return GetRejectionStatus<false>(attach, physics);
  }
};
} // namespace o2::gpu

#endif
