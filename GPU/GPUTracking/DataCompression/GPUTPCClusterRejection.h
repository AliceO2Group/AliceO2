// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterRejection.h
/// \author David Rohr

#ifndef GPUTPCCLUSTERREJECTION_H
#define GPUTPCCLUSTERREJECTION_H

#include "GPUTPCGMMergerTypes.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterRejection {
  template <bool C, class T = void, class S = void>
  static constexpr inline bool GetProtectionStatus(int attach, bool& physics, bool& protect, T* counts = nullptr, S* mev200 = nullptr)
  {
    (void)counts; // Avoid incorrect -Wunused-but-set-parameter warning
    (void)mev200;
    if (attach == 0) {
      return false;
    } else if ((attach & gputpcgmmergertypes::attachGoodLeg) == 0) {
      if constexpr (C) {
        counts->nLoopers++;
      }
      return true;
    } else if (attach & gputpcgmmergertypes::attachHighIncl) {
      if constexpr (C) {
        counts->nHighIncl++;
      }
      return true;
    } else if (attach & gputpcgmmergertypes::attachTube) {
      protect = true;
      if constexpr (C) {
        if (*mev200) {
          counts->nTube200++;
        } else {
          counts->nTube++;
        }
      }
      return false;
    } else if ((attach & gputpcgmmergertypes::attachGood) == 0) {
      protect = true;
      if constexpr (C) {
        counts->nRejected++;
      }
      return false;
    } else {
      physics = true;
      return false;
    }
  }

  static constexpr inline bool GetIsRejected(int attach)
  {
    bool physics = false, protect = false;
    return GetProtectionStatus<false>(attach, physics, protect);
  }
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
