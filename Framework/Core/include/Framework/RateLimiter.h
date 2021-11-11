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

#ifndef O2_FRAMERWORK_CORE_RATELIMITER_H
#define O2_FRAMERWORK_CORE_RATELIMITER_H

#include "Framework/ProcessingContext.h"
#include <cstddef>
#include <cstdint>

namespace o2::framework
{
class RateLimiter
{
 public:
  void check(ProcessingContext& ctx, int maxInFlight, size_t minSHM);

 private:
  int64_t mConsumedTimeframes = 0;
  int64_t mSentTimeframes = 0;
};
} // namespace o2::framework

#endif
