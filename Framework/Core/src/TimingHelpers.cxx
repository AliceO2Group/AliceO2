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

#include "Framework/TimingHelpers.h"
#include <uv.h>
#include <chrono>

namespace o2::framework
{

std::function<void(int64_t&, int64_t&)> TimingHelpers::defaultRealtimeBaseConfigurator(uint64_t startTimeOffset, uv_loop_t* loop)
{
  return [startTimeOffset, loop](int64_t& base, int64_t& offset) {
    uv_update_time(loop);
    base = startTimeOffset + uv_now(loop);
    offset = uv_now(loop);
  };
}

// Implement getTimestampConfigurator based on getRealtimeBaseConfigurator
std::function<int64_t(int64_t, int64_t)> TimingHelpers::defaultCPUTimeConfigurator(uv_loop_t* loop)
{
  return [loop](int64_t base, int64_t offset) -> int64_t {
    return base + (uv_now(loop) - offset);
  };
}

int64_t TimingHelpers::getRealtimeSinceEpochStandalone()
{
  /// Get milliseconds from epoch minimising the amount of system calls
  /// we do to get the time.
  static int64_t base = 0;
  static int64_t offset = 0;

  // Get the time with the std system clock
  if (base == 0) {
    base = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    offset = uv_hrtime();
  }
  uint64_t current = uv_hrtime();
  if (current - offset > 60000000000) {
    base = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    offset = uv_hrtime();
    current = offset;
  }
  return base + (current - offset) / 1000000;
}

} // namespace o2::framework
