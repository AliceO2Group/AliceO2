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

#ifndef O2_FRAMEWORK_TIMINGHELPERS_H_
#define O2_FRAMEWORK_TIMINGHELPERS_H_
#include <functional>
#include <cstdint>

using uv_loop_t = struct uv_loop_s;

namespace o2::framework
{
struct TimingHelpers {
  /// Return a function which can be used to retrieve the base timestamp and the
  /// associated fast offset for the realtime clock.
  static std::function<void(int64_t& base, int64_t& offset)> defaultRealtimeBaseConfigurator(uint64_t offset, uv_loop_t* loop);
  static std::function<int64_t(int64_t base, int64_t offset)> defaultCPUTimeConfigurator(uv_loop_t* loop);

  /// Milliseconds since epoch, using the standard C++ clock.
  /// This will do a system call every minute or so to synchronize the clock
  /// and minimise drift.
  static int64_t getRealtimeSinceEpochStandalone();
};
} // namespace o2::framework
#endif // O2_FRAMEWORK_TIMINGHELPERS_H_
