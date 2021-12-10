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

#ifndef DETECTOR_TOFFEELIGHTINFO_H_
#define DETECTOR_TOFFEELIGHTINFO_H_

#include "Rtypes.h"
#include <array>

/// @brief Class providing TOFFEElight information

namespace o2
{
namespace tof
{

struct TOFFEElightInfo {
  static constexpr int NTRIGGERMAPS = 72;
  static constexpr int NCHANNELS = 157248;

  int mVersion = -1;   // version
  int mRunNumber = -1; // run number
  int mRunType = -1;   // run type
  std::array<bool, NCHANNELS> mChannelEnabled{false};
  std::array<int, NCHANNELS> mMatchingWindow{0};      // can it be int32_t?
  std::array<int, NCHANNELS> mLatencyWindow{0};       // can it be int32_t?
  std::array<uint64_t, NTRIGGERMAPS> mTriggerMask{0}; // trigger mask, can it be uint32_t?
  TOFFEElightInfo() = default;

  void resetAll()
  {
    mVersion = -1;
    mRunNumber = -1;
    mRunType = -1;
    mChannelEnabled.fill(false);
    mMatchingWindow.fill(0);
    mLatencyWindow.fill(0);
    mTriggerMask.fill(0);
  }

  int getVersion() const { return mVersion; }
  int getRunNumber() const { return mRunNumber; }
  int getRunType() const { return mRunType; }
  bool getChannelEnabled(int idx) const { return idx < NCHANNELS ? mChannelEnabled[idx] : false; }
  int getMatchingWindow(int idx) const { return idx < NCHANNELS ? mMatchingWindow[idx] : 0; }
  int getLatencyWindow(int idx) const { return idx < NCHANNELS ? mLatencyWindow[idx] : 0; }
  uint64_t getTriggerMask(int ddl) const { return ddl < NTRIGGERMAPS ? mTriggerMask[ddl] : 0; }

  ClassDefNV(TOFFEElightInfo, 1);
};

} // namespace tof
} // namespace o2

#endif
