// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTOR_TOFFEELIGHTCONFIG_H_
#define DETECTOR_TOFFEELIGHTCONFIG_H_

#include "Rtypes.h"
#include "TOFBase/Geo.h"
#include <array>

using namespace o2::tof;

namespace o2
{
namespace tof
{

struct TOFFEEchannelConfig {

  enum EStatus_t {
    kStatusEnabled = 0x1
  };
  unsigned char mStatus = 0x0; // status
  int mMatchingWindow = 0;     // matching window [ns] // can this be int32?
  int mLatencyWindow = 0;      // latency window [ns] // can this be int32?
  TOFFEEchannelConfig() = default;
  bool isEnabled() const { return mStatus & kStatusEnabled; };

  ClassDefNV(TOFFEEchannelConfig, 1);
};

//_____________________________________________________________________________

struct TOFFEEtriggerConfig {

  uint64_t mStatusMap = 0; // status // can it be uint32?
  TOFFEEtriggerConfig() = default;

  ClassDefNV(TOFFEEtriggerConfig, 1);
};

//_____________________________________________________________________________

struct TOFFEElightConfig {

  static constexpr int NCHANNELS = 172800;
  static constexpr int NTRIGGERMAPS = Geo::kNCrate;

  int mVersion = 0;   // version
  int mRunNumber = 0; // run number
  int mRunType = 0;   // run type
  // std::array<TOFFEEchannelConfig, NCHANNELS> mChannelConfig;
  TOFFEEchannelConfig mChannelConfig[Geo::kNCrate][Geo::kNTRM][Geo::kNChain][Geo::kNTdc][Geo::kNCh];
  std::array<TOFFEEtriggerConfig, NTRIGGERMAPS> mTriggerConfig;
  TOFFEElightConfig() = default;
  TOFFEEchannelConfig* getChannelConfig(int icrate, int itrm, int ichain, int itdc, int ich);
  TOFFEEtriggerConfig* getTriggerConfig(int idx) { return idx < NTRIGGERMAPS ? &mTriggerConfig[idx] : nullptr; }

  ClassDefNV(TOFFEElightConfig, 1);
};

} // namespace tof
} // namespace o2

#endif
