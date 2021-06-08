// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTOR_TOFFEELIGHTREADER_H_
#define DETECTOR_TOFFEELIGHTREADER_H_

#include "Rtypes.h"
#include "TOFCalibration/TOFFEElightConfig.h"
#include "TOFBase/Geo.h"
#include <gsl/span>
#include <memory>

/// @brief Class to read the TOFFEElight information

namespace o2
{
namespace tof
{

using namespace o2::tof;

struct TOFFEElightInfo {

  int mVersion = -1;   // version
  int mRunNumber = -1; // run number
  int mRunType = -1;   // run type
  std::array<bool, Geo::NCHANNELS> mChannelEnabled;
  std::array<int, Geo::NCHANNELS> mMatchingWindow;                    // can it be int32_t?
  std::array<int, Geo::NCHANNELS> mLatencyWindow;                     // can it be int32_t?
  std::array<uint64_t, TOFFEElightConfig::NTRIGGERMAPS> mTriggerMask; // trigger mask, can it be uint32_t?
  TOFFEElightInfo()
  {
    mVersion = -1;
    mRunNumber = -1;
    mRunType = -1;
    mChannelEnabled.fill(false);
    mMatchingWindow.fill(0);
    mLatencyWindow.fill(0);
    mTriggerMask.fill(0);
  }

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
  bool getChannelEnabled(int idx) const { return idx < Geo::NCHANNELS ? mChannelEnabled[idx] : false; }
  int getMatchingWindow(int idx) const { return idx < Geo::NCHANNELS ? mMatchingWindow[idx] : 0; }
  int getLatencyWindow(int idx) const { return idx < Geo::NCHANNELS ? mLatencyWindow[idx] : 0; }
  uint64_t getTriggerMask(int ddl) const { return ddl < TOFFEElightConfig::NTRIGGERMAPS ? mTriggerMask[ddl] : 0; }

  ClassDefNV(TOFFEElightInfo, 1);
};

class TOFFEElightReader
{

 public:
  TOFFEElightReader() = default;  // default constructor
  ~TOFFEElightReader() = default; // default destructor

  void loadFEElightConfig(const char* fileName); // load FEElight config
  void loadFEElightConfig(gsl::span<const char> configBuf); // load FEElight config
  int parseFEElightConfig(bool verbose = false);            // parse FEElight config

  const TOFFEElightConfig* getTOFFEElightConfig() { return mFEElightConfig; }
  TOFFEElightInfo& getTOFFEElightInfo() { return mFEElightInfo; }

 private:
  const TOFFEElightConfig* mFEElightConfig = nullptr; // FEElight config
  TOFFEElightInfo mFEElightInfo;     // what will go to CCDB
  std::unique_ptr<char[]> mFileLoadBuff; // temporary buffer to be used when we load the configuration from file

  ClassDefNV(TOFFEElightReader, 1);
};

} // namespace tof
} // namespace o2

#endif
