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

/// @brief Class to read the TOFFEElight information

namespace o2
{
namespace tof
{

using namespace o2::tof;

struct TOFFEElightInfo {

  std::array<bool, Geo::NCHANNELS> mChannelEnabled;
  std::array<int, Geo::NCHANNELS> mMatchingWindow;                    // can it be int32_t?
  std::array<int, Geo::NCHANNELS> mLatencyWindow;                     // can it be int32_t?
  std::array<uint64_t, TOFFEElightConfig::NTRIGGERMAPS> mTriggerMask; // trigger mask, can it be uint32_t?
  TOFFEElightInfo()
  {
    mChannelEnabled.fill(false);
    mMatchingWindow.fill(0);
    mLatencyWindow.fill(0);
    mTriggerMask.fill(0);
  }

  void resetAll()
  {
    mChannelEnabled.fill(false);
    mMatchingWindow.fill(0);
    mLatencyWindow.fill(0);
    mTriggerMask.fill(0);
  }

  bool getChannelEnabled(int idx) { return idx < Geo::NCHANNELS ? mChannelEnabled[idx] : false; }
  int getMatchingWindow(int idx) { return idx < Geo::NCHANNELS ? mMatchingWindow[idx] : 0; }
  int getLatencyWindow(int idx) { return idx < Geo::NCHANNELS ? mLatencyWindow[idx] : 0; }
  uint64_t getTriggerMask(int ddl) { return ddl < TOFFEElightConfig::NTRIGGERMAPS ? mTriggerMask[ddl] : 0; }

  ClassDefNV(TOFFEElightInfo, 1);
};

class TOFFEElightReader
{

 public:
  TOFFEElightReader() = default;  // default constructor
  ~TOFFEElightReader() = default; // default destructor

  void loadFEElightConfig(const char* fileName); // load FEElight config
  //  void createFEElightConfig(const char *filename); // create FEElight config
  int parseFEElightConfig(); // parse FEElight config

  TOFFEElightConfig& getTOFFEElightConfig() { return mFEElightConfig; }
  TOFFEElightInfo& getTOFFEElightInfo() { return mFEElightInfo; }

 private:
  TOFFEElightConfig mFEElightConfig; // FEElight config
  TOFFEElightInfo mFEElightInfo;     // what will go to CCDB

  ClassDefNV(TOFFEElightReader, 1);
};

} // namespace tof
} // namespace o2

#endif
