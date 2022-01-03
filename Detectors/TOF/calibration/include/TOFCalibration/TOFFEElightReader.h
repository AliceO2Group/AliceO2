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

#ifndef DETECTOR_TOFFEELIGHTREADER_H_
#define DETECTOR_TOFFEELIGHTREADER_H_

#include "Rtypes.h"
#include "DataFormatsTOF/TOFFEElightInfo.h"
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
