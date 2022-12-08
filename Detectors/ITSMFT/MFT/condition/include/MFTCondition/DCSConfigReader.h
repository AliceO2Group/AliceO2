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

#ifndef MFTDCSCONFIGPROCESSOR_H_
#define MFTDCSCONFIGPROCESSOR_H_

#include "Rtypes.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "MFTCondition/DCSConfigInfo.h"
#include "MFTCondition/DCSConfigUtils.h"
#include <gsl/span>
#include <memory>

/// @brief Class to read the MFT config information

namespace o2
{
namespace mft
{

using namespace o2::mft;

class DCSConfigReader
{

 public:
  DCSConfigReader() = default;  // default constructor
  ~DCSConfigReader() = default; // default destructor

  void init(bool);
  void loadConfig(gsl::span<const char> configBuf); // load MFT config
  void clear();

  const std::vector<o2::mft::DCSConfigInfo>& getConfigInfo() const { return mDCSConfig; }
  const o2::itsmft::NoiseMap& getNoiseMap() const { return mNoiseMap; }

 private:
  void parseConfig();

  std::string mParams;
  bool mVerbose = false;

  std::vector<o2::mft::DCSConfigInfo> mDCSConfig;
  o2::itsmft::NoiseMap mNoiseMap{936};

  DCSConfigUtils* mDCSUtils;

  ClassDefNV(DCSConfigReader, 1);
};

} // namespace mft
} // namespace o2

#endif
