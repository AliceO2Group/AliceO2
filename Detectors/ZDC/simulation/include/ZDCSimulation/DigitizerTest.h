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

#ifndef DETECTORS_ZDC_DIGITIZER_TEST_H_
#define DETECTORS_ZDC_DIGITIZER_TEST_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCSimulation/Digitizer.h"
#include "ZDCSimulation/SimCondition.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include <vector>
#include <array>
#include <deque>
#include <bitset>

/// \file DigitizerTest.h
/// \brief Fast digitization of a signal of given amplitude and time in pre-allocated bunch containers. This class assumes that the waveforms used in the simulation have been acquired with a system locked to the LHC clock
/// \author cortese@to.infn.it

namespace o2
{
namespace zdc
{

class SimCondition;

class DigitizerTest
{
 public:
  DigitizerTest() = default;
  ~DigitizerTest() = default;
  void setCCDBServer(const std::string& s) { mCCDBServer = s; }
  void setMask(uint32_t ich, uint32_t mask);
  void init();
  o2::zdc::Digitizer::BCCache& getCreateBCCache(const o2::InteractionRecord& ir);
  double add(int ic, float myAmp, const o2::InteractionRecord irpk,
             float myShift, bool hasJitter = true);
  const std::vector<o2::zdc::OrbitData>& getZDCOrbitData() const { return zdcOrbitData; }
  const std::vector<o2::zdc::BCData>& getZDCBCData() const { return zdcBCData; }
  const std::vector<o2::zdc::ChannelData>& getZDCChannelData() const { return zdcChData; }
  void digitize();
  void clear();

 private:
  // TODO: these should be common with Digitizer.h
  static constexpr int BCCacheMin = -1, BCCacheMax = 5, NBC2Cache = 1 + BCCacheMax - BCCacheMin;
  std::string mCCDBServer = "";
  SimCondition* mSimCondition = nullptr;
  ModuleConfig* mModuleConfig = nullptr;
  uint32_t mMask[NChannels] = {0};
  std::deque<Digitizer::BCCache> mCache; // cached BCs data
  BCData& getCreateBCData(const o2::InteractionRecord& ir);
  Digitizer::BCCache* getBCCache(const o2::InteractionRecord& ir);
  std::vector<o2::zdc::OrbitData> zdcOrbitData;
  std::vector<o2::zdc::BCData> zdcBCData;
  std::vector<o2::zdc::ChannelData> zdcChData;
};

} // namespace zdc
} // namespace o2

#endif /* DETECTORS_ZDC_DIGITIZER_TEST_H_ */
