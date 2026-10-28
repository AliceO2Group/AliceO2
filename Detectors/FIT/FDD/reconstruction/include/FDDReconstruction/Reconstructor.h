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

/// \file CollisionTimeRecoTask.h
/// \brief Definition of the FDD reconstruction
#ifndef ALICEO2_FDD_RECONSTRUCTOR_H
#define ALICEO2_FDD_RECONSTRUCTOR_H

#include <vector>
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/RecPoint.h"
#include "DataFormatsFIT/DeadChannelMap.h"
#include "DataFormatsFDD/FDDRecoConfig.h"
namespace o2
{
namespace fdd
{
class Reconstructor
{
 public:
  Reconstructor() = default;
  ~Reconstructor() = default;
  void process(o2::fdd::Digit const& digitBC,
               gsl::span<const o2::fdd::ChannelData> inChData,
               std::vector<o2::fdd::RecPoint>& RecPoints,
               std::vector<o2::fdd::ChannelDataFloat>& outChData);
  void finish();

  void setDeadChannelMap(o2::fit::DeadChannelMap const* deadChannelMap)
  {
    LOG(info) << "Updated dead channel map";
    mDeadChannelMap = deadChannelMap;
  }

 private:
  bool isInAmplitudeWindow(int16_t amplitude) {
    return amplitude >= FDDRecoConfig::Instance().ChannelAmplitudeLower && amplitude <= FDDRecoConfig::Instance().ChannelAmplitudeUpper;
  }
  bool isInTimeWindow(int16_t time) {
    return time >= FDDRecoConfig::Instance().ChannelTimeLower && time <= FDDRecoConfig::Instance().ChannelTimeUpper;
  }
  bool isChannelDataGood(uint8_t channelBits) {
    return FDDRecoConfig::Instance().isChannelDataGood(channelBits);
  }

  bool isInCollisionTimeAmplitudeWindow(int16_t amplitude) {
    return amplitude >= FDDRecoConfig::Instance().CollisionTimeAmplitudeLower && amplitude <= FDDRecoConfig::Instance().CollisionTimeAmpltudeUpper;
  }
  bool isInCollisionTimeTimeWindow(int16_t time) {
    return time >= FDDRecoConfig::Instance().CollisionTimeTimeLower && time <= FDDRecoConfig::Instance().CollisionTimeTimeUpper;
  }
  bool isChannelDataGoodForCollisionTime(uint8_t channelBits) {
    return FDDRecoConfig::Instance().isChannelDataGoodForCollisionTime(channelBits);
  }

  bool checkSwOrGateConditions(const ChannelData& channelData) {
    bool isAmplitudeInRange = channelData.mChargeADC >= FDDRecoConfig::Instance().SwTriggerAmplitudeLowThreshold 
      && channelData.mChargeADC <= FDDRecoConfig::Instance().SwTriggerAmplitudeHighThreshold;
    bool isInTimeWindow = std::abs(channelData.mTime) >= FDDRecoConfig::Instance().SwTriggerTimeLowThreshold
      && std::abs(channeData.mTime) <= FDDRecoConfig::Instance().SwTriggerTimeHighThreshold;
    bool isCfdInAdcGate = channelData.getFlag(ChannelData::EEventDataBit::kIsCFDinADCgate);
    return isAmplitudeInRange && isInTimeWindow && isCfdInAdcGate;
  }

  bool calculateSwTrigger(float meanTimeSideA, float meanTimeSideC, bool orGateSideA, bool orGateSideC)
  {
    return (meanTimeSideC - meanTimeSideA) >= FDDRecoConfig::SwTriggerTimeDiffMin
      && (meanTimeSideC - meanTimeSideA) <= FDDRecoConfig::SwTriggerTimeDiffMax
      && orGateSideA
      && orGateSideC;
  }
  static constexpr uint8_t sSideAMaxId = 7;

  o2::fit::DeadChannelMap const* mDeadChannelMap = nullptr;
  ClassDefNV(Reconstructor, 3);
};
} // namespace fdd
} // namespace o2
#endif
