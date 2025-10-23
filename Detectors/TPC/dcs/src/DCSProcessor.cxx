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

/// @file DCSProcessor.cxx
/// @brief TPC DCS data point processor
/// @author Jens Wiechula

#include <string_view>

// O2 includes
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "Framework/Logger.h"

#include "TPCdcs/DCSProcessor.h"

using namespace o2::tpc;
using namespace o2::dcs;

void DCSProcessor::process(const gsl::span<const DPCOM> dps)
{
  if (dps.size() == 0) {
    return;
  }

  mHasData = true;

  using namespace std::literals;
  constexpr auto TEMP_ID{"TPC_PT"sv};
  constexpr auto HV_ID{"TPC_HV"sv};
  constexpr auto GAS_ID1{"TPC_GC"sv};
  constexpr auto GAS_ID2{"TPC_An"sv};
  constexpr auto PRESS_ID1{"Cavern"sv};
  constexpr auto PRESS_ID2{"Surfac"sv};

  for (const auto& dp : dps) {
    const std::string_view alias(dp.id.get_alias());
    const auto id = alias.substr(0, 6);
    if (id == TEMP_ID) {
      LOGP(debug, "Temperature DP: {}", alias);
      fillTemperature(dp);
    } else if (id == HV_ID) {
      LOGP(debug, "HV DP: {}", alias);
      fillHV(dp);
    } else if (id == GAS_ID1 || id == GAS_ID2) {
      LOGP(debug, "Gas DP: {}", alias);
      fillGas(dp);
    } else if (id == PRESS_ID1 || id == PRESS_ID2) {
      LOGP(debug, "Pressure DP: {}", alias);
      fillPressure(dp);
    } else {
      LOGP(warning, "Unknown data point: {} with id {}", alias, id);
    }
  }
}

float DCSProcessor::getValueF(const DPCOM& dp) const
{
  if (dp.id.get_type() == DeliveryType::DPVAL_FLOAT) {
    return o2::dcs::getValue<float>(dp);
  } else if (dp.id.get_type() == DeliveryType::DPVAL_DOUBLE) {
    return static_cast<float>(o2::dcs::getValue<double>(dp));
  } else {
    LOGP(warning, "Unexpected delivery type for {}: {}", dp.id.get_alias(), (int)dp.id.get_type());
  }

  return 0.f;
}

void DCSProcessor::fillTemperature(const DPCOM& dp)
{
  const std::string_view alias(dp.id.get_alias());
  const auto value = getValueF(dp);
  const auto time = dp.data.get_epoch_time();
  mTemperature.fill(alias, time, value);
}

void DCSProcessor::fillHV(const DPCOM& dp)
{
  const std::string_view alias(dp.id.get_alias());
  const auto time = dp.data.get_epoch_time();

  const auto type = dp.id.get_type();
  if (alias.back() == 'S') { //
    uint32_t value;
    // TODO: Remove once type is clear
    static bool statTypePrinted = false;
    if (!statTypePrinted) {
      LOGP(info, "Delivery type for STATUS ({}): {}", alias, (int)type);
      statTypePrinted = true;
    }
    if (type == DeliveryType::DPVAL_UINT) {
      value = o2::dcs::getValue<uint32_t>(dp);
    } else if (type == DeliveryType::DPVAL_INT) {
      value = uint32_t(o2::dcs::getValue<int32_t>(dp));
    } else {
      value = uint32_t(getValueF(dp));
    }
    mHighVoltage.fillStatus(alias, time, value);
  } else {
    // TODO: Remove once type is clear
    static bool uiTypePrinted = false;
    if (!uiTypePrinted) {
      LOGP(info, "Delivery type for current, voltage ({}): {}", alias, (int)type);
      uiTypePrinted = true;
    }
    const auto value = getValueF(dp);
    mHighVoltage.fillUI(alias, time, value);
  }
}

void DCSProcessor::fillGas(const DPCOM& dp)
{
  const std::string_view alias(dp.id.get_alias());
  const auto value = getValueF(dp);
  const auto time = dp.data.get_epoch_time();
  mGas.fill(alias, time, value);
}

void DCSProcessor::fillPressure(const DPCOM& dp)
{
  const std::string_view alias(dp.id.get_alias());
  const auto value = getValueF(dp);
  const auto time = dp.data.get_epoch_time();
  mPressure.fill(alias, time, value);
}

void DCSProcessor::finalizeSlot()
{
  finalizeTemperature();
  finalizeHighVoltage();
  finalizeGas();
  finalizePressure();
  mHasData = false;
}

void DCSProcessor::finalizeTemperature()
{
  mTemperature.sortAndClean();
  mTemperature.fitTemperature(Side::A, mFitInterval, mRoundToInterval);
  mTemperature.fitTemperature(Side::C, mFitInterval, mRoundToInterval);
  mTimeTemperature = {getMinTime(mTemperature.raw, mRoundToInterval, mFitInterval), getMaxTime(mTemperature.raw)};
}

void DCSProcessor::finalizeHighVoltage()
{
  mHighVoltage.sortAndClean();

  auto minTime = getMinTime(mHighVoltage.currents, mRoundToInterval, mFitInterval);
  minTime = std::min(minTime, getMinTime(mHighVoltage.voltages, mRoundToInterval, mFitInterval));
  minTime = std::min(minTime, getMinTime(mHighVoltage.states, mRoundToInterval, mFitInterval));

  auto maxTime = getMaxTime(mHighVoltage.currents);
  maxTime = std::max(maxTime, getMaxTime(mHighVoltage.voltages));
  maxTime = std::max(maxTime, getMaxTime(mHighVoltage.states));

  mTimeHighVoltage = {minTime, maxTime};
}

void DCSProcessor::finalizeGas()
{
  mGas.sortAndClean();
  mTimeGas = {mGas.getMinTime(), mGas.getMaxTime()};
}

void DCSProcessor::finalizePressure()
{
  mPressure.sortAndClean();
  mTimePressure = {mPressure.getMinTime(), mPressure.getMaxTime()};
  // if there is data perform the processing
  if (mTimePressure.last > 0) {
    mPressure.makeRobustPressure(mPressureInterval, mPressureIntervalRef, mTimePressure.first, mTimePressure.last);
  }
}

void DCSProcessor::writeDebug()
{
  if (!mDebugStream) {
    mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugOutputName.data(), "recreate");
  }

  *mDebugStream << "dcs"
                << "Temperature=" << mTemperature
                << "HV=" << mHighVoltage
                << "Gas=" << mGas
                << "Pressure=" << mPressure
                << "\n";
}

void DCSProcessor::finalize()
{
  if (mDebugStream) {
    mDebugStream->Close();
  }
}
