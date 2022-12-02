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

// ROOT includes
#include "TLinearFitter.h"
#include "TVectorD.h"

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
    } else {
      LOGP(warning, "Unknown data point: {}", alias);
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
    LOGP(warning, "Unexpected delivery type for {}: {}", dp.id.get_alias(), dp.id.get_type());
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
      LOGP(info, "Delivery type for STATUS ({}): {}", alias, type);
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
      LOGP(info, "Delivery type for current, voltage ({}): {}", alias, type);
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

void DCSProcessor::finalizeSlot()
{
  finalizeTemperature();
  finalizeHighVoltage();
  finalizeGas();
  mHasData = false;
}

void DCSProcessor::fitTemperature(Side side)
{
  //// temperature fits in x-y
  TLinearFitter fitter(3, "x0 ++ x1 ++ x2");
  bool nextInterval = true;
  std::array<size_t, dcs::Temperature::SensorsPerSide> startPos{};
  const size_t sensorOffset = (side == Side::C) ? dcs::Temperature::SensorsPerSide : 0;
  dcs::TimeStampType refTime = getMinTime(mTemperature.raw);

  while (nextInterval) {
    // TODO: check if we should use refTime
    dcs::TimeStampType firstTime = std::numeric_limits<dcs::TimeStampType>::max();

    nextInterval = false;
    for (size_t iSensor = 0; iSensor < dcs::Temperature::SensorsPerSide; ++iSensor) {
      const auto& sensor = mTemperature.raw[iSensor + sensorOffset];

      LOGP(debug, "sensor {}, start {}, size {}", sensor.sensorNumber, startPos[iSensor], sensor.data.size());
      while (startPos[iSensor] < sensor.data.size()) {
        const auto& dataPoint = sensor.data[startPos[iSensor]];
        if ((dataPoint.time - refTime) >= mFitInterval) {
          LOGP(debug, "sensor {}, {} - {} >= {}", sensor.sensorNumber, dataPoint.time, refTime, mFitInterval);
          break;
        }
        nextInterval = true;
        firstTime = std::min(firstTime, dataPoint.time);
        const auto temperature = dataPoint.value;
        // sanity check
        if (temperature < 15 || temperature > 25) {
          ++startPos[iSensor];
          continue;
        }
        const auto& pos = dcs::Temperature::SensorPosition[iSensor + sensorOffset];
        double x[] = {1., double(pos.x), double(pos.y)};
        fitter.AddPoint(x, temperature, 1);
        ++startPos[iSensor];
      }
    }
    if (firstTime < std::numeric_limits<dcs::TimeStampType>::max()) {
      fitter.Eval();
      LOGP(info, "Side {}, fit interval {} - {} with {} points", int(side), refTime, refTime + mFitInterval - 1, fitter.GetNpoints());

      auto& stats = (side == Side::A) ? mTemperature.statsA : mTemperature.statsC;
      auto& stat = stats.data.emplace_back();
      stat.time = firstTime;
      stat.value.mean = fitter.GetParameter(0);
      stat.value.gradX = fitter.GetParameter(1);
      stat.value.gradY = fitter.GetParameter(2);

      fitter.ClearPoints();
      refTime += mFitInterval;
    }
  }
}

void DCSProcessor::finalizeTemperature()
{
  mTemperature.sortAndClean();
  fitTemperature(Side::A);
  fitTemperature(Side::C);
  mTimeTemperature = {getMinTime(mTemperature.raw), getMaxTime(mTemperature.raw)};
}

void DCSProcessor::finalizeHighVoltage()
{
  mHighVoltage.sortAndClean();

  auto minTime = getMinTime(mHighVoltage.currents);
  minTime = std::min(minTime, getMinTime(mHighVoltage.voltages));
  minTime = std::min(minTime, getMinTime(mHighVoltage.states));

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

void DCSProcessor::writeDebug()
{
  if (!mDebugStream) {
    mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugOutputName.data(), "recreate");
  }

  *mDebugStream << "dcs"
                << "Temperature=" << mTemperature
                << "HV=" << mHighVoltage
                << "Gas=" << mGas
                << "\n";
}

void DCSProcessor::finalize()
{
  if (mDebugStream) {
    mDebugStream->Close();
  }
}
