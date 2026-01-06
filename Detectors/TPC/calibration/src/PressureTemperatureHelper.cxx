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

/// \file PressureTemperatureHelper.cxx
/// \brief Helper class to extract pressure and temperature
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#include "TPCCalibration/PressureTemperatureHelper.h"
#include "TPCBaseRecSim/CDBTypes.h"
#include "Framework/ProcessingContext.h"
#include "DataFormatsTPC/DCS.h"
#include "Framework/InputRecord.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataAllocator.h"

using namespace o2::tpc;
using namespace o2::framework;

void PressureTemperatureHelper::extractCCDBInputs(ProcessingContext& pc) const
{
  pc.inputs().get<dcs::Pressure*>("pressure");
  pc.inputs().get<dcs::Temperature*>("temperature");
}

bool PressureTemperatureHelper::accountCCDBInputs(const ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "PRESSURECCDB", 0)) {
    LOGP(info, "Updating pressure");
    const auto& pressure = ((dcs::Pressure*)obj);
    mPressure.second = pressure->robustPressure.time;
    mPressure.first = pressure->robustPressure.robustPressure;
    return true;
  }

  if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TEMPERATURECCDB", 0)) {
    LOGP(info, "Updating temperature");
    auto temp = *(dcs::Temperature*)obj;
    temp.fitTemperature(o2::tpc::Side::A, mFitIntervalMS, false);
    temp.fitTemperature(o2::tpc::Side::C, mFitIntervalMS, false);

    mTemperatureA.first.clear();
    mTemperatureC.first.clear();
    mTemperatureA.second.clear();
    mTemperatureC.second.clear();

    for (const auto& dp : temp.statsA.data) {
      mTemperatureA.first.emplace_back(toKelvin(dp.value.mean));
      mTemperatureA.second.emplace_back(dp.time);
    }

    for (const auto& dp : temp.statsC.data) {
      mTemperatureC.first.emplace_back(toKelvin(dp.value.mean));
      mTemperatureC.second.emplace_back(dp.time);
    }

    // check if temperature data is available
    if (mTemperatureA.first.empty() && mTemperatureC.first.empty()) {
      float temperature = toKelvin(temp.getMeanTempRaw());
      mTemperatureA.first.emplace_back(temperature);
      mTemperatureA.second.emplace_back(0);
      mTemperatureC.first.emplace_back(temperature);
      mTemperatureC.second.emplace_back(0);
      LOGP(warning, "No temperature data available from fit. Using average temperature {} K", temperature);
    }

    return true;
  }
  return false;
}

void PressureTemperatureHelper::requestCCDBInputs(std::vector<InputSpec>& inputs)
{
  addInput(inputs, {"pressure", o2::header::gDataOriginTPC, "PRESSURECCDB", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalPressure), {}, 1)});
  addInput(inputs, {"temperature", o2::header::gDataOriginTPC, "TEMPERATURECCDB", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalTemperature), {}, 1)});
}

void PressureTemperatureHelper::addInput(std::vector<InputSpec>& inputs, InputSpec&& isp)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

void PressureTemperatureHelper::setOutputs(std::vector<OutputSpec>& outputs)
{
  addOutput(outputs, {o2::header::gDataOriginTPC, o2::tpc::PressureTemperatureHelper::getDataDescriptionPressure(), 0, Lifetime::Timeframe});
  addOutput(outputs, {o2::header::gDataOriginTPC, o2::tpc::PressureTemperatureHelper::getDataDescriptionTemperature(), 0, Lifetime::Timeframe});
}

void PressureTemperatureHelper::addOutput(std::vector<OutputSpec>& outputs, OutputSpec&& osp)
{
  if (std::find(outputs.begin(), outputs.end(), osp) == outputs.end()) {
    outputs.emplace_back(osp);
  }
}

float PressureTemperatureHelper::interpolate(const std::vector<ULong64_t>& timestamps, const std::vector<float>& values, ULong64_t timestamp) const
{
  if (auto idxClosest = o2::math_utils::findClosestIndices(timestamps, timestamp)) {
    auto [idxLeft, idxRight] = *idxClosest;
    if (idxRight > idxLeft) {
      const auto x0 = timestamps[idxLeft];
      const auto x1 = timestamps[idxRight];
      const float y0 = values[idxLeft];
      const float y1 = values[idxRight];
      const float y = (y0 * (x1 - timestamp) + y1 * (timestamp - x0)) / (x1 - x0);
      return y;
    } else {
      return values[idxLeft];
    }
  }
  return 0; // this should never happen
}

void PressureTemperatureHelper::sendPTForTS(o2::framework::ProcessingContext& pc, const ULong64_t timestamp) const
{
  const float pressure = getPressure(timestamp);
  const auto temp = getTemperature(timestamp);
  LOGP(info, "Sending pressure {}, temperature A {} and temperature C {} for timestamp {}", pressure, temp.first, temp.second, timestamp);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTPC, o2::tpc::PressureTemperatureHelper::getDataDescriptionTemperature()}, temp);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTPC, o2::tpc::PressureTemperatureHelper::getDataDescriptionPressure()}, pressure);
}

float PressureTemperatureHelper::getTP(int64_t ts) const
{
  const float pressure = getPressure(ts);
  const auto temp = getMeanTemperature(ts);
  if (pressure <= 0) {
    LOGP(error, "Pressure {} is zero or negative, cannot compute T/P ratio for timestamp {}", pressure, ts);
    return 0;
  }
  const float tp = temp / pressure;
  return tp;
}

float PressureTemperatureHelper::getMeanTemperature(const ULong64_t timestamp) const
{
  const auto temp = getTemperature(timestamp);

  float sumT = 0;
  int w = 0;
  constexpr float minTemp = toKelvin(15);
  constexpr float maxTemp = toKelvin(25);
  if (auto t = temp.first; t > minTemp && t < maxTemp) {
    sumT += t;
    ++w;
  }
  if (auto t = temp.second; t > minTemp && t < maxTemp) {
    sumT += t;
    ++w;
  }

  if (w == 0) {
    constexpr float defaultTemp = toKelvin(19.6440f);
    LOGP(info, "Returning default temperature of {}K", defaultTemp);
    return defaultTemp;
  }

  const float meanT = sumT / w;
  return meanT;
}

std::pair<ULong64_t, ULong64_t> PressureTemperatureHelper::getMinMaxTime() const
{
  ULong64_t minTime = std::numeric_limits<ULong64_t>::max();
  ULong64_t maxTime = 0;

  if (!mPressure.first.empty()) {
    minTime = std::min(minTime, mPressure.second.front());
    maxTime = std::max(maxTime, mPressure.second.back());
  }
  if (!mTemperatureA.first.empty()) {
    minTime = std::min(minTime, mTemperatureA.second.front());
    maxTime = std::max(maxTime, mTemperatureA.second.back());
  }
  if (!mTemperatureC.first.empty()) {
    minTime = std::min(minTime, mTemperatureC.second.front());
    maxTime = std::max(maxTime, mTemperatureC.second.back());
  }

  return {minTime, maxTime};
}
