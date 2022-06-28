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

/// @file   DCS.cxx
/// @author Jens Wiechula
/// @brief  DCS data point data formats

#include <limits>

#include "DataFormatsTPC/DCS.h"

using namespace o2::tpc::dcs;

//==============================================================================
//
//
const std::unordered_map<std::string, int> Temperature::SensorNameMap = {
  {"TPC_PT_351_TEMPERATURE", 0},
  {"TPC_PT_376_TEMPERATURE", 1},
  {"TPC_PT_415_TEMPERATURE", 2},
  {"TPC_PT_447_TEMPERATURE", 3},
  {"TPC_PT_477_TEMPERATURE", 4},
  {"TPC_PT_488_TEMPERATURE", 5},
  {"TPC_PT_537_TEMPERATURE", 6},
  {"TPC_PT_575_TEMPERATURE", 7},
  {"TPC_PT_589_TEMPERATURE", 8},
  {"TPC_PT_629_TEMPERATURE", 9},
  {"TPC_PT_664_TEMPERATURE", 10},
  {"TPC_PT_695_TEMPERATURE", 11},
  {"TPC_PT_735_TEMPERATURE", 12},
  {"TPC_PT_757_TEMPERATURE", 13},
  {"TPC_PT_797_TEMPERATURE", 14},
  {"TPC_PT_831_TEMPERATURE", 15},
  {"TPC_PT_851_TEMPERATURE", 16},
  {"TPC_PT_895_TEMPERATURE", 17},
};

Temperature::Temperature() noexcept : raw(SensorsPerSide * SIDES)
{
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i].sensorNumber = i;
  }
}

//==============================================================================
//
//
HV::HV() noexcept : voltages(2 * GEMSTACKSPERSECTOR * GEMSPERSTACK * SECTORSPERSIDE * SIDES),
                    currents(2 * GEMSTACKSPERSECTOR * GEMSPERSTACK * SECTORSPERSIDE * SIDES),
                    states(GEMSTACKSPERSECTOR * SECTORSPERSIDE * SIDES)
{
  for (size_t i = 0; i < voltages.size(); ++i) {
    voltages[i].sensorNumber = i;
    currents[i].sensorNumber = i;
  }
  for (size_t i = 0; i < states.size(); ++i) {
    states[i].sensorNumber = i;
  }
}

const std::unordered_map<HV::StackState, std::string> HV::StackStateNameMap =
  {
    {StackState::OFF, "OFF"},
    {StackState::STBY_CONFIGURED, "STBY_CONFIGURED"},
    {StackState::INTERMEDIATE, "INTERMEDIATE"},
    {StackState::ON, "ON"},
    {StackState::ERROR, "ERROR"},
    {StackState::ERROR_LOCAL, "ERROR_LOCAL"},
    {StackState::SOFT_INTERLOCK, "SOFT_INTERLOCK"},
    {StackState::INTERLOCK, "INTERLOCK"},
    {StackState::RAMPIG_UP_LOW, "RAMPIG_UP_LOW"},
    {StackState::RAMPIG_DOWN_LOW, "RAMPIG_DOWN_LOW"},
    {StackState::RAMPIG_UP, "RAMPIG_UP"},
    {StackState::RAMPIG_DOWN, "RAMPIG_DOWN"},
    {StackState::MIXED, "MIXED"},
    {StackState::NO_CONTROL, "NO_CONTROL"},
};

TimeStampType Gas::getMinTime() const
{
  constexpr auto max = std::numeric_limits<dcs::TimeStampType>::max();
  const std::vector<TimeStampType> times{
    neon.data.size() ? neon.data.front().time : max,
    co2.data.size() ? co2.data.front().time : max,
    n2.data.size() ? n2.data.front().time : max,
    argon.data.size() ? argon.data.front().time : max,
    h2o.data.size() ? h2o.data.front().time : max,
    o2.data.size() ? o2.data.front().time : max,
    h2oSensor.data.size() ? h2oSensor.data.front().time : max,
    o2Sensor.data.size() ? o2Sensor.data.front().time : max,
  };

  return *std::min_element(times.begin(), times.end());
}

TimeStampType Gas::getMaxTime() const
{
  constexpr auto min = 0;
  const std::vector<TimeStampType> times{
    neon.data.size() ? neon.data.back().time : min,
    co2.data.size() ? co2.data.back().time : min,
    n2.data.size() ? n2.data.back().time : min,
    argon.data.size() ? argon.data.back().time : min,
    h2o.data.size() ? h2o.data.back().time : min,
    o2.data.size() ? o2.data.back().time : min,
    h2oSensor.data.size() ? h2oSensor.data.back().time : min,
    o2Sensor.data.size() ? o2Sensor.data.back().time : min,
  };

  return *std::max_element(times.begin(), times.end());
}
