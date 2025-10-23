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

/// \file PressureTemperatureHelper.h
/// \brief Helper class to extract pressure and temperature
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef PRESSURETEMPERATUREHELPER_H_
#define PRESSURETEMPERATUREHELPER_H_

#include "GPUCommonRtypes.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/Pair.h"

namespace o2::framework
{
class ProcessingContext;
class ConcreteDataMatcher;
class InputSpec;
class OutputSpec;
} // namespace o2::framework

namespace o2::tpc
{

class PressureTemperatureHelper
{
 public:
  PressureTemperatureHelper() = default;

  /// check for new CCDB objects
  bool accountCCDBInputs(const o2::framework::ConcreteDataMatcher& matcher, void* obj);

  /// trigger checking for CCDB objects
  void extractCCDBInputs(o2::framework::ProcessingContext& pc) const;

  // add required inputs
  static void requestCCDBInputs(std::vector<o2::framework::InputSpec>& inputs);

  /// define outputs in case pressure and temperature will be send
  static void setOutputs(std::vector<o2::framework::OutputSpec>& outputs);

  /// send temperature and pressure for given time stamp
  void sendPTForTS(o2::framework::ProcessingContext& pc, const ULong64_t timestamp) const;

  /// set fit interval range for temperature in ms
  void setFitIntervalTemp(const int fitIntervalMS) { mFitIntervalMS = fitIntervalMS; }

  /// \brief interpolate input values for given timestamp
  /// \param timestamps time stamps of the data
  /// \param values data points
  /// \param timestamp time where to interpolate the values
  float interpolate(const std::vector<ULong64_t>& timestamps, const std::vector<float>& values, ULong64_t timestamp) const;

  /// get pressure for given time stamp in ms
  float getPressure(const ULong64_t timestamp) const { return interpolate(mPressure.second, mPressure.first, timestamp); }

  /// manually set the pressure
  void setPressure(const std::pair<std::vector<float>, std::vector<ULong64_t>>& pressure) { mPressure = pressure; }

  /// manually set the temperature
  void setTemperature(const std::pair<std::vector<float>, std::vector<ULong64_t>>& temperatureA, const std::pair<std::vector<float>, std::vector<ULong64_t>>& temperatureC)
  {
    mTemperatureA = temperatureA;
    mTemperatureC = temperatureC;
  }

  /// get temperature for given time stamp in ms
  dataformats::Pair<float, float> getTemperature(const ULong64_t timestamp) const { return dataformats::Pair<float, float>{interpolate(mTemperatureA.second, mTemperatureA.first, timestamp), interpolate(mTemperatureC.second, mTemperatureC.first, timestamp)}; }

  /// get mean temperature over A and C side
  float getMeanTemperature(const ULong64_t timestamp) const;

  // get ratio of temperature over pressure for given time stamp
  float getTP(int64_t ts) const;

  static constexpr o2::header::DataDescription getDataDescriptionPressure() { return o2::header::DataDescription{"pressure"}; }
  static constexpr o2::header::DataDescription getDataDescriptionTemperature() { return o2::header::DataDescription{"temperature"}; }

  /// get minimum and maximum time stamps of the pressure and temperature data
  std::pair<ULong64_t, ULong64_t> getMinMaxTime() const;

 protected:
  static void addInput(std::vector<o2::framework::InputSpec>& inputs, o2::framework::InputSpec&& isp);
  static void addOutput(std::vector<o2::framework::OutputSpec>& outputs, o2::framework::OutputSpec&& osp);
  static constexpr float toKelvin(float celsius) { return celsius + 273.15f; } // convert Celsius to Kelvin

  std::pair<std::vector<float>, std::vector<ULong64_t>> mPressure;     ///< pressure values for both measurements
  std::pair<std::vector<float>, std::vector<ULong64_t>> mTemperatureA; ///< temperature values A-side
  std::pair<std::vector<float>, std::vector<ULong64_t>> mTemperatureC; ///< temperature values C-side
  int mFitIntervalMS{5 * 60 * 1000};                                   ///< fit interval for the temperature

  ClassDefNV(PressureTemperatureHelper, 1);
};
} // namespace o2::tpc
#endif
