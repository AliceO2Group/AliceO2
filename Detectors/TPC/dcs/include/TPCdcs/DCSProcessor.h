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

/// @file DCSProcessor.h
/// @brief TPC DCS data point processor
/// @author Jens Wiechula

#ifndef O2_TPC_DCSProcessor_H_
#define O2_TPC_DCSProcessor_H_

#include <memory>
#include <gsl/span>
#include <algorithm>
#include <limits>

#include "Rtypes.h"

#include "DetectorsDCS/DataPointCompositeObject.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsTPC/DCS.h"

using DPCOM = o2::dcs::DataPointCompositeObject;

namespace o2::tpc
{
class DCSProcessor
{
 public:
  struct TimeRange {
    uint64_t first{};
    uint64_t last{};
  };

  void process(const gsl::span<const DPCOM> dps);

  float getValueF(const DPCOM& dp) const;

  void fillTemperature(const DPCOM& dp);
  void fillHV(const DPCOM& dp);
  void fillGas(const DPCOM& dp);
  void finalizeSlot();
  void finalize();

  void finalizeTemperature();
  void finalizeHighVoltage();
  void finalizeGas();

  void fitTemperature(Side side);

  /// get minimum time over all sensors. Assumes data is sorted in time
  template <typename T>
  dcs::TimeStampType getMinTime(const std::vector<dcs::DataPointVector<T>>& data);

  /// get maximum time over all sensors. Assumes data is sorted in time
  template <typename T>
  dcs::TimeStampType getMaxTime(const std::vector<dcs::DataPointVector<T>>& data);

  /// name of the debug output tree
  void setDebugOutputName(std::string_view name) { mDebugOutputName = name; }

  /// if to write debug information
  void setWriteDebug(const bool debug = true) { mWriteDebug = debug; }

  /// write the debug output tree
  void writeDebug();

  /// set the fit interval
  void setFitInterval(dcs::TimeStampType interval) { mFitInterval = interval; }

  /// get fit interval
  auto getFitInterval() const { return mFitInterval; }

  /// round to fit interval
  void setRoundToInterval(const bool round = true) { mRoundToInterval = round; }

  /// reset all data
  void reset()
  {
    mTemperature.clear();
    mHighVoltage.clear();
    mGas.clear();

    mTimeTemperature = {};
    mTimeHighVoltage = {};
    mTimeGas = {};
  }

  /// if data to process
  bool hasData() const { return mHasData; }

  const auto& getTimeTemperature() const { return mTimeTemperature; }
  const auto& getTimeHighVoltage() const { return mTimeHighVoltage; }
  const auto& getTimeGas() const { return mTimeGas; }

  auto& getTemperature() { return mTemperature; }
  auto& getHighVoltage() { return mHighVoltage; }
  auto& getGas() { return mGas; }

 private:
  dcs::Temperature mTemperature; ///< temperature value store
  dcs::HV mHighVoltage;          ///< HV value store
  dcs::Gas mGas;                 ///< Gas value store

  TimeRange mTimeTemperature; ///< Time range for temperature values
  TimeRange mTimeHighVoltage; ///< Time range for high voltage values
  TimeRange mTimeGas;         ///< Time range for gas values

  dcs::TimeStampType mFitInterval{5 * 60 * 1000};                ///< fit interval (ms) e.g. for temparature data
  bool mWriteDebug{false};                                       ///< switch to dump debug tree
  bool mRoundToInterval{false};                                  ///< round to full fit interval e.g. full minute
  bool mHasData{false};                                          ///< if there are data to process
  std::string mDebugOutputName{"DCS_debug.root"};                ///< name of the debug output tree
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream; //!< debug output streamer

  ClassDefNV(DCSProcessor, 0);
};

template <typename T>
dcs::TimeStampType DCSProcessor::getMinTime(const std::vector<dcs::DataPointVector<T>>& data)
{
  constexpr auto max = std::numeric_limits<dcs::TimeStampType>::max();
  dcs::TimeStampType firstTime = std::numeric_limits<dcs::TimeStampType>::max();
  for (const auto& sensor : data) {
    const auto time = sensor.data.size() ? sensor.data.front().time : max;
    firstTime = std::min(firstTime, time);
  }

  // mFitInterval is is seconds. Round to full amount.
  // if e.g. mFitInterval = 5min, then round 10:07:20.510 to 10:05:00.000
  if (mRoundToInterval) {
    firstTime -= (firstTime % mFitInterval);
  }

  return firstTime;
}

template <typename T>
dcs::TimeStampType DCSProcessor::getMaxTime(const std::vector<dcs::DataPointVector<T>>& data)
{
  constexpr auto min = 0;
  dcs::TimeStampType lastTime = 0;
  for (const auto& sensor : data) {
    const auto time = sensor.data.size() ? sensor.data.back().time : 0;
    lastTime = std::max(lastTime, time);
  }

  // mFitInterval is is seconds. Round to full amount.
  // if e.g. mFitInterval = 5min, then round 10:07:20.510 to 10:05:00.000
  // TODO: fix this
  // if (mRoundToInterval) {
  // lastTime -= (lastTime % mFitInterval);
  //}

  return lastTime;
}

} // namespace o2::tpc
#endif
