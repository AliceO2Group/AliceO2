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
  void fillPressure(const DPCOM& dp);
  void finalizeSlot();
  void finalize();

  void finalizeTemperature();
  void finalizeHighVoltage();
  void finalizeGas();
  void finalizePressure();

  /// name of the debug output tree
  void setDebugOutputName(std::string_view name) { mDebugOutputName = name; }

  /// if to write debug information
  void setWriteDebug(const bool debug = true) { mWriteDebug = debug; }

  /// write the debug output tree
  void writeDebug();

  /// set the fit interval
  void setFitInterval(dcs::TimeStampType interval) { mFitInterval = interval; }

  /// set the interval for averaging the pressure values
  void setPressureInterval(dcs::TimeStampType interval) { mPressureInterval = interval; }

  void setRefPressureInterval(dcs::TimeStampType interval) { mPressureIntervalRef = interval; }

  /// get fit interval
  auto getFitInterval() const { return mFitInterval; }

  /// get fit interval
  auto getPressureInterval() const { return mPressureInterval; }

  /// round to fit interval
  void setRoundToInterval(const bool round = true) { mRoundToInterval = round; }

  /// reset all data
  void reset()
  {
    mTemperature.clear();
    mHighVoltage.clear();
    mGas.clear();
    mPressure.clear();

    mTimeTemperature = {};
    mTimeHighVoltage = {};
    mTimeGas = {};
  }

  /// if data to process
  bool hasData() const { return mHasData; }

  const auto& getTimeTemperature() const { return mTimeTemperature; }
  const auto& getTimeHighVoltage() const { return mTimeHighVoltage; }
  const auto& getTimeGas() const { return mTimeGas; }
  const auto& getTimePressure() const { return mTimePressure; }

  auto& getTemperature() { return mTemperature; }
  auto& getHighVoltage() { return mHighVoltage; }
  auto& getGas() { return mGas; }
  auto& getPressure() { return mPressure; }

 private:
  dcs::Temperature mTemperature; ///< temperature value store
  dcs::HV mHighVoltage;          ///< HV value store
  dcs::Gas mGas;                 ///< Gas value store
  dcs::Pressure mPressure;       ///< Pressure value

  TimeRange mTimeTemperature; ///< Time range for temperature values
  TimeRange mTimeHighVoltage; ///< Time range for high voltage values
  TimeRange mTimeGas;         ///< Time range for gas values
  TimeRange mTimePressure;    ///< Time range for pressure values

  dcs::TimeStampType mFitInterval{5 * 60 * 1000};                ///< fit interval (ms) e.g. for temparature data
  dcs::TimeStampType mPressureInterval{200 * 1000};              ///< interval (ms) for averaging pressure values
  dcs::TimeStampType mPressureIntervalRef{60 * 60 * 1000};       ///< interval (ms) for averaging pressure values for longer reference time interval
  bool mWriteDebug{false};                                       ///< switch to dump debug tree
  bool mRoundToInterval{false};                                  ///< round to full fit interval e.g. full minute
  bool mHasData{false};                                          ///< if there are data to process
  std::string mDebugOutputName{"DCS_debug.root"};                ///< name of the debug output tree
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream; //!< debug output streamer

  ClassDefNV(DCSProcessor, 0);
};

} // namespace o2::tpc
#endif
