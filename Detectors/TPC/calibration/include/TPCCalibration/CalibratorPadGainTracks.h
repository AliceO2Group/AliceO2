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

///
/// @file   CalibratorPadGainTracks.h
/// @author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de
///

#ifndef ALICEO2_TPC_CALIBRATORPADGAINTRACKS_H
#define ALICEO2_TPC_CALIBRATORPADGAINTRACKS_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "TPCCalibration/CalibPadGainTracksBase.h"

namespace o2::tpc
{
/// \brief calibrator class for the residual gain map extraction used on an aggregator node
class CalibratorPadGainTracks : public o2::calibration::TimeSlotCalibration<CalibPadGainTracksBase::DataTHistos, CalibPadGainTracksBase>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<CalibPadGainTracksBase>;
  using TFinterval = std::vector<std::pair<TFType, TFType>>;
  using CalibVector = std::vector<std::unordered_map<std::string, CalPad>>; // extracted gain map

 public:
  /// construcor
  CalibratorPadGainTracks() = default;

  /// destructor
  ~CalibratorPadGainTracks() final = default;

  /// clearing the output
  void initOutput() final;

  /// \brief Check if there are enough data to compute the calibration.
  /// \return false if any of the histograms has less entries than mMinEntries
  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->hasEnoughData(mMinEntries); };

  /// process time slot (create pad-by-pad gain map from tracks)
  void finalizeSlot(Slot& slot) final;

  /// Creates new time slot
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  /// \param minEntries minimum number of entries per pad-by-pad histogram
  void setMinEntries(const size_t minEntries) { mMinEntries = minEntries; }

  /// \return returns minimum number of entries per pad-by-pad histogram
  size_t getMinEntries() const { return mMinEntries; }

  /// \param low lower truncation range for calculating the rel gain
  /// \param high upper truncation range
  void setTruncationRange(const float low = 0.05f, const float high = 0.6f);

  /// \param writeDebug writting debug output
  void setWriteDebug(const bool writeDebug) { mWriteDebug = writeDebug; }

  /// \return returns if debug fileswill be written
  bool getWriteDebug() const { return mWriteDebug; }

  /// \return returns lower truncation range
  float getTruncationRangeLow() const { return mLowTruncation; }

  /// \return returns upper truncation range
  float getTruncationRangeUp() const { return mUpTruncation; }

  /// \return CCDB output informations
  const TFinterval& getTFinterval() const { return mIntervals; }

  /// \return returns calibration objects (pad-by-pad gain maps)
  const auto& getCalibs() { return mCalibs; }

  /// check if calibration data is available
  bool hasCalibrationData() const { return mCalibs.size() > 0; }

 private:
  TFinterval mIntervals;       ///< start and end time frames of each calibration time slots
  CalibVector mCalibs;         ///< Calibration object containing for each pad a histogram with normalized charge
  float mLowTruncation{0.05f}; ///< lower truncation range for calculating mean of the histograms
  float mUpTruncation{0.6f};   ///< upper truncation range for calculating mean of the histograms
  size_t mMinEntries{30};      ///< Minimum amount of tracks in each time slot, to get enough statics
  bool mWriteDebug = false;    ///< if to save debug trees

  ClassDefOverride(CalibratorPadGainTracks, 1);
};
} // namespace o2::tpc

#endif
