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

/// \file CalibratordEdx.h
/// \brief This file provides the time dependent dE/dx calibrator, tracking the MIP position over time.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBRATORDEDX_H_
#define ALICEO2_TPC_CALIBRATORDEDX_H_

#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "TPCCalibration/CalibdEdx.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DetectorsBase/Propagator.h"

namespace o2::tpc
{

/// dE/dx calibrator class
class CalibratordEdx final : public o2::calibration::TimeSlotCalibration<o2::tpc::CalibdEdx>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<CalibdEdx>;
  using TFinterval = std::vector<std::pair<TFType, TFType>>;
  using TimeInterval = std::vector<std::pair<long, long>>;
  using CalibVector = std::vector<CalibdEdxCorrection>;

 public:
  CalibratordEdx() = default;

  void setHistParams(int dEdxBins, float mindEdx, float maxdEdx, int angularBins, bool fitSnp)
  {
    mdEdxBins = dEdxBins;
    mMindEdx = mindEdx;
    mMaxdEdx = maxdEdx;
    mAngularBins = angularBins;
    mFitSnp = fitSnp;
  }
  void setCuts(const TrackCuts& cuts) { mCuts = cuts; }
  void setMinEntries(int minEntries) { mMinEntries = minEntries; }
  void setFitThresholds(int minEntriesSector, int minEntries1D, int minEntries2D) { mFitThreshold = {minEntriesSector, minEntries1D, minEntriesSector}; }
  void setApplyCuts(bool apply) { mApplyCuts = apply; }
  void setElectronCut(std::tuple<float, int, float> values) { mElectronCut = values; }
  void setMaterialType(o2::base::Propagator::MatCorrType materialType) { mMatType = materialType; }

  /// \brief Check if there are enough data to compute the calibration.
  /// \return false if any of the histograms has less entries than mMinEntries
  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->hasEnoughData(mMinEntries); };

  /// Empty the output vectors
  void initOutput() final;

  /// Process time slot data and compute its calibration
  void finalizeSlot(Slot&) final;

  /// Creates new time slot
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  /// \return the computed calibrations
  const CalibVector& getCalibs() const { return mCalibs; }

  /// \return Time frame ID information
  const TFinterval& getTFinterval() const { return mTFIntervals; }

  /// \return Time frame time information
  const TimeInterval& getTimeIntervals() const { return mTimeIntervals; }

  /// Enable debug output to file of the time slots calibrations outputs and dE/dx histograms
  void enableDebugOutput(std::string_view fileName);

  /// Disable debug output to file. Also writes and closes stored time slots.
  void disableDebugOutput();

  /// \return if debug output is enabled
  bool hasDebugOutput() const { return static_cast<bool>(mDebugOutputStreamer); }

  /// Write debug output to file
  void finalizeDebugOutput() const;

 private:
  int mdEdxBins{};                              ///< Number of dEdx bins
  float mMindEdx{};                             ///< Minimum value for the dEdx histograms
  float mMaxdEdx{};                             ///< Maximum value for the dEdx histograms
  int mAngularBins{};                           ///< Number of bins for angular data, like Tgl and Snp
  bool mFitSnp{};                               ///< enable Snp correction
  int mMinEntries{};                            ///< Minimum amount of tracks in each time slot, to get enough statics
  std::array<int, 3> mFitThreshold{};           ///< Minimum entries per stack to perform sector, 1D and 2D fit
  bool mApplyCuts{true};                        ///< Flag to enable tracks cuts
  std::tuple<float, int, float> mElectronCut{}; ///< Values passed to CalibdEdx::setElectronCut
  TrackCuts mCuts;                              ///< Cut object
  o2::base::Propagator::MatCorrType mMatType{}; ///< material type for track propagation

  TFinterval mTFIntervals;     ///< start and end time frame IDs of each calibration time slots
  TimeInterval mTimeIntervals; ///< start and end times of each calibration time slots
  CalibVector mCalibs;         ///< vector of MIP positions, each element is filled in "process" when we finalize one slot (multiple can be finalized during the same "process", which is why we have a vector. Each element is to be considered the output of the device

  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugOutputStreamer; ///< Debug output streamer

  ClassDefOverride(CalibratordEdx, 2);
};

} // namespace o2::tpc
#endif
