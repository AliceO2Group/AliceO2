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
  using TimeInterval = std::vector<std::pair<long, long>>;

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

  /// set how the extracted gain map is normalized
  void setNormalizationType(const CalibPadGainTracksBase::NormType type) { mNormType = type; }

  /// \return returns minimum number of entries per pad-by-pad histogram
  size_t getMinEntries() const { return mMinEntries; }

  /// \param low lower truncation range for calculating the rel gain
  /// \param high upper truncation range
  void setTruncationRange(const float low = 0.05f, const float high = 0.6f);

  /// \param minRelgain minimum accpeted relative gain (if the gain is below this value it will be set to minRelgain)
  /// \param maxRelgain maximum accpeted relative gain (if the gain is above this value it will be set to maxRelgain)
  void setRelGainRange(const float minRelgain = 0.1f, const float maxRelgain = 2.f);

  /// minEntries minimum number of entries in pad-by-pad histogram to calculate the mean
  void setMinEntriesMean(const int minEntriesMean) { mMinEntriesMean = minEntriesMean; }

  /// \param writeDebug writting debug output
  void setWriteDebug(const bool writeDebug) { mWriteDebug = writeDebug; }

  /// \param useLastMap buffer last extracted gain map
  void setUseLastExtractedMapAsReference(const bool useLastMap) { mUseLastExtractedMapAsReference = useLastMap; }

  /// \return returns if debug fileswill be written
  bool getWriteDebug() const { return mWriteDebug; }

  /// \return returns lower truncation range
  float getTruncationRangeLow() const { return mLowTruncation; }

  /// \return returns upper truncation range
  float getTruncationRangeUp() const { return mUpTruncation; }

  /// \return CCDB output informations
  const TFinterval& getTFinterval() const { return mIntervals; }

  /// \return Time frame time information
  const TimeInterval& getTimeIntervals() const { return mTimeIntervals; }

  /// \return returns calibration objects (pad-by-pad gain maps)
  auto getCalibs() && { return std::move(mCalibs); }

  /// check if calibration data is available
  bool hasCalibrationData() const { return mCalibs.size() > 0; }

 private:
  TFinterval mIntervals;                                                      ///< start and end time frames of each calibration time slots
  TimeInterval mTimeIntervals;                                                ///< start and end times of each calibration time slots
  CalibVector mCalibs;                                                        ///< Calibration object containing for each pad a histogram with normalized charge
  float mLowTruncation{0.05f};                                                ///< lower truncation range for calculating mean of the histograms
  float mUpTruncation{0.6f};                                                  ///< upper truncation range for calculating mean of the histograms
  float mMinRelgain{0.1f};                                                    ///< minimum accpeted relative gain (if the gain is below this value it will be set to 1)
  float mMaxRelgain{2.f};                                                     ///< maximum accpeted relative gain (if the gain is above this value it will be set to 1)
  int mMinEntriesMean{10};                                                    ///< minEntries minimum number of entries in pad-by-pad histogram to calculate the mean
  size_t mMinEntries{30};                                                     ///< Minimum amount of clusters per pad in each time slot, to get enough statics
  bool mWriteDebug = false;                                                   ///< if to save debug trees
  CalibPadGainTracksBase::NormType mNormType{CalibPadGainTracksBase::region}; ///< Normalization type for the extracted gain map
  bool mUseLastExtractedMapAsReference{false};                                ///< Multiply the current extracted gain map with the last extracted gain map
  std::unique_ptr<CalPad> mGainMapLastIteration;                              ///< gain map extracted from particle tracks from the last iteration

  ClassDefOverride(CalibratorPadGainTracks, 1);
};
} // namespace o2::tpc

#endif
