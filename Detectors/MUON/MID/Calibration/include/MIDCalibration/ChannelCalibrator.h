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

/// \file   MIDCalibration/ChannelCalibrator.h
/// \brief  MID noise and dead channels calibrator
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 February 2022

#ifndef O2_MID_CHANNELCALIBRATOR_H
#define O2_MID_CHANNELCALIBRATOR_H

#include <string>
#include <vector>
#include <gsl/span>
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"

#include "MIDFiltering/ChannelScalers.h"

namespace o2
{
namespace mid
{

class NoiseData
{
  using Slot = o2::calibration::TimeSlot<NoiseData>;

 public:
  /// Fills the data
  /// \param data Noisy digits
  void fill(const gsl::span<const ColumnData> data);

  /// Merges data
  /// \param prev Previous container
  void merge(const NoiseData* prev);

  /// Prints scalers
  void print();

  /// Gets the channel scalers
  const ChannelScalers& getScalers() { return mChannelScalers; }

 private:
  ChannelScalers mChannelScalers;

  ClassDefNV(NoiseData, 1);
};

class ChannelCalibrator final : public o2::calibration::TimeSlotCalibration<ColumnData, NoiseData>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<NoiseData>;

 public:
  /// Initialize the output
  void initOutput() final;

  /// Checks if there is enough data
  /// \param slot TimeSlot container
  /// \return true if there is enough data
  bool hasEnoughData(const Slot& slot) const final;

  /// Finalizes the slot
  /// \param slot TimeSlot container
  /// \return true if there is enough data
  void finalizeSlot(Slot& slot) final;

  /// Creates a new time slot
  /// \param front Emplaces at front if true
  /// \param tstart Start time
  /// \param tend End time
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  /// Add number of calibration triggers processed
  /// \param nEvents Number of calibration triggers in this TF
  void addEvents(unsigned long int nEvents) { mEventsCounter += nEvents; }

  /// Returns the bad channels
  const std::vector<ColumnData>& getBadChannels() const { return mBadChannels; }

  /// Returns mask as string
  std::string getMasksAsString() { return mMasksString; }

  /// Sets reference masks
  void setReferenceMasks(const std::vector<ColumnData>& refMasks) { mRefMasks = refMasks; }

 private:
  std::vector<ColumnData> mBadChannels; /// List of bad channels
  std::vector<ColumnData> mRefMasks;    /// Reference masks
  std::string mMasksString;             /// Masks as string
  double mThreshold = 0.9;              /// Noise threshold
  unsigned long int mEventsCounter = 0; /// Events counter

  ClassDefOverride(ChannelCalibrator, 1);
};
} // namespace mid
} // namespace o2

#endif
