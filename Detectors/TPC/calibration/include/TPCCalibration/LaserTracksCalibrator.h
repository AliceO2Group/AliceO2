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

/// \file LaserTracksCalibrator.h
/// \brief time slot calibration using laser tracks
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef TPC_LaserTracksCalibrator_H_
#define TPC_LaserTracksCalibrator_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"

#include "TPCCalibration/CalibLaserTracks.h"

namespace o2::tpc
{

class LaserTracksCalibrator : public o2::calibration::TimeSlotCalibration<TrackTPC, CalibLaserTracks>
{
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<o2::tpc::CalibLaserTracks>;

 public:
  LaserTracksCalibrator() = default;
  LaserTracksCalibrator(size_t minEntries) : mMinEntries(minEntries) {}
  ~LaserTracksCalibrator() final = default;

  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->getMatchedPairs() >= mMinEntries; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  const auto& getDVperSlot() { return mDVperSlot; }

 private:
  size_t mMinEntries = 100;
  std::vector<TimePair> mDVperSlot; ///< drift velocity per slot

  ClassDefOverride(LaserTracksCalibrator, 1);
};
} // namespace o2::tpc
#endif
