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
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::tpc::CalibLaserTracks>;

 public:
  LaserTracksCalibrator() = default;
  LaserTracksCalibrator(size_t minTFs) : mMinTFs(minTFs) {}
  ~LaserTracksCalibrator() final = default;

  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->hasEnoughData(mMinTFs); }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  const auto& getCalibPerSlot() { return mCalibPerSlot; }
  bool hasCalibrationData() const { return mCalibPerSlot.size() > 0; }

  void setWriteDebug(bool debug = true) { mWriteDebug = debug; }
  bool getWriteDebug() const { return mWriteDebug; }

  void setMinTFs(size_t minTFs) { mMinTFs = minTFs; }
  size_t getMinTFs() const { return mMinTFs; }

 private:
  size_t mMinTFs = 100;                    ///< laser tracks of these amount of time frames
  std::vector<LtrCalibData> mCalibPerSlot; ///< drift velocity per slot
  bool mWriteDebug = false;                ///< if to save debug trees

  ClassDefOverride(LaserTracksCalibrator, 1);
};
} // namespace o2::tpc
#endif
