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

#ifndef FT0_CALIB_COLLECTOR_H_
#define FT0_CALIB_COLLECTOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsFT0/FT0CalibrationInfoObject.h"
#include "FT0Base/Geometry.h"

#include <array>

namespace o2
{
namespace ft0
{

class FT0CalibInfoSlot
{

  using Slot = o2::calibration::TimeSlot<o2::ft0::FT0CalibInfoSlot>;
  using Geo = o2::ft0::Geometry;

 public:
  static constexpr int NCHANNELS = Geo::Nchannels;
  static constexpr int HISTO_RANGE = 200;

  FT0CalibInfoSlot()
  {
    for (int ch = 0; ch < NCHANNELS; ch++) {
      mEntriesSlot[ch] = 0;
    }
  }

  ~FT0CalibInfoSlot() = default;

  void print() const;
  void printEntries() const;
  void fill(const gsl::span<const o2::ft0::FT0CalibrationInfoObject> data);
  void merge(const FT0CalibInfoSlot* prev);

  auto& getEntriesPerChannel() const { return mEntriesSlot; }
  auto& getEntriesPerChannel() { return mEntriesSlot; }
  auto& getCollectedCalibInfoSlot() { return mFT0CollectedCalibInfoSlot; }
  auto& getCollectedCalibInfoSlot() const { return mFT0CollectedCalibInfoSlot; }

 private:
  std::array<int, NCHANNELS> mEntriesSlot;                                   // vector containing number of entries per channel
  std::vector<o2::ft0::FT0CalibrationInfoObject> mFT0CollectedCalibInfoSlot; ///< output FT0 calibration info

  ClassDefNV(FT0CalibInfoSlot, 1);
};

class FT0CalibCollector final : public o2::calibration::TimeSlotCalibration<o2::ft0::FT0CalibrationInfoObject, o2::ft0::FT0CalibInfoSlot>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::ft0::FT0CalibInfoSlot>;
  static constexpr int NCHANNELS = o2::ft0::Geometry::Nchannels;

 public:
  FT0CalibCollector(bool TFsendingPolicy, int maxNumOfHits) : mTFsendingPolicy(TFsendingPolicy), mMaxNumOfHits(maxNumOfHits){};

  ~FT0CalibCollector() final = default;

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  auto& getCollectedCalibInfo() const { return mFT0CollectedCalibInfo; }
  auto& getEntriesPerChannel() const { return mEntries; }
  void setIsMaxNumberOfHitsAbsolute(bool absNumber) { mAbsMaxNumOfHits = absNumber; }

 private:
  bool mTFsendingPolicy = false;                                         // whether we will send information at every TF or only when we have a certain statistics
  int mMaxNumOfHits = 1000000;                                           // maximum number of hits for one single channel to trigger the sending of the information (if mTFsendingPolicy = false)
  bool mAbsMaxNumOfHits = true;                                          // to decide if the mMaxNumOfHits should be multiplied by the number of FT0 channels
  std::array<int, NCHANNELS> mEntries;                                   // vector containing number of entries per channel
  std::vector<o2::ft0::FT0CalibrationInfoObject> mFT0CollectedCalibInfo; ///< output FT0 calibration info

  ClassDefOverride(FT0CalibCollector, 1);
};

} // end namespace ft0
} // end namespace o2

#endif /* FT0 CALIB COLLECTOR */
