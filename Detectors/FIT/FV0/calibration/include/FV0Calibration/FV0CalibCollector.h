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

#ifndef FV0_CALIB_COLLECTOR_H_
#define FV0_CALIB_COLLECTOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsFV0/FV0CalibrationInfoObject.h"
#include "FV0Base/Constants.h"

#include <array>

namespace o2
{
namespace fv0
{

class FV0CalibInfoSlot
{
  using Slot = o2::calibration::TimeSlot<o2::fv0::FV0CalibInfoSlot>;

 public:
  static constexpr int NCHANNELS = Constants::nFv0Channels;
  static constexpr int HISTO_RANGE = 200;

  FV0CalibInfoSlot()
  {
    for (int ch = 0; ch < NCHANNELS; ch++) {
      mEntriesSlot[ch] = 0;
    }
  }

  ~FV0CalibInfoSlot() = default;

  void print() const;
  void printEntries() const;
  void fill(const gsl::span<const o2::fv0::FV0CalibrationInfoObject> data);
  void merge(const FV0CalibInfoSlot* prev);

  auto& getEntriesPerChannel() const { return mEntriesSlot; }
  auto& getEntriesPerChannel() { return mEntriesSlot; }
  auto& getCollectedCalibInfoSlot() { return mFV0CollectedCalibInfoSlot; }
  auto& getCollectedCalibInfoSlot() const { return mFV0CollectedCalibInfoSlot; }

 private:
  std::array<int, NCHANNELS> mEntriesSlot{};                                 // vector containing number of entries per channel
  std::vector<o2::fv0::FV0CalibrationInfoObject> mFV0CollectedCalibInfoSlot; ///< output FV0 calibration info

  ClassDefNV(FV0CalibInfoSlot, 1);
};

class FV0CalibCollector final : public o2::calibration::TimeSlotCalibration<o2::fv0::FV0CalibrationInfoObject, o2::fv0::FV0CalibInfoSlot>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::fv0::FV0CalibInfoSlot>;
  static constexpr int NCHANNELS = Constants::nFv0Channels;

 public:
  FV0CalibCollector(bool TFsendingPolicy, int maxNumOfHits, bool test = false) : mTFsendingPolicy(TFsendingPolicy), mMaxNumOfHits(maxNumOfHits), mTest(test){};

  ~FV0CalibCollector() final = default;

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  void setIsTest(bool istest) { mTest = istest; }
  auto& getCollectedCalibInfo() const { return mFV0CollectedCalibInfo; }
  auto& getEntriesPerChannel() const { return mEntries; }
  void setIsMaxNumberOfHitsAbsolute(bool absNumber) { mAbsMaxNumOfHits = absNumber; }

 private:
  bool mTFsendingPolicy = false;                                         // whether we will send information at every TF or only when we have a certain statistics
  int mMaxNumOfHits = 1000000;                                           // maximum number of hits for one single channel to trigger the sending of the information (if mTFsendingPolicy = false)
  bool mTest = false;                                                    // flag to say whether we are in test mode or not
  bool mAbsMaxNumOfHits = true;                                          // to decide if the mMaxNumOfHits should be multiplied by the number of FV0 channels
  std::array<int, NCHANNELS> mEntries{};                                 // vector containing number of entries per channel
  std::vector<o2::fv0::FV0CalibrationInfoObject> mFV0CollectedCalibInfo; ///< output FV0 calibration info

  ClassDefOverride(FV0CalibCollector, 1);
};

} // end namespace fv0
} // end namespace o2

#endif /* FV0 CALIB COLLECTOR */
