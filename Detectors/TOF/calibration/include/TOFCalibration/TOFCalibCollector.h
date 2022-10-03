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

#ifndef TOF_CALIB_COLLECTOR_H_
#define TOF_CALIB_COLLECTOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/CalibInfoCluster.h"
#include "DataFormatsTOF/CalibInfoTOFshort.h"

#include <array>

namespace o2
{
namespace tof
{

class TOFCalibInfoSlot
{

  using Slot = o2::calibration::TimeSlot<o2::tof::TOFCalibInfoSlot>;
  using Geo = o2::tof::Geo;

 public:
  static constexpr int NCHANNELSXSECTOR = o2::tof::Geo::NCHANNELS / o2::tof::Geo::NSECTORS;

  TOFCalibInfoSlot(float phase = 0.0) : mLHCphase(phase)
  {
    for (int ch = 0; ch < Geo::NCHANNELS; ch++) {
      mEntriesSlot[ch] = 0;
    }
  }

  ~TOFCalibInfoSlot() = default;

  void print() const;
  void printEntries() const;
  void fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data);
  void fill(const gsl::span<const o2::tof::CalibInfoCluster> data);
  void merge(const TOFCalibInfoSlot* prev);
  void setLHCphase(float val) { mLHCphase = val; }
  float getLHCphase() const { return mLHCphase; }

  auto& getEntriesPerChannel() const { return mEntriesSlot; }
  auto& getEntriesPerChannel() { return mEntriesSlot; }
  auto& getCollectedCalibInfoSlot() { return mTOFCollectedCalibInfoSlot; }
  auto& getCollectedCalibInfoSlot() const { return mTOFCollectedCalibInfoSlot; }

 private:
  std::array<int, Geo::NCHANNELS> mEntriesSlot;                               // vector containing number of entries per channel
  std::vector<o2::dataformats::CalibInfoTOF> mTOFCollectedCalibInfoSlot;      ///< output TOF calibration info
  float mLHCphase = 0;                                                        // current LHCphase inside the BC (-5000 < phase < 20000)

  ClassDefNV(TOFCalibInfoSlot, 1);
};

class TOFCalibCollector final : public o2::calibration::TimeSlotCalibration<o2::dataformats::CalibInfoTOF, o2::tof::TOFCalibInfoSlot>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::tof::TOFCalibInfoSlot>;

 public:
  TOFCalibCollector(bool TFsendingPolicy, int maxNumOfHits, bool test = false) : mTFsendingPolicy(TFsendingPolicy), mMaxNumOfHits(maxNumOfHits), mTest(test){};

  ~TOFCalibCollector() final = default;

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  void setIsTest(bool istest) { mTest = istest; }
  auto& getCollectedCalibInfo() const { return mTOFCollectedCalibInfo; }
  auto& getEntriesPerChannel() const { return mEntries; }
  void setIsMaxNumberOfHitsAbsolute(bool absNumber) { mAbsMaxNumOfHits = absNumber; }
  void setLHCphase(float val) { mLHCphase = val; }
  float getLHCphase() const { return mLHCphase; }

 private:
  bool mTFsendingPolicy = false;                                          // whether we will send information at every TF or only when we have a certain statistics
  int mMaxNumOfHits = 500;                                                // maximum number of hits for one single channel to trigger the sending of the information (if mTFsendingPolicy = false)
  bool mTest = false;                                                     // flag to say whether we are in test mode or not
  bool mAbsMaxNumOfHits = true;                                           // to decide if the mMaxNumOfHits should be multiplied by the number of TOF channels
  std::array<int, Geo::NCHANNELS> mEntries;                               // vector containing number of entries per channel
  std::vector<o2::dataformats::CalibInfoTOF> mTOFCollectedCalibInfo;      ///< output TOF calibration info
  float mLHCphase = 0;                                                    // current LHCphase inside the BC (-5000 < phase < 20000)

  ClassDefOverride(TOFCalibCollector, 3);
};

} // end namespace tof
} // end namespace o2

#endif /* TOF_CHANNEL_CALIBRATOR_H_ */
