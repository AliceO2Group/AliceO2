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

#ifndef CPV_NOISE_CALIBRATOR_H_
#define CPV_NOISE_CALIBRATOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/BadChannelMap.h"
#include "CCDB/CcdbObjectInfo.h"

namespace o2
{
namespace cpv
{
//=========================================================================
using Digit = o2::cpv::Digit;
struct NoiseCalibData {
  uint32_t mNEvents = 0;
  float mNoiseThreshold = 10.; // ADC counts threshold
  std::vector<int> mOccupancyMap;

  NoiseCalibData(float noiseThreshold = 10.);
  ~NoiseCalibData() = default;

  void fill(const gsl::span<const o2::cpv::Digit> data);
  void merge(const NoiseCalibData* prev);
  void print();

}; // end NoiseCalibData
//=========================================================================

using NoiseTimeSlot = o2::calibration::TimeSlot<o2::cpv::NoiseCalibData>;
class NoiseCalibrator final : public o2::calibration::TimeSlotCalibration<o2::cpv::NoiseCalibData>
{
 public:
  NoiseCalibrator();
  ~NoiseCalibrator() final = default;
  std::vector<o2::ccdb::CcdbObjectInfo>& getCcdbInfoBadChannelMapVector() { return mCcdbInfoBadChannelMapVec; }
  const std::vector<o2::cpv::BadChannelMap>& getBadChannelMapVector() const { return mBadChannelMapVec; }

  void setPedEfficiencies(std::vector<float>* pedEffs) { mPedEfficiencies = pedEffs; }
  void setDeadChannels(std::vector<int>* deadChs) { mDeadChannels = deadChs; }
  void setHighPedChannels(std::vector<int>* highPeds) { mHighPedChannels = highPeds; }
  void setPersistentBadChannels(std::vector<int>* persBadChs) { mPersistentBadChannels = persBadChs; }

  bool isSettedPedEfficiencies() { return mPedEfficiencies == nullptr ? false : true; }
  bool isSettedDeadChannels() { return mDeadChannels == nullptr ? false : true; }
  bool isSettedHighPedChannels() { return mHighPedChannels == nullptr ? false : true; }
  bool isSettedPersistentBadChannels() { return mPersistentBadChannels == nullptr ? false : true; }

  bool hasEnoughData(const NoiseTimeSlot& slot) const final
  {
    LOG(info) << "hasEnoughData() is being called";
    return slot.getContainer()->mNEvents >= mMinEvents;
  }
  void initOutput() final;
  void finalizeSlot(NoiseTimeSlot& slot) final;
  NoiseTimeSlot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  void configParameters();

 private:
  std::vector<float>* mPedEfficiencies = nullptr;
  std::vector<int>* mDeadChannels = nullptr;
  std::vector<int>* mHighPedChannels = nullptr;
  std::vector<int>* mPersistentBadChannels = nullptr;
  uint32_t mMinEvents = 100;
  float mNoiseFrequencyCriteria = 0.5; // how often channel should appear to be considered as noisy
  float mToleratedChannelEfficiencyLow = 0.9;
  float mToleratedChannelEfficiencyHigh = 1.01;
  float mNoiseThreshold = 10.;
  std::vector<o2::ccdb::CcdbObjectInfo> mCcdbInfoBadChannelMapVec;
  std::vector<o2::cpv::BadChannelMap> mBadChannelMapVec;

}; // end NoiseCalibrator

} // namespace cpv
} // namespace o2

#endif /* CPV_NOISE_CALIBRATOR_H_ */
