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

#ifndef CPV_PEDESTAL_CALIBRATIOR_H_
#define CPV_PEDESTAL_CALIBRATIOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/Pedestals.h"
#include "CCDB/CcdbObjectInfo.h"
//#include "TString.h"

namespace o2
{
namespace cpv
{
using Digit = o2::cpv::Digit;

//===================================================================
class PedestalSpectrum
{
 public:
  PedestalSpectrum();
  ~PedestalSpectrum() = default;
  PedestalSpectrum& operator+=(const PedestalSpectrum& rhs);
  void fill(uint16_t amplitude);
  uint16_t getNPeaks();
  float getPeakMean(uint16_t iPeak); // return mean value of i-th peak
  float getPeakRMS(uint16_t iPeak);  // return RMS value of i-th peak
  float getPedestalValue();          // return final decision for pedestal value
  float getPedestalRMS();            // return final decision for pedestal RMS
  uint32_t getNEntries() { return mNEntries; }

 private:
  void analyze();
  uint32_t mNEntries = 0;
  uint16_t mNPeaks = 0;
  bool mIsAnalyzed = false;
  uint16_t mToleratedGapWidth = 5;
  float mZSnSigmas = 3.;
  float mSuspiciousPedestalRMS = 20.;
  float mPedestalValue;
  float mPedestalRMS;
  std::vector<float> mMeanOfPeaks, mRMSOfPeaks;
  std::vector<uint32_t> mPeakCounts;
  std::map<uint16_t, uint32_t> mSpectrumContainer;
}; // end PedestalSpectrum

//===================================================================
struct PedestalCalibData {
  int mNEvents = 0;
  std::vector<PedestalSpectrum> mPedestalSpectra;

  PedestalCalibData();
  ~PedestalCalibData() = default;

  void fill(const gsl::span<const o2::cpv::Digit> data);
  void merge(const PedestalCalibData* prev);
  void print();

}; //end PedestalCalibData

using PedestalTimeSlot = o2::calibration::TimeSlot<o2::cpv::PedestalCalibData>;
//===================================================================
class PedestalCalibrator final : public o2::calibration::TimeSlotCalibration<o2::cpv::Digit, o2::cpv::PedestalCalibData>
{
 public:
  PedestalCalibrator();
  ~PedestalCalibrator() final = default;
  std::vector<o2::ccdb::CcdbObjectInfo> getCcdbInfoPedestalsVector() { return mCcdbInfoPedestalsVec; }
  std::vector<o2::cpv::Pedestals> getPedestalsVector() { return mPedestalsVec; }
  std::vector<o2::ccdb::CcdbObjectInfo> getCcdbInfoThresholdsFEEVector() { return mCcdbInfoThresholdsFEEVec; }
  std::vector<std::vector<int>> getThresholdsFEEVector() { return mThresholdsFEEVec; }
  std::vector<o2::ccdb::CcdbObjectInfo> getCcdbInfoDeadChannelsVector() { return mCcdbInfoDeadChannelsVec; }
  std::vector<std::vector<int>> getDeadChannelsVector() { return mDeadChannelsVec; }
  std::vector<o2::ccdb::CcdbObjectInfo> getCcdbInfoHighPedChannelsVector() { return mCcdbInfoHighPedChannelsVec; }
  std::vector<std::vector<int>> getHighPedChannelsVector() { return mHighPedChannelsVec; }
  std::vector<o2::ccdb::CcdbObjectInfo> getCcdbInfoEfficienciesVector() { return mCcdbInfoPedEfficienciesVec; }
  std::vector<std::vector<float>> getEfficienciesVector() { return mPedEfficienciesVec; }

  bool hasEnoughData(const PedestalTimeSlot& slot) const final
  {
    LOG(info) << "hasEnoughData() is being called";
    return slot.getContainer()->mNEvents >= mMinEvents;
  }
  void initOutput() final;
  void finalizeSlot(PedestalTimeSlot& slot) final;
  PedestalTimeSlot& emplaceNewSlot(bool front, uint64_t tstart, uint64_t tend) final;

 private:
  int mMinEvents = 100;
  float mZSnSigmas = 3.;
  std::vector<o2::ccdb::CcdbObjectInfo> mCcdbInfoPedestalsVec;
  std::vector<o2::cpv::Pedestals> mPedestalsVec;
  std::vector<o2::ccdb::CcdbObjectInfo> mCcdbInfoThresholdsFEEVec;
  std::vector<std::vector<int>> mThresholdsFEEVec;
  std::vector<o2::ccdb::CcdbObjectInfo> mCcdbInfoDeadChannelsVec;
  std::vector<std::vector<int>> mDeadChannelsVec;
  std::vector<o2::ccdb::CcdbObjectInfo> mCcdbInfoHighPedChannelsVec;
  std::vector<std::vector<int>> mHighPedChannelsVec;
  std::vector<o2::ccdb::CcdbObjectInfo> mCcdbInfoPedEfficienciesVec;
  std::vector<std::vector<float>> mPedEfficienciesVec;
};
} //end namespace cpv
} //end namespace o2

#endif /* CPV_PEDESTAL_CALIBRATIOR_H_ */