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

#ifndef CPV_GAIN_CALIBRATOR_H_
#define CPV_GAIN_CALIBRATOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/CalibParams.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CPVBase/Geometry.h"
#include <TH1F.h>

namespace o2
{
namespace cpv
{
//=============================================================================
class AmplitudeSpectrum
{
 public:
  static constexpr uint16_t nBins = 1000; // N bins of amplitude histo
  static constexpr float lRange = 0.;     // Range (left) of amplitude histo
  static constexpr float rRange = 1000.;  // Range (right) of amplitude histo

  AmplitudeSpectrum();
  ~AmplitudeSpectrum() = default;
  AmplitudeSpectrum(const AmplitudeSpectrum&) = default;
  void reset();
  AmplitudeSpectrum& operator+=(const AmplitudeSpectrum& rhs);
  void fill(float amplitude);
  double getMean() const { return mSumA / mNEntries; }                                                  // return mean of distribution
  double getRMS() const { return (mSumA2 / mNEntries) - ((mSumA * mSumA) / (mNEntries * mNEntries)); }; // return RMS of distribution
  uint32_t getNEntries() const { return mNEntries; }
  const uint32_t* getBinContent() { return mBinContent.data(); } // since C++17 std::array::data is constexpr
  void fillBinData(TH1F* h);

 private:
  uint32_t mNEntries;
  double mSumA;
  double mSumA2;
  std::array<uint32_t, nBins> mBinContent; // we are going use ROOT::Fit which works with doubles

  ClassDef(AmplitudeSpectrum, 1);
}; // end AmplitudeSpectrum
//=============================================================================
class GainCalibData
{
 public:
  GainCalibData() = default;
  ~GainCalibData() = default;

  void fill(const gsl::span<const Digit> data);
  void merge(const GainCalibData* prev);
  void print() const;

  std::array<AmplitudeSpectrum, Geometry::kNCHANNELS> mAmplitudeSpectra;

  ClassDef(GainCalibData, 1);
}; // end GainCalibData
//=============================================================================
using GainTimeSlot = o2::calibration::TimeSlot<GainCalibData>;
class GainCalibrator final : public o2::calibration::TimeSlotCalibration<Digit, GainCalibData>
{
 public:
  GainCalibrator();
  ~GainCalibrator() final = default;
  std::vector<o2::ccdb::CcdbObjectInfo>& getCcdbInfoGainsVector() { return mCcdbInfoGainsVec; }
  const std::vector<CalibParams>& getGainsVector() const { return mGainsVec; }
  std::vector<o2::ccdb::CcdbObjectInfo>& getCcdbInfoGainCalibDataVector() { return mCcdbInfoGainCalibDataVec; }
  const std::vector<GainCalibData>& getGainCalibDataVector() const { return mGainCalibDataVec; }

  void setPreviousGains(CalibParams* previousGains) { mPreviousGains.reset(previousGains); }
  bool isSettedPreviousGains() { return mPreviousGains.get() == nullptr ? false : true; }
  void setPreviousGainCalibData(GainCalibData* previousGainCalibData) { mPreviousGainCalibData.reset(previousGainCalibData); }
  bool isSettedPreviousGainCalibData() { return mPreviousGainCalibData.get() == nullptr ? false : true; }

  bool hasEnoughData(const GainTimeSlot& slot) const final;
  void initOutput() final;
  void finalizeSlot(GainTimeSlot& slot) final;
  GainTimeSlot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  void prepareForEnding();
  void setUpdateTFInterval(uint32_t interval) { mUpdateTFInterval = interval; }
  // read configurable parameters from CPVCalibParams
  void configParameters();

 private:
  uint32_t mMinEvents = 1000;                ///< Minimal number of events to produce calibration
  uint32_t mMinNChannelsToCalibrate = 10000; ///< Minimal number of channels per one calibration
  float mDesiredLandauMPV = 200.;            ///< Desired LandauMPV of the spectrum: gain coeff = mDesiredLandauMPV/(max Ampl of the cluster)
  float mToleratedChi2PerNDF = 100.;         ///< Tolerated max Chi2 of the fit
  float mMinAllowedCoeff = 0.1;              ///< Min value of gain coeff at which DesiredLandauMPV is achived
  float mMaxAllowedCoeff = 10.;              ///< Max value of gain coeff at which DesiredLandauMPV is achived
  float mFitRangeL = 10.;                    ///< Fit range of amplitude spectrum (left)
  float mFitRangeR = 1000.;                  ///< Fit range of amplitude spectrum (right)
  uint32_t mUpdateTFInterval = 100;          ///< Update interval (in TF)

  std::unique_ptr<CalibParams> mPreviousGains = nullptr; ///< previous calibration read from CCDB
  std::unique_ptr<GainCalibData> mPreviousGainCalibData; ///< previous GainCalibData read from CCDB
  std::vector<o2::ccdb::CcdbObjectInfo> mCcdbInfoGainsVec;
  std::vector<o2::ccdb::CcdbObjectInfo> mCcdbInfoGainCalibDataVec;
  std::vector<CalibParams> mGainsVec;
  std::vector<GainCalibData> mGainCalibDataVec;

  ClassDef(GainCalibrator, 1);
}; // end GainCalibrator

} // namespace cpv
} // namespace o2

#endif /* CPV_GAIN_CALIBRATOR_H_ */
