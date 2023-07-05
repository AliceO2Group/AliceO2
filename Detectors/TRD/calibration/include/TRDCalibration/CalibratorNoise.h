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

/// \file CalibratorNoise.h
/// \brief TRD pad calibration

#ifndef O2_TRD_CALIBRATORNOISE_H
#define O2_TRD_CALIBRATORNOISE_H

#include "DataFormatsTRD/NoiseCalibration.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Constants.h"
#include "TRDCalibration/CalibrationParams.h"
#include "CCDB/CcdbObjectInfo.h"
#include "Rtypes.h"

namespace o2
{
namespace trd
{

struct ChannelInfoDetailed {
  ChannelInfoDetailed() = default;
  ChannelInfoDetailed(const ChannelInfoDetailed&) = default;
  ChannelInfoDetailed& operator=(const ChannelInfoDetailed& rhs) = default;

  float getRMS() { return nEntries > 0 ? std::sqrt(variance / nEntries) : -1.f; }

  uint32_t det{0};           ///< detector number
  uint32_t sec{0};           ///< sector
  uint32_t stack{0};         ///< stack
  uint32_t layer{0};         ///< layer
  uint32_t row{0};           ///< pad row
  int col{0};                ///< pad column, not unsigned, since outer shared pads get negative value assigned
  uint32_t channelGlb{0};    ///< global channel number within pad row 0..NCHANNELSPERROW
  uint32_t indexGlb{0};      ///< global channel index 0..NCHANNELSTOTAL
  float adcMean{0};          ///< mean ADC value for this pad
  uint64_t adcSum{0};        ///< sum of ADC_i values
  uint64_t adcSumSquared{0}; ///< sum of ADC_i^2
  uint32_t nEntries{0};      ///< number of ADC values stored
  float variance{0};         ///< the sum of (ADC_i - ADC_mean)^2
  bool isShared{false};      ///< flag, whether this is a shared pad
  ClassDefNV(ChannelInfoDetailed, 1);
};

class CalibratorNoise
{

 public:
  // Default c'tor
  CalibratorNoise() { mChannelInfosDetailed.resize(constants::NCHANNELSTOTAL); };

  // Check if total number of digits seen is higher than threshold
  bool hasEnoughData() const { return (mNDigitsSeen > mParams.minNumberOfDigits); }

  // Add information from digits to mChannelInfosDetailed
  void process(const gsl::span<const Digit>& digits);

  // Getter for internal data when this is run from a macro
  const std::vector<ChannelInfoDetailed>& getInternalChannelInfos() { return mChannelInfosDetailed; }

  // Fill mean and RMS for each channel into mCCDBObject
  void collectChannelInfo();

  // Make CCDB object accessible from outside
  const ChannelInfoContainer& getCcdbObject() const { return mCCDBObject; }

 private:
  ChannelInfoContainer mCCDBObject{};                        ///< array with information for each TRD readout channel (to be put in the CCDB)
  std::vector<ChannelInfoDetailed> mChannelInfosDetailed{};  ///< used to collect detailed information from each channel
  uint64_t mNDigitsSeen{0};                                  ///< counter to decide when to stop processing input data
  const TRDCalibParams& mParams{TRDCalibParams::Instance()}; ///< reference to calibration parameters
  ClassDefNV(CalibratorNoise, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CALIBRATORNOISE_H
