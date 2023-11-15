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

/// \file CalibratorNoise.cxx
/// \brief TRD pad calibration

#include "TRDCalibration/CalibratorNoise.h"

using namespace o2::trd::constants;

namespace o2::trd
{

void CalibratorNoise::process(const gsl::span<const Digit>& digits)
{
  for (const auto& digit : digits) {
    int indexGlobal = HelperMethods::getGlobalChannelIndex(digit.getDetector(), digit.getROB(), digit.getMCM(), digit.getChannel());
    auto& info = mChannelInfosDetailed[indexGlobal];
    if (info.nEntries == 0) {
      // the first time we see data from this channel we fill the detector information etc.
      info.det = digit.getDetector();
      info.sec = HelperMethods::getSector(digit.getDetector());
      info.stack = HelperMethods::getStack(digit.getDetector());
      info.layer = HelperMethods::getLayer(digit.getDetector());
      info.row = digit.getPadRow();
      info.col = digit.getPadCol();
      info.isShared = digit.isSharedDigit();
      info.channelGlb = HelperMethods::getChannelIndexInColumn(digit.getROB(), digit.getMCM(), digit.getChannel());
      info.indexGlb = indexGlobal;
    }
    // the ADC information we always want to have
    for (int i = 0; i < TIMEBINS; ++i) {
      auto adc = digit.getADC()[i];
      info.adcSum += adc;
      info.adcSumSquared += adc * adc;
      // mean and variance are calculated recursively
      float meanCurrent = info.adcMean;
      info.adcMean += (adc - meanCurrent) / (info.nEntries + 1);
      info.variance += info.nEntries * (info.nEntries + 1) * (info.adcMean - meanCurrent) * (info.adcMean - meanCurrent);
      info.nEntries += 1;
    }
  }
  mNDigitsSeen += digits.size();
  LOG(info) << "Processed " << digits.size() << " digits for this TF. Total number of processed digits: " << mNDigitsSeen;
}

void CalibratorNoise::collectChannelInfo()
{
  for (const auto& channel : mChannelInfosDetailed) {
    // create ChannelInfo object for array
    if (channel.nEntries > 0) {
      // we have seen data from this channel
      auto& ch = mCCDBObject.getChannel(channel.indexGlb);
      ch.setMean(channel.adcMean);
      ch.setRMS(std::sqrt(channel.variance / channel.nEntries));
      ch.setNentries(channel.nEntries);
    }
  }
}

} // namespace o2::trd
