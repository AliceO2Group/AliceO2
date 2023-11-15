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

/** @file PedestalDigit.h
 * C++ Muon MCH digit with ADC samples information.
 * @author Andrea Ferrero
 */

#ifndef O2_MCH_CALIBRATION_PEDESTAL_DIGIT_H_
#define O2_MCH_CALIBRATION_PEDESTAL_DIGIT_H_

#include "Rtypes.h"

#define MCH_PEDESTALS_MAX_SAMPLES 20

namespace o2::mch::calibration
{

/**
 * @class PedestalDigit
 * @brief "Fat" digit for pedestal data.
 *
 * In contrast to a "regular" digit, a PedestalDigit stores ADC samples
 * (up to `MCH_PEDESTALS_MAX_SAMPLES` samples)
 *
 */
class PedestalDigit
{
 public:
  PedestalDigit() = default;

  PedestalDigit(int solarid, int ds, int ch, uint32_t trigTime, uint32_t time, std::vector<uint16_t> samples);
  ~PedestalDigit() = default;

  uint32_t getTime() const { return mTime; }
  uint32_t getTriggerTime() const { return mTrigTime; }

  uint16_t nofSamples() const { return mNofSamples; }
  int16_t getSample(uint16_t s) const;

  int getSolarId() const { return mSolarId; }
  int getDsId() const { return mDsId; }
  int getChannel() const { return mChannel; }

 private:
  uint32_t mTime{0};
  uint32_t mTrigTime{0};
  uint16_t mNofSamples{0}; /// number of samples in the signal
  uint16_t mSamples[MCH_PEDESTALS_MAX_SAMPLES];
  int mSolarId;
  int mDsId;    /// PadIndex to which the digit corresponds to
  int mChannel; /// Amplitude of signal

  ClassDefNV(PedestalDigit, 1);
};

} // namespace o2::mch::calibration
#endif
