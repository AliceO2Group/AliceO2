// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file PedestalDigit.h
 * C++ Muon MCH digit with ADC samples information.
 * @author  Andrea Ferrero
 */

#ifndef ALICEO2_MCH_CALIBRATION_PEDESTAL_DIGIT_H_
#define ALICEO2_MCH_CALIBRATION_PEDESTAL_DIGIT_H_

#include <vector>
#include "Rtypes.h"

#define MCH_PEDESTALS_MAX_SAMPLES 20

namespace o2
{
namespace mch
{
namespace calibration
{

// \class PedestalDigit
/// \brief MCH "fat" digit implementation for pedestal data
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
}; //class PedestalDigit

} //namespace calibration
} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_CALIBRATION_PEDESTAL_DIGIT_H_
