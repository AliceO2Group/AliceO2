// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHCalibration/PedestalDigit.h"
#include <cmath>

namespace o2::mch::calibration
{

PedestalDigit::PedestalDigit(int solarid, int dsid, int ch, uint32_t trigTime, uint32_t time, std::vector<uint16_t> samples)
  : mSolarId(solarid), mDsId(dsid), mChannel(ch), mTrigTime(trigTime), mTime(time), mNofSamples(samples.size())
{
  mNofSamples = samples.size();
  if (mNofSamples > MCH_PEDESTALS_MAX_SAMPLES) {
    mNofSamples = MCH_PEDESTALS_MAX_SAMPLES;
  }

  for (uint16_t s = 0; s < mNofSamples; s++) {
    mSamples[s] = samples[s];
  }
}

int16_t PedestalDigit::getSample(uint16_t s) const
{
  int16_t result = -1;
  if (s < mNofSamples) {
    result = static_cast<int16_t>(mSamples[s]);
  }

  return result;
}

} // namespace o2::mch::calibration
