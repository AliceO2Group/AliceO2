// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsTRD/CompressedDigit.h"
#include <iostream>

namespace o2::trd
{

using namespace constants;

CompressedDigit::CompressedDigit(const int det, const int rob, const int mcm, const int channel, const std::array<uint16_t, constants::TIMEBINS>& adc)
{
  setDetector(det);
  setROB(rob);
  setMCM(mcm);
  setChannel(channel);
  setADC(adc);
}

CompressedDigit::CompressedDigit(const int det, const int rob, const int mcm, const int channel) // add adc data in a seperate step
{
  setDetector(det);
  setROB(rob);
  setMCM(mcm);
  setChannel(channel);
}

bool CompressedDigit::isSharedCompressedDigit() const
{
  if (getChannel() == 0 || getChannel() == 1 || getChannel() == NADCMCM - 1) {
    return 1;
  } else {
    return 0;
  }
}

std::ostream& operator<<(std::ostream& stream, CompressedDigit& d)
{
  stream << "CompressedDigit Det: " << d.getDetector() << " ROB: " << d.getROB() << " MCM: " << d.getMCM() << " Channel: " << d.getChannel() << " ADCs:";
  for (unsigned int i = 0; i < constants::TIMEBINS; i++) {
    stream << "[" << d[i] << "]";
  }
  return stream;
}

} // namespace o2::trd
