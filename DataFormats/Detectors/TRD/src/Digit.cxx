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

#include "DataFormatsTRD/Digit.h"
#include <iostream>

namespace o2::trd
{

using namespace constants;

Digit::Digit(const int det, const int row, const int pad, const ArrayADC adc)
{
  setDetector(det);
  setROB(row, pad);
  setMCM(row, pad);
  setADC(adc);
  setChannel(NADCMCM - 2 - (pad % NCOLMCM));
}

Digit::Digit(const int det, const int row, const int pad) // add adc data in a seperate step
{
  setDetector(det);
  setROB(row, pad);
  setMCM(row, pad);
  setChannel(NADCMCM - 2 - (pad % NCOLMCM));
}

Digit::Digit(const int det, const int rob, const int mcm, const int channel, const ArrayADC adc)
{
  setDetector(det);
  setROB(rob);
  setMCM(mcm);
  setChannel(channel);
  setADC(adc);
}

Digit::Digit(const int det, const int rob, const int mcm, const int channel) // add adc data in a seperate step
{
  setDetector(det);
  setROB(rob);
  setMCM(mcm);
  setChannel(channel);
}

bool Digit::isSharedDigit() const
{
  if (mChannel == 0 || mChannel == 1 || mChannel == NADCMCM - 1) {
    return 1;
  } else {
    return 0;
  }
}

std::ostream& operator<<(std::ostream& stream, const Digit& d)
{
  stream << "Digit Det: " << d.getDetector() << " ROB: " << d.getROB() << " MCM: " << d.getMCM() << " Channel: " << d.getChannel() << " ADCs:";
  for (int i = 0; i < constants::TIMEBINS; i++) {
    stream << "[" << d.getADC()[i] << "]";
  }
  return stream;
}

} // namespace o2::trd
