// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDBase/Digit.h"
namespace o2::trd
{
Digit::Digit(const int det, const int row, const int pad, const ArrayADC adc)
{
  setDetector(det);
  setROB(row, pad);
  setMCM(row, pad);
  setADC(adc);
}
Digit::Digit(const int det, const int row, const int pad) // add adc data in a seperate step
{
  setDetector(det);
  setROB(row, pad);
  setMCM(row, pad);
}
Digit::Digit(const int det, const int rob, const int mcm, const int channel, const ArrayADC adc)
{
  setROB(rob);
  setMCM(mcm);
  setChannel(channel);
  setADC(adc);
}
Digit::Digit(const int det, const int rob, const int mcm, const int channel) // add adc data in a seperate step
{
  setROB(rob);
  setMCM(mcm);
  setChannel(channel);
}
int Digit::isSharedDigit()
{
  if (mChannel == 0 || mChannel == 19 || mChannel == 20) {
    return 1;
  } else {
    return 0;
  }
}

} // namespace o2::trd
