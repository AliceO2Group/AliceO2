// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digit.h
/// \brief Definition of the ZDC Digit class

#ifndef ALICEO2_ZDC_DIGIT_H_
#define ALICEO2_ZDC_DIGIT_H_

#include "CommonDataFormat/TimeStamp.h"

namespace o2
{
namespace zdc
{
using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{
 public:
  Digit() = default;

  void add(short adc) { mADC += adc; }
  void setDetInfo(char detID, char sectorID)
  {
    mDetID = detID;
    mSecID = sectorID;
  }
  short getADC() const { return mADC; }
  short getDetID() const { return mDetID; }
  short getSector() const { return mSecID; }

 private:
  short mADC = 0; // adc amplitude value ~ charge
                  // potentially generalize to have complete signal shape
  char mDetID = -1;
  char mSecID = -1; // sector x channel information where this digit is recorded ((0 to 4) x (0 to 5))

  ClassDefNV(Digit, 1);
};

} // namespace zdc
} // namespace o2

#endif
