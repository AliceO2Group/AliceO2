// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file Digit.h
 * C++ simple Muon MCH digit.
 * @author  Michael Winn
 */

#ifndef ALICEO2_MCH_DIGIT_H_
#define ALICEO2_MCH_DIGIT_H_

#include "CommonDataFormat/TimeStamp.h"

namespace o2
{
namespace mch
{

// \class Digit
/// \brief MCH digit implementation
using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{
 public:
  Digit() = default;

  Digit(double time, int detid, int pad, double adc);
  ~Digit() = default;

  int getDetID() const { return mDetID; }
  void setDetID(int detid) { mDetID = detid; }

  int getPadID() const { return mPadID; }
  void setPadID(int pad) { mPadID = pad; }

  double getADC() const { return mADC; }
  void setADC(double adc) { mADC = adc; }

 private:
  int mDetID;
  int mPadID;  /// PadIndex to which the digit corresponds to
  double mADC; /// Amplitude of signal

  ClassDefNV(Digit, 1);
}; //class Digit

} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_DIGIT_H_
