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

#ifndef ALICEO2_MCH_BASE_DIGIT_H_
#define ALICEO2_MCH_BASE_DIGIT_H_

#include "Rtypes.h"

namespace o2
{
namespace mch
{

// \class Digit
/// \brief MCH digit implementation
class Digit
{
 public:
  Digit() = default;

  Digit(int detid, int pad, uint32_t adc, int32_t time, uint16_t nSamples = 1);
  ~Digit() = default;

  bool operator==(const Digit&) const;

  // time in bunch crossing units, relative to the beginning of the TimeFrame
  void setTime(int32_t t) { mTFtime = t; }
  int32_t getTime() const { return mTFtime; }

  void setNofSamples(uint16_t n);
  uint16_t nofSamples() const { return (mNofSamples & 0x7FFF); }

  void setSaturated(bool sat);
  bool isSaturated() const { return ((mNofSamples & 0x8000) > 0); }

  int getDetID() const { return mDetID; }

  int getPadID() const { return mPadID; }
  void setPadID(int padID) { mPadID = padID; }

  uint32_t getADC() const { return mADC; }
  void setADC(uint32_t adc) { mADC = adc; }

 private:
  int32_t mTFtime;      /// time since the beginning of the time frame, in bunch crossing units
  uint16_t mNofSamples; /// number of samples in the signal
  int mDetID;           /// ID of the Detection Element to which the digit corresponds to
  int mPadID;           /// PadIndex to which the digit corresponds to
  uint32_t mADC;        /// Amplitude of signal

  ClassDefNV(Digit, 3);
}; //class Digit

} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_DIGIT_H_
