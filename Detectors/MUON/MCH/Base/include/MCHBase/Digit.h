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
  struct Time {
    union {
      // default value
      uint64_t time = 0x0000000000000000;
      struct {                       ///
        uint32_t sampaTime : 10;     /// bit 0 to 9: sampa time
        uint32_t bunchCrossing : 20; /// bit 10 to 29: bunch crossing counter
        uint32_t reserved : 2;       /// bit 30 to 31: reserved
        uint32_t orbit;              /// bit 32 to 63: orbit
      };                             ///
    };

    Time() = default;
    Time(uint32_t s, uint32_t b, uint32_t o) : sampaTime(s), bunchCrossing(b), orbit(o) { }
    Time(uint64_t t) : time(t) { }

    uint64_t getBXTime()
    {
      return (bunchCrossing + (sampaTime * 4));
    }

    bool operator==(const Time&) const;
  };

  Digit() = default;

  Digit(int detid, int pad, unsigned long adc, Time time, uint16_t nSamples = 1);
  ~Digit() = default;

  bool operator==(const Digit&) const;

  Time getTime() const { return mTime; }

  uint16_t nofSamples() const { return mNofSamples; }
  void setNofSamples(uint16_t n) { mNofSamples = n; }

  int getDetID() const { return mDetID; }

  int getPadID() const { return mPadID; }
  void setPadID(int padID) { mPadID = padID; }

  unsigned long getADC() const { return mADC; }
  void setADC(unsigned long adc) { mADC = adc; }

 private:
  Time mTime;
  uint16_t mNofSamples; /// number of samples in the signal
  int mDetID;
  int mPadID;         /// PadIndex to which the digit corresponds to
  unsigned long mADC; /// Amplitude of signal

  ClassDefNV(Digit, 2);
}; //class Digit

} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_DIGIT_H_
