// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DIGIT_H_
#define ALICEO2_TRD_DIGIT_H_

#include <cstdint>
#include <vector>
#include <array>
#include <unordered_map>
#include <numeric>
#include "Rtypes.h" // for ClassDef

#include "DataFormatsTRD/HelperMethods.h"
#include "DataFormatsTRD/Constants.h"
#include <gsl/span>

namespace o2
{
namespace trd
{

using ADC_t = std::uint16_t;
using ArrayADC = std::array<ADC_t, constants::TIMEBINS>;

// Digit class for TRD
// Notes:
//    Shared pads:
//        the lower mcm and rob is chosen for a given shared pad.
//        this negates the need for need alternate indexing strategies.
//        if you are trying to go from mcm/rob/adc to pad/row and back to mcm/rob/adc ,
//        you may not end up in the same place, you need to remember to manually check for shared pads.

class Digit
{
 public:
  Digit() = default;
  ~Digit() = default;
  Digit(const int det, const int row, const int pad, const ArrayADC adc);
  Digit(const int det, const int row, const int pad); // add adc data in a seperate step
  Digit(const int det, const int rob, const int mcm, const int channel, const ArrayADC adc);
  Digit(const int det, const int rob, const int mcm, const int channel); // add adc data in a seperate step

  // Copy
  Digit(const Digit&) = default;
  // Assignment
  Digit& operator=(const Digit&) = default;
  // Modifiers
  void setROB(int rob) { mROB = rob; }
  void setROB(int row, int pad) { mROB = HelperMethods::getROBfromPad(row, pad); }
  void setMCM(int mcm) { mMCM = mcm; }
  void setMCM(int row, int pad) { mMCM = HelperMethods::getMCMfromPad(row, pad); }
  void setChannel(int channel) { mChannel = channel; }
  void setDetector(int det) { mDetector = det; }
  void setADC(ArrayADC const& adc) { mADC = adc; }
  void setADC(const gsl::span<ADC_t>& adc) { std::copy(adc.begin(), adc.end(), mADC.begin()); }
  // Get methods
  int getDetector() const { return mDetector; }
  int getHCId() const { return mDetector * 2 + (mROB % 2); }
  int getRow() const { return HelperMethods::getPadRowFromMCM(mROB, mMCM); }
  int getPad() const { return HelperMethods::getPadColFromADC(mROB, mMCM, mChannel); }
  int getROB() const { return mROB; }
  int getMCM() const { return mMCM; }
  int getChannel() const { return mChannel; }
  bool isSharedDigit() const;

  ArrayADC const& getADC() const { return mADC; }
  ADC_t getADCsum() const { return std::accumulate(mADC.begin(), mADC.end(), (ADC_t)0); }

  bool operator==(const Digit& o) const
  {
    return mDetector == o.mDetector && mROB == o.mROB && mMCM == o.mMCM && mChannel == o.mChannel && mADC == o.mADC;
  }

 private:
  std::uint16_t mDetector{0}; // detector, the chamber [0-539]
  std::uint8_t mROB{0};       // read out board within chamber [0-7] [0-5] depending on C0 or C1
  std::uint8_t mMCM{0};       // MCM chip this digit is attached [0-15]
  std::uint8_t mChannel{0};   // channel of this chip the digit is attached to, see TDP chapter ?? TODO fill in later the figure number of ROB to MCM mapping picture

  ArrayADC mADC{}; // ADC vector (30 time-bins)
  ClassDefNV(Digit, 3);
};

std::ostream& operator<<(std::ostream& stream, const Digit& d);

} // namespace trd
} // namespace o2

#endif
