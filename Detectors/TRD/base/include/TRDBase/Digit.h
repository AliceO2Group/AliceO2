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

#include "TRDBase/FeeParam.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{

using ADC_t = std::uint16_t;
using ArrayADC = std::array<ADC_t, constants::TIMEBINS>;

// Digit class for TRD
// Notes:
//    Shared pads:
//        the lowe mcm and rob is chosen for a given shared pad.
//        this negates the need for need alternate indexing strategies.
//        it does however mean that if you are trying to go from pad/row to mcm/rob you need to remember to manually do the shared ones.
//        TODO we could change the get methods to return the value in negaitve to indicate a shared pad, but I feel this is just added complexity? Comments?
//        if you are going to have to check for negative you may as well check for it being shared.
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
  void setROB(int row, int pad) { mROB = FeeParam::getROBfromPad(row, pad); }
  void setMCM(int mcm) { mMCM = mcm; }
  void setMCM(int row, int pad) { mMCM = FeeParam::getMCMfromPad(row, pad); }
  void setChannel(int channel) { mChannel = channel; }
  void setDetector(int det) { mDetector = det; }
  void setADC(ArrayADC const& adc) { mADC = adc; }
  // Get methods
  int getDetector() const { return mROB; }
  int getRow() const { return FeeParam::getPadRowFromMCM(mROB, mMCM); }
  int getPad() const { return FeeParam::getPadColFromADC(mROB, mMCM, mChannel); }
  int getROB() const { return mROB; }
  int getMCM() const { return mMCM; }
  int getChannel() const { return mChannel; }
  int isSharedDigit();

  ArrayADC const& getADC() const { return mADC; }
  ADC_t getADCsum() const { return std::accumulate(mADC.begin(), mADC.end(), (ADC_t)0); }

 private:
  std::uint16_t mDetector{0}; // detector, the chamber
  std::uint8_t mROB{0};       // read out board with in chamber
  std::uint8_t mMCM{0};       // MCM chip this digit is attached to
  std::uint8_t mChannel{0};   // channel of this chip the digit is attached to.

  ArrayADC mADC{}; // ADC vector (30 time-bins)
  ClassDefNV(Digit, 3);
};

} // namespace trd
} // namespace o2

#endif
