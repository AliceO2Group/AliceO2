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
//    Pre Trigger phase:
//        LHC clock runs at 40.08MHz, ADC run at 1/4 or 10MHz and the trap runs at 120MHz or LHC*3.
//        The trap clock can therefore be in 1 of 12 positions relative to the ADC clock.
//        Only 4 of those positions are valid, dependent on the TRAP assembly program timing offset, as of 11/2022 pre-trigger phase: 0,3,6,9.
//        vaguely graphically below:
//        LHC  ___---___---___---___---___---___---___---___---___---___---___---___---___---___---___---___---___---___---___---___---___---___---___
//        TRAP _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//        ADC  ___------------____________------------____________------------____________------------____________------------____________------------
//             _________------------____________------------____________------------____________------------____________------------____________------
//             ---____________------------____________------------____________------------____________------------____________------------____________
//             ---------____________------------____________------------____________------------____________------------____________------------______
// PreTrig     _________------________________________------________________________------________________________------________________________------

class Digit
{
 public:
  Digit() = default;
  ~Digit() = default;
  Digit(int det, int row, int pad, ArrayADC adc, int phase = 0);
  Digit(int det, int row, int pad); // add adc data and pretrigger phase in a separate step
  Digit(int det, int rob, int mcm, int channel, ArrayADC adc, int phase = 0);
  Digit(int det, int rob, int mcm, int channel); // add adc data in a seperate step

  // Copy
  Digit(const Digit&) = default;
  // Assignment
  Digit& operator=(const Digit&) = default;
  // Modifiers
  void setROB(int rob) { mROB = rob; }
  void setMCM(int mcm) { mMCM = mcm; }
  void setROB(int row, int col) { mROB = HelperMethods::getROBfromPad(row, col); } // set ROB from pad row, column
  void setMCM(int row, int col) { mMCM = HelperMethods::getMCMfromPad(row, col); } // set MCM from pad row, column
  void setChannel(int channel) { mChannel = channel; }
  void setDetector(int det) { mDetector = ((mDetector & 0xf000) | (det & 0xfff)); }
  void setADC(ArrayADC const& adc) { mADC = adc; }
  void setADC(const gsl::span<ADC_t>& adc) { std::copy(adc.begin(), adc.end(), mADC.begin()); }
  void setPreTrigPhase(int phase) { mDetector = (((phase & 0xf) << 12) | (mDetector & 0xfff)); }
  // Get methods
  int getDetector() const { return mDetector & 0xfff; }
  int getHCId() const { return (mDetector & 0xfff) * 2 + (mROB % 2); }
  int getPadRow() const { return HelperMethods::getPadRowFromMCM(mROB, mMCM); }
  int getPadCol() const { return HelperMethods::getPadColFromADC(mROB, mMCM, mChannel); }
  int getROB() const { return mROB; }
  int getMCM() const { return mMCM; }
  int getChannel() const { return mChannel; }
  int getPreTrigPhase() const { return ((mDetector >> 12) & 0xf); }
  bool isSharedDigit() const;
  bool isNeighbour(const Digit& other) const;

  ArrayADC const& getADC() const { return mADC; }
  ADC_t getADCsum() const { return std::accumulate(mADC.begin(), mADC.end(), (ADC_t)0); }
  // returns the max ADC value and sets idx to the time bin with the largest ADC value
  ADC_t getADCmax(int& idx) const;
  ADC_t getADCval(int tb) const { return mADC[tb]; }

  bool operator==(const Digit& o) const
  {
    return mDetector == o.mDetector && mROB == o.mROB && mMCM == o.mMCM && mChannel == o.mChannel && mADC == o.mADC;
  }

 private:
  /// starting from ClassDef version 4, mDetector keeps both the chamber number and the trigger phase
  ///
  /// bits 0-11 contain the chamber number (valid range from 0-539)
  /// bits 12-15 contain the trigger phase obtained from digit HC header.
  /// |15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
  /// |  phase    |         chamber ID                |
  std::uint16_t mDetector{0};
  std::uint8_t mROB{0};     ///< read out board within chamber [0-7] [0-5] depending on C0 or C1
  std::uint8_t mMCM{0};     ///< MCM chip this digit is attached [0-15]
  std::uint8_t mChannel{0}; ///< channel of this chip the digit is attached to, see TDP chapter ?? TODO fill in later the figure number of ROB to MCM mapping picture
  ArrayADC mADC{};          ///< ADC values for 30 time bins (fixed size array)
  ClassDefNV(Digit, 4);
};

std::ostream& operator<<(std::ostream& stream, const Digit& d);

} // namespace trd
} // namespace o2

#endif
