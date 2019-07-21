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
#include "Rtypes.h" // for ClassDef

#include "TRDBase/TRDCommonParam.h" // For kNLayer

namespace o2
{
namespace trd
{

class Digit;

constexpr int kTB = 30;
constexpr int kNpad_rows = 16, kNpads = 144; // number of pad rows and pads per row
constexpr int KEY_MIN = 0;
constexpr int KEY_MAX = 2211727;

typedef std::uint16_t ADC_t;           // the ADC value type
typedef std::vector<ADC_t> ArrayADC_t; // the array ADC

class Digit
{
 public:
  Digit() = default;
  ~Digit() = default;
  Digit(const int det, const int row, const int pad, const ArrayADC_t adc)
    : mDetector(det), mRow(row), mPad(pad), mADC(adc) {}
  // Copy
  Digit(const Digit&) = default;
  // Assignment
  Digit& operator=(const Digit&) = default;
  // Modifiers
  void setDetector(int det) { mDetector = det; }
  void setRow(int row) { mRow = row; }
  void setPad(int pad) { mPad = pad; }
  void setADC(ArrayADC_t adc) { mADC = adc; }
  // Get methods
  int getDetector() const { return mDetector; }
  int getRow() const { return mRow; }
  int getPad() const { return mPad; }
  ArrayADC_t getADC() const { return mADC; }

  // Set of static helper methods
  static int getPos(const int, const int, const int, const int);
  static void convertSignalsToDigits(const std::vector<ADC_t>&, std::vector<Digit>&);

 private:
  std::uint16_t mDetector{ 0 }; // TRD detector number, 0-539
  std::uint8_t mRow{ 0 };       // pad row, 0-15
  std::uint8_t mPad{ 0 };       // pad within pad row, 0-143
  ArrayADC_t mADC{};            // ADC vector (30 time-bins)

  ClassDefNV(Digit, 1);
};

int getPos(const int det, const int row, const int pad, const int tb)
{
  // return det * (kNpad_rows * kNpads * kTB) + row * (kNpads * kTB) + pad * kTB + tb;
  return ((det * kNpad_rows + row) * kNpads + pad) * kTB + tb;
};

void convertSignalsToDigits(const std::vector<ADC_t>& signals, std::vector<Digit>& digits)
{
  for (int det = 0; det < kNdet; ++det) {
    for (int row = 0; row < kNpad_rows; row++) {
      for (int col = 0; col < kNpads; col++) {
        const int pos = Digit::calculateKey(det, row, col, 0);
        // check if pad has no signals
        const bool is_empty = std::all_of(signals.begin() + pos, signals.begin() + pos + kTB, [](const int& a) { return a == 0; });
        if (is_empty) {
          continue; // go to the next row
        }
        ArrayADC_t adc(signals.begin() + pos, signals.begin() + pos + kTB);
        digits.emplace_back(det, row, col, adc);
      } // loop over cols
    }   // loop over rows
  }     // loop over dets
}

} // namespace trd
} // namespace o2

#endif
