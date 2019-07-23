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

namespace o2
{
namespace trd
{

class Digit;

constexpr int kTB = 30;
constexpr int KEY_MIN = 0;
constexpr int KEY_MAX = 2211727;

typedef std::uint16_t ADC_t;                                   // the ADC value type
typedef std::array<ADC_t, kTB> ArrayADC_t;                     // the array ADC
typedef std::vector<Digit> DigitContainer_t;                   // the digit container type
typedef std::unordered_map<int, ArrayADC_t> SignalContainer_t; // a map container type for signal handling during digitization

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
  static int calculateKey(const int det, const int row, const int col) { return ((det << 12) | (row << 8) | col); }
  static int getDetectorFromKey(const int key) { return (key >> 12) & 0xFFF; }
  static int getRowFromKey(const int key) { return (key >> 8) & 0xF; }
  static int getColFromKey(const int key) { return key & 0xFF; }
  static void convertMapToVectors(const SignalContainer_t& adcMapCont,
                                  DigitContainer_t& digitCont)
  {
    //
    // Create a digit and a digit-index container from a map container
    //
    digitCont.reserve(adcMapCont.size());
    for (const auto& element : adcMapCont) {
      const int key = element.first;
      digitCont.emplace_back(Digit::getDetectorFromKey(key),
                             Digit::getRowFromKey(key),
                             Digit::getColFromKey(key),
                             element.second);
    }
  }
  static void convertVectorsToMap(const DigitContainer_t& digitCont,
                                  SignalContainer_t& adcMapCont)
  {
    //
    // Create a map container from a digit and a digit-index container
    //
    for (const auto& element : digitCont) {
      const int key = calculateKey(element.getDetector(),
                                   element.getRow(),
                                   element.getPad());
      adcMapCont[key] = element.getADC();
    }
  }

 private:
  std::uint16_t mDetector{ 0 }; // TRD detector number, 0-539
  std::uint8_t mRow{ 0 };       // pad row, 0-15
  std::uint8_t mPad{ 0 };       // pad within pad row, 0-143
  ArrayADC_t mADC{};            // ADC vector (30 time-bins)

  ClassDefNV(Digit, 1);
};

} // namespace trd
} // namespace o2

#endif
