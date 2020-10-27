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

#include "CommonDataFormat/TimeStamp.h"
#include "TRDBase/TRDCommonParam.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{

using ADC_t = std::uint16_t;
using ArrayADC = std::array<ADC_t, constants::TIMEBINS>;

using TimeStamp = o2::dataformats::TimeStamp<double>;

class Digit : public TimeStamp
{
 public:
  Digit() = default;
  ~Digit() = default;
  Digit(const int det, const int row, const int pad, const ArrayADC adc, double t)
    : mDetector(det), mRow(row), mPad(pad), mADC(adc), TimeStamp(t) {}
  // Copy
  Digit(const Digit&) = default;
  // Assignment
  Digit& operator=(const Digit&) = default;
  // Modifiers
  void setDetector(int det) { mDetector = det; }
  void setRow(int row) { mRow = row; }
  void setPad(int pad) { mPad = pad; }
  void setADC(ArrayADC const& adc) { mADC = adc; }
  // Get methods
  int getDetector() const { return mDetector; }
  int getRow() const { return mRow; }
  int getPad() const { return mPad; }
  ArrayADC const& getADC() const { return mADC; }
  ADC_t getADCsum() const { return std::accumulate(mADC.begin(), mADC.end(), (ADC_t)0); }

 private:
  std::uint16_t mDetector{0}; // TRD detector number, 0-539
  std::uint8_t mRow{0};       // pad row, 0-15
  std::uint8_t mPad{0};       // pad within pad row, 0-143
  ArrayADC mADC{};            // ADC vector (30 time-bins)
  ClassDefNV(Digit, 2);
};

} // namespace trd
} // namespace o2

#endif
