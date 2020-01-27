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

#include "CommonDataFormat/TimeStamp.h"
#include "TRDBase/TRDCommonParam.h"

namespace o2
{
namespace trd
{

using ADC_t = std::uint16_t;
using ArrayADC = std::array<ADC_t, kTimeBins>;

using TimeStamp = o2::dataformats::TimeStamp<double>;

class Digit : public TimeStamp
{
 public:
  Digit() = default;
  ~Digit() = default;
  Digit(const int det, const int row, const int pad, const ArrayADC adc, const size_t idx, double t)
    : mDetector(det), mRow(row), mPad(pad), mADC(adc), mLabelIdx(idx), TimeStamp(t) {}
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
  size_t getLabelIndex() const { return mLabelIdx; }

 private:
  std::uint16_t mDetector{0}; // TRD detector number, 0-539
  std::uint8_t mRow{0};       // pad row, 0-15
  std::uint8_t mPad{0};       // pad within pad row, 0-143
  ArrayADC mADC{};            // ADC vector (30 time-bins)
  size_t mLabelIdx{0};        // index for mc label

  ClassDefNV(Digit, 1);
};

} // namespace trd
} // namespace o2

#endif
