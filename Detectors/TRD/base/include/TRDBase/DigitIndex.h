// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DIGITINDEX_H_
#define ALICEO2_TRD_DIGITINDEX_H_

#include <cstdint>

namespace o2
{
namespace trd
{

class DigitIndex
{
 public:
  DigitIndex() = default;
  ~DigitIndex() = default;
  DigitIndex(const DigitIndex&) = default;
  DigitIndex(const int det, const int row, const int pad)
    : mDetector(det),
      mRow(row),
      mPad(pad) {}
  // Modifiers
  void setDetector(int det) { mDetector = det; }
  void setRow(int row) { mRow = row; }
  void setPad(int pad) { mPad = pad; }
  // Get methods
  int getDetector() const { return mDetector; }
  int getRow() const { return mRow; }
  int getPad() const { return mPad; }

 private:
  std::uint16_t mDetector{ 0 }; // TRD detector number, 0-539
  std::uint8_t mRow{ 0 };       // pad row, 0-15
  std::uint8_t mPad{ 0 };       // pad within pad row, 0-14
};

} // namespace trd
} // namespace o2

#endif
