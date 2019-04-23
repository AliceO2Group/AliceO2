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

namespace o2
{
namespace trd
{

class DigitIndex
{
 public:
  DigitIndex() = default;
  ~DigitIndex() = default;
  DigitIndex(const int det, const int row, const int pad, const int index)
    : mDetector(det), mRow(row), mPad(pad), mIndex(index) {}
  DigitIndex(const DigitIndex&) = default;
  // Modifiers
  void setDetector(int det) { mDetector = det; }
  void setRow(int row) { mRow = row; }
  void setPad(int pad) { mPad = pad; }
  void setIndex(int index) { mIndex = index; }
  // Get methods
  int getDetector() const { return mDetector; }
  int getRow() const { return mRow; }
  int getPad() const { return mPad; }
  int getIndex() const { return mIndex; }

 private:
  unsigned int mDetector : 10; // 10 TRD detector number, 0-539
  unsigned int mRow : 4;       // 4 pad row, 0-15
  unsigned int mPad : 8;       // 8 pad within pad row, 0-14
  unsigned int mIndex : 25;    // 25 index of first time-bin data within adc_data array, max value 540*16*144*30=37324800, needs 25 bits
};

} // namespace trd
} // namespace o2

#endif
