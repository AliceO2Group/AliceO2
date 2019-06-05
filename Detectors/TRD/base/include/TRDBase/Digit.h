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
#include <map>

#include "TRDBase/DigitIndex.h"

namespace o2
{
namespace trd
{

class DigitIndex;

typedef std::uint16_t Digit_t;                               // The digit type
typedef std::vector<Digit_t> DigitContainer_t;               // the digit container type
typedef std::vector<DigitIndex> DigitIndexContainer_t;       // the digit-index container type
typedef std::map<int, DigitContainer_t> DigitMapContainer_t; // a map container type for signal handling during digitization

constexpr int kTB = 30;

class Digit
{
 public:
  static int calculateKey(const int det, const int row, const int col)
  {
    return ((det << 12) | (row << 8) | col);
  }
  static int getDetectorFromKey(const int key)
  {
    return (key >> 12) & 0xFFF;
  }
  static int getRowFromKey(const int key)
  {
    return (key >> 8) & 0xF;
  }
  static int getColFromKey(const int key)
  {
    return key & 0xFF;
  }
  static void convertMapToVectors(DigitMapContainer_t& signalCont, DigitContainer_t& digits, DigitIndexContainer_t digit_index)
  {
    //
    // Create a digit and a digit-index container from a map container
    //
    int idx = 0;
    for (const auto& signal : signalCont) {
      int key = signal.first;
      digit_index.emplace_back(Digit::getDetectorFromKey(key),
                               Digit::getRowFromKey(key),
                               Digit::getColFromKey(key),
                               idx);
      DigitContainer_t dd = signal.second;
      std::copy(dd.begin(), dd.end(), std::back_inserter(digits));
      idx += kTB;
    }
  }
  static void convertVectorsToMap(DigitMapContainer_t& signalCont, DigitContainer_t& digits, DigitIndexContainer_t digit_index)
  {
    //
    // Create a map container from a digit and a digit-index container
    //
  }
};

} // namespace trd
} // namespace o2

#endif
