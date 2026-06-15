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

//
//  TOF Chip class: it will be used to store the digits at TOF that
//  fall in the same Chip
//

////////////////////////////////////
// To put in O2/Detectors/Upgrades/ALICE3/IOTOF/base/include/IOTOFBase/Chip.h

#ifndef ALICEO2_IOTOF_CHIP_H_
#define ALICEO2_IOTOF_CHIP_H_

#include <DataFormatsIOTOF/Digit.h>
#include <TObject.h>
#include <exception>
#include <map>
#include <sstream>
#include <vector>
#include "MathUtils/Cartesian.h"

namespace o2::iotof
{

/// @class Chip
/// @brief Container for similated points connected to a given TOF Chip
/// This will be used in order to allow a more efficient clusterization
/// that can happen only between digits that belong to the same Chip
///

class Chip
{

 public:
  /// Default constructor
  Chip() = default;

  /// Destructor
  ~Chip() = default;

  /// Main constructor
  /// @param Chipindex Index of the Chip
  /// @param mat Transformation matrix
  Chip(Int_t index);

  /// Copy constructor
  /// @param ref Reference for the copy
  Chip(const Chip& ref) = default;

  /// Empties the point container
  /// @param option unused
  void clear() { mDigits.clear(); }

  std::map<ULong64_t, o2::iotof::LabeledDigit>& getDigits() { return mDigits; }
  bool isEmpty() const { return mDigits.empty(); }

  void setChipIndex(Int_t index) { mChipIndex = index; }
  Int_t getChipIndex() const { return mChipIndex; }

  void disable(bool disable) { mDisabled = disable; }
  bool isDisabled() const { return mDisabled; }

  /// Get the number of point assigned to the chip
  /// @return Number of points assigned to the chip
  Int_t getNumberOfDigits() const { return mDigits.size(); }

  /// reset points container
  o2::iotof::LabeledDigit* findDigit(ULong64_t key);

  void addDigit(UShort_t row, UShort_t col, Int_t charge, double time, o2::MCCompLabel label);

 protected:
  Int_t mChipIndex = -1;                                ///< Chip ID
  bool mDisabled = false;                               ///< Flag to indicate if the chip is disabled (e.g. due to dead channels)
  std::map<ULong64_t, o2::iotof::LabeledDigit> mDigits; ///< Map of fired digits, possibly in multiple frames

  ClassDefNV(Chip, 1);
};

inline o2::iotof::LabeledDigit* Chip::findDigit(ULong64_t key)
{
  // finds the digit corresponding to global key
  auto digitentry = mDigits.find(key);
  return digitentry != mDigits.end() ? &(digitentry->second) : nullptr;
}

} // namespace o2::iotof

#endif /* defined(ALICEO2_IOTOF_CHIP_H_) */