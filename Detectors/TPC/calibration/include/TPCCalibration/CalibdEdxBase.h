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

/// \file CalibdEdxBase.h
/// \brief
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBDEDXBASE_H_
#define ALICEO2_TPC_CALIBDEDXBASE_H_

#include <array>
#include <cstddef>

namespace o2::tpc
{

/// Define the commom properties of dEdx containers. Used by CalibdEdxHistos and CalibdEdx.
template <typename Entry>
class CalibdEdxBase
{
 public:
  static const size_t stacksPerTurn = 18;
  static const size_t totalStacks = 18 * 8;

  using Container = std::array<Entry, totalStacks>;

  enum class ReadoutType : unsigned short {
    IROC,
    OROC1,
    OROC2,
    OROC3,
  };

  enum class TPCSide {
    A,
    C
  };

  /// \brief Find the index of a stack.
  static size_t stackIndex(size_t stackNumber, ReadoutType, TPCSide);

  const Container& getTotEntries() const { return mTotEntries; };
  const Container& getMaxEntries() const { return mMaxEntries; };

  Entry getTotEntry(size_t stackNumber, ReadoutType type, TPCSide side) const
  {
    const size_t index = stackIndex(stackNumber, type, side);
    return mTotEntries[index];
  }

  Entry getMaxEntry(size_t stackNumber, ReadoutType type, TPCSide side) const
  {
    const size_t index = stackIndex(stackNumber, type, side);
    return mTotEntries[index];
  }

 protected:
  Container mTotEntries;
  Container mMaxEntries;
};

template <typename Entry>
size_t CalibdEdxBase<Entry>::stackIndex(size_t stackNumber, const ReadoutType type, const TPCSide side)
{
  // FIXME: stackNumber cant be greater than 17. How to handle input error?
  if (stackNumber > stacksPerTurn - 1) {
    stackNumber = stacksPerTurn - 1;
  };

  // Account for the readout type: IROC, OROC1, ...
  const auto type_number = static_cast<unsigned short>(type);
  stackNumber += type_number * stacksPerTurn;

  // Account for side
  constexpr size_t sideOffset = stacksPerTurn * 4;
  if (side == TPCSide::C) {
    stackNumber += sideOffset;
  }
  return stackNumber;
}

} // namespace o2::tpc

#endif
