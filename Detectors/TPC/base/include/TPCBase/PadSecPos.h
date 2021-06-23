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

///
/// @file   PadSecPos.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Pad and row inside a sector
///
/// This class encapsulates the pad and row inside a sector
/// @see TPCBase/PadSecPos.h
/// @see TPCBase/Sector.h
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_PadSecPos_H
#define AliceO2_TPC_PadSecPos_H

#include "TPCBase/Sector.h"
#include "TPCBase/PadPos.h"

namespace o2
{
namespace tpc
{
class PadSecPos
{
 public:
  PadSecPos() = default;

  PadSecPos(const int sector, const int rowInSector, const int padInRow) : mSector(sector), mPadPos(PadPos(rowInSector, padInRow)) {}
  PadSecPos(const Sector& sec, const PadPos& padPosition) : mSector(sec), mPadPos(padPosition) {}

  Sector getSector() const { return mSector; }

  Sector& getSector() { return mSector; }

  const PadPos& getPadPos() const { return mPadPos; }

  PadPos& getPadPos() { return mPadPos; }

 private:
  Sector mSector{};
  PadPos mPadPos{};
};
} // namespace tpc
} // namespace o2
#endif
