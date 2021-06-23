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

#ifndef AliceO2_TPC_DigitPos_H
#define AliceO2_TPC_DigitPos_H

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/CRU.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/PadSecPos.h"

namespace o2
{
namespace tpc
{

class DigitPos
{
 public:
  DigitPos() = default;
  DigitPos(CRU c, PadPos pad) : mCRU(c), mPadPos(pad) {}
  const CRU& getCRU() const { return mCRU; }
  CRU& cru() { return mCRU; }
  PadPos getPadPos() const { return mPadPos; }
  PadPos getGlobalPadPos() const;
  PadSecPos getPadSecPos() const;

  PadPos& padPos() { return mPadPos; }

  bool isValid() const { return mPadPos.isValid(); }

  bool operator==(const DigitPos& other) const { return (mCRU == other.mCRU) && (mPadPos == other.mPadPos); }
  bool operator!=(const DigitPos& other) const { return (mCRU != other.mCRU) || (mPadPos != other.mPadPos); }
  bool operator<(const DigitPos& other) const { return (mCRU < other.mCRU) && (mPadPos < other.mPadPos); }

 private:
  CRU mCRU{};
  PadPos mPadPos{}; /// Pad position in the local partition coordinates: row starts from 0 for each partition
};

} // namespace tpc
} // namespace o2
#endif
