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

#ifndef ALICEO2_TPC_CalibRawPartInfo_H_
#define ALICEO2_TPC_CalibRawPartInfo_H_

#include <cstdint>
#include "CommonDataFormat/TFIDInfo.h"

namespace o2::tpc
{

struct CalibRawPartInfo {
  int calibType{-1};
  uint32_t publishCycle{};
  dataformats::TFIDInfo tfIDInfo{};
};

constexpr bool operator==(const CalibRawPartInfo& lhs, const CalibRawPartInfo& rhs)
{
  return (lhs.calibType == rhs.calibType) &&
         (lhs.publishCycle == rhs.publishCycle) &&
         (lhs.tfIDInfo.firstTForbit == rhs.tfIDInfo.firstTForbit) &&
         (lhs.tfIDInfo.tfCounter == rhs.tfIDInfo.tfCounter) &&
         (lhs.tfIDInfo.runNumber == rhs.tfIDInfo.runNumber) &&
         (lhs.tfIDInfo.startTime == rhs.tfIDInfo.startTime) &&
         (lhs.tfIDInfo.creation == rhs.tfIDInfo.creation);
}

constexpr bool operator!=(const CalibRawPartInfo& lhs, const CalibRawPartInfo& rhs)
{
  return !(lhs == rhs);
}

} // namespace o2::tpc

#endif
