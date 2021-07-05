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

/// \file Scalers.h
/// \brief definition of CTPScalerRaw, CTPScalerO2
/// \author Roman Lietava

#ifndef _CTP_SCALERS_H_
#define _CTP_SCALERS_H_
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsCTP/Digits.h"

#include <bitset>

namespace o2
{
namespace ctp
{
/// raw scalers produced by CTP and send to O2 either via 
/// - ZeroMQ published at CTP control machine
/// - CTPreadout to FLP
struct  CTPScalerRaw {
  uint8_t classIndex;
  uint32_t lmBefore;
  uint32_t lmAfter;
  uint32_t l0Before;
  uint32_t l0After;
  uint32_t l1Before;
  uint32_t l1After;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPScalerRaw, 1);
};
/// Scalers produced from raw scalers corrected for overflow
struct  CTPScalerO2 {
  uint8_t classIndex;
  uint64_t lmBefore;
  uint64_t lmAfter;
  uint64_t l0Before;
  uint64_t l0After;
  uint64_t l1Before;
  uint64_t l1After;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPScalerO2, 1);
};
struct CTPScalerRecordRaw
{
  o2::InteractionRecord intRecord;
  std::vector<CTPScalerRaw> scalers;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPScalerRecordRaw, 1);
};
struct CTPScalerRecordO2
{
  o2::InteractionRecord intRecord;
  std::vector<CTPScalerO2> scalers;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPScalerRecordO2, 1);
};
class CTPRunScalers
{
 public:
 private:
  uint8_t mVersion;
  uint32_t mRunNumber;
  std::bitset<CTP_NCLASSES> mClassMask;
  std::vector<CTPScalerRecordRaw> mScalerRecordRaw;
  std::vector<CTPScalerRecordO2> mScalerRecordO2;
  ClassDefNV(CTPRunScalers, 1);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_DIGITS_H
