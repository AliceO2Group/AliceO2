// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _CTP_DIGITS_H_
#define _CTP_DIGITS_H_
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/Digit.h"
#include <bitset>
#include <iosfwd>

namespace o2
{
namespace ctp
{
static constexpr uint32_t CTP_NINPUTS = 46;
static constexpr uint32_t CTP_NCLASSES = 64;
static constexpr uint32_t CTP_MAXL0PERDET = 5;
// Positions of CTP Detector inputs in CTPInputMask
static constexpr std::pair<uint32_t, std::bitset<CTP_MAXL0PERDET>> CTP_INPUTMASK_FV0(0, 0x1f);
static constexpr std::pair<uint32_t, std::bitset<CTP_MAXL0PERDET>> CTP_INPUTMASK_FT0(5, 0x1f);
struct CTPDigit {
  o2::InteractionRecord mIntRecord;
  std::bitset<CTP_NINPUTS> mCTPInputMask;
  std::bitset<CTP_NCLASSES> mCTPClassMask;
  CTPDigit() = default;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPDigit, 1);
};
struct CTPInputDigit {
  o2::InteractionRecord mIntRecord;
  std::bitset<CTP_MAXL0PERDET> mInputsMask;
  std::int32_t mDetector;
  CTPInputDigit() = default;
  CTPInputDigit(o2::InteractionRecord IntRecord, std::bitset<CTP_MAXL0PERDET> InputsMask, uint32_t DetID)
  {
    mIntRecord = IntRecord;
    mInputsMask = InputsMask;
    mDetector = DetID;
  }
  ClassDefNV(CTPInputDigit, 1)
};
} // namespace ctp
} // namespace o2
#endif //_CTP_DIGITS_H
