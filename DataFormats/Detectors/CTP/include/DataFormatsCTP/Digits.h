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
static constexpr uint64_t NCTPINPUTS = 46;
static constexpr uint64_t NCTPCLASSES = 64;
static constexpr uint32_t MAXCTPL0PERDET = 5;
struct CTPRawData {

  o2::InteractionRecord mIntRecord;
  std::bitset<NCTPINPUTS> mCTPInputMask;
  std::bitset<NCTPCLASSES> mCTPClassMask;
  CTPRawData() = default;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPRawData, 1);
};
struct CTPInputDigit{
    std::bitset<MAXCTPL0PERDET> mInputsMask;
    std::int32_t mDetector;
    CTPInputDigit() = default;
    CTPInputDigit(std::bitset<MAXCTPL0PERDET> InputsMask, uint32_t DetID)
    {
        mInputsMask = InputsMask;
        mDetector=DetID;
    }
};
struct CTPDigit{
    o2::InteractionRecord mIntRecord;
    std::vector<CTPInputDigit> mInputs;
    CTPDigit() = default;
    ClassDefNV(CTPDigit,1);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_DIGITS_H
