// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "CommonDataFormat/InteractionRecord.h"
#include <bitset>
#include <iosfwd>

#ifndef _CTP_DIGITS_H_
#define _CTP_DIGITS_H_
namespace o2
{
namespace ctp
{
struct CTPDigit {
  static constexpr uint64_t NCTPINPUTS = 46;
  static constexpr uint64_t NCTPCLASSES = 64;
  o2::InteractionRecord mIntRecord;
  std::bitset<NCTPINPUTS> mCTPInputMask;
  std::bitset<NCTPCLASSES> mCTPClassMask;
  CTPDigit() = default;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPDigit, 1);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_DIGITS_H
