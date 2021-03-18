//
// Created by rl on 3/17/21.
//
#include "CommonDataFormat/InteractionRecord.h"
#include <bitset>
#include <iostream>

#ifndef _CTP_DIGITS_H_
#define _CTP_DIGITS_H_
namespace o2
{
namespace  ctp
{
struct CTPdigit
{
  static constexpr uint64_t NCTPINPUTS=46;
  static constexpr uint64_t NCTPCLASSES=64;
  o2::InteractionRecord mIntRecord;
  std::bitset<NCTPINPUTS> mCTPInputMask;
  std::bitset<NCTPCLASSES> mCTPClassMask;
  CTPdigit() = default;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPdigit,1);
};
}
}
#endif //_CTP_DIGITS_H
