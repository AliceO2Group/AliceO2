//
// Created by rl on 3/17/21.
//
#include "DataFormatsCTP/Digits.h"

using namespace o2::ctp;

void CTPdigit::printStream(std::ostream& stream) const
{
  stream << "CTP Digit:  BC " << mIntRecord.bc << " orbit " << mIntRecord.orbit << std::endl;
  stream << "Input Mask: " << mCTPInputMask << " Class Mask:  " << mCTPClassMask << std::endl;
}