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

/// \file Digits.cxx
/// \author Roman Lietava

#include "DataFormatsCTP/Digits.h"
#include <iostream>

using namespace o2::ctp;

std::ostream& o2::ctp::operator<<(std::ostream& os, const o2::ctp::CTPDigit& d)
{
  os << "CTP Digit: " << d.intRecord << " Input Mask: " << d.CTPInputMask << " Class Mask: " << d.CTPClassMask;
  return os;
}

void CTPDigit::printStream(std::ostream& stream) const
{
  stream << *this << std::endl;
}

void CTPDigit::setInputMask(gbtword80_t mask)
{
  for (uint32_t i = 0; i < CTP_NINPUTS; i++) {
    CTPInputMask[i] = mask[i];
  }
}
void CTPDigit::setClassMask(gbtword80_t mask)
{
  for (uint32_t i = 0; i < CTP_NCLASSES; i++) {
    CTPClassMask[i] = mask[i];
  }
}
