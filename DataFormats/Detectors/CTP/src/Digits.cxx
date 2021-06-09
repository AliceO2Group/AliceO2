// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digits.cxx
/// \author Roman Lietava

#include "DataFormatsCTP/Digits.h"
#include <iostream>

using namespace o2::ctp;

void CTPDigit::printStream(std::ostream& stream) const
{
  stream << "CTP Digit:  BC " << intRecord.bc << " orbit " << intRecord.orbit << std::endl;
  stream << "Input Mask: " << CTPInputMask << std::endl;
}
