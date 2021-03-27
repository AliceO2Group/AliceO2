// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsCTP/Digits.h"
#include <iostream>

using namespace o2::ctp;

void CTPRawData::printStream(std::ostream& stream) const
{
  stream << "CTP Digit:  BC " << mIntRecord.bc << " orbit " << mIntRecord.orbit << std::endl;
  stream << "Input Mask: " << mCTPInputMask << " Class Mask:  " << mCTPClassMask << std::endl;
}