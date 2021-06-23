// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFV0/RawEventData.h"
#include <sstream>
#include <iostream>

using namespace o2::fv0;

ClassImp(RawEventData);
void RawEventData::printHexEventHeader() const
{
  std::stringstream ssheader;
  ssheader << std::setfill('0') << std::setw(16) << std::hex << mEventHeader.word[0] << " " << std::setw(16) << mEventHeader.word[1] << "\n       ";
  ssheader << std::setw(3) << (0x00000fff & mEventHeader.bc) << " "
           << std::setw(8) << (0xffffffff & mEventHeader.orbit) << " "
           << std::setw(5) << (0x000fffff & mEventHeader.reservedField1) << " "
           << std::setw(2) << (0x000000ff & mEventHeader.reservedField2) << " "
           << std::setw(1) << (0x0000000f & mEventHeader.nGBTWords) << " "
           << std::setw(1) << (0x0000000f & mEventHeader.startDescriptor) << " "
           << std::setw(12) << (0xffffffffffff & mEventHeader.reservedField3);
  LOG(DEBUG) << ssheader.str();
}

void RawEventData::printHexEventData(uint64_t i) const
{
  std::stringstream ssdata;
  ssdata << "D0:0x ";
  ssdata << std::setfill('0') << std::hex << std::setw(16) << mEventData[2 * i].word << "\n                   ";
  ssdata << std::setw(3) << (0x0fff & mEventData[2 * i].time) << " "
         << std::setw(8) << (0x1fff & mEventData[2 * i].charge) << "\n             ";
  ssdata << "D1:0x ";
  ssdata << std::setfill('0') << std::hex << std::setw(16) << mEventData[2 * i + 1].word << "\n                   ";
  ssdata << std::setw(3) << (0x0fff & mEventData[2 * i + 1].time) << " "
         << std::setw(8) << (0x1fff & mEventData[2 * i + 1].charge);
  LOG(DEBUG) << "    | " << ssdata.str();
}
