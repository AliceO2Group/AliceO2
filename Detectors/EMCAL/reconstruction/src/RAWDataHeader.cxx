// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <bitset>
#include <iostream>
#include <iomanip>
#include "EMCALReconstruction/RAWDataHeader.h"

using namespace o2::emcal;

void RAWDataHeader::printStream(std::ostream& stream) const
{
  stream << "EMCAL CRORC RAW Header:\n"
         << "  Size (WORD0): " << mSize << "(" << std::hex << mSize << std::dec << ")\n"
         << "  WORD1: " << mWord1 << "(" << std::hex << mWord1 << std::dec << ")\n"
         << "    Version:                " << int(getVersion()) << "\n"
         << "    L1 Trigger Message:     " << int(getL1TriggerMessage()) << "\n"
         << "    EventID1:               " << getEventID1() << "\n"
         << "  Offset and size (WORD2): " << mBlockSizeOffset << "(" << std::hex << mBlockSizeOffset << std::dec << ")\n"
         << "    Offset:                 " << getOffset() << "\n"
         << "    Size:                   " << getBlockSize() << "\n"
         << "  Package counter and link ID (WORD3): (" << std::hex << mPacketCounterLink << std::dec << ")\n"
         << "    Packet counter:         " << int(getPacketCounter()) << "\n"
         << "    Link ID:                " << int(getLink()) << "\n"
         << "  Status and mini eventID: " << mStatusMiniEventID << " (" << std::hex << mStatusMiniEventID << std::dec << ")\n"
         << "    Status:                 " << getStatus() << "\n"
         << "    Mini EventID:           " << getMiniEventID() << "\n"
         << "  Trigger Classes: ( " << std::hex << mTriggerClassLow << " " << mTriggerClassesMiddleLow << " " << mTriggerClassesMiddleHigh << std::dec << ")\n"
         << "    First 50:               " << std::bitset<sizeof(decltype(getTriggerClasses()))>(getTriggerClasses()) << "\n"
         << "    Second 50:              " << std::bitset<sizeof(decltype(getTriggerClassesNext50()))>(getTriggerClassesNext50()) << "\n"
         << "  ROI: (" << std::hex << mROILowTriggerClassHigh << " " << mROIHigh << std::dec << ")\n"
         << "    ROI:                    " << std::bitset<sizeof(decltype(getROI()))>(getROI()) << "\n"
         << "End Header" << std::endl;
}

void RAWDataHeader::readStream(std::istream& stream)
{
  //std::cout << "called, 10 words" << std::endl;
  uint32_t message[10];
  auto address = reinterpret_cast<char*>(message);
  for (int i = 0; i < 10; i++) {
    stream.read(address + i * sizeof(uint32_t) / sizeof(char), sizeof(message[i]));
    //std::cout << "Word " << i << ":  " << std::hex << message[i] << std::dec << std::endl;
  }
  mSize = message[0];
  mWord1 = message[1];
  mBlockSizeOffset = message[2];
  mPacketCounterLink = message[3];
  mStatusMiniEventID = message[4];
  mTriggerClassLow = message[5];
  mTriggerClassesMiddleLow = message[6];
  mTriggerClassesMiddleHigh = message[7];
  mROILowTriggerClassHigh = message[8];
  mROIHigh = message[9];
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::RAWDataHeader& header)
{
  header.printStream(stream);
  return stream;
}

std::istream& o2::emcal::operator>>(std::istream& stream, o2::emcal::RAWDataHeader& header)
{
  header.readStream(stream);
  return stream;
}