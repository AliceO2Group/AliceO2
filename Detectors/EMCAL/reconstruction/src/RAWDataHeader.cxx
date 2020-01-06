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

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::RAWDataHeader& header)
{
  stream << "EMCAL CRORC RAW Header:\n"
         << "  Size (WORD0): " << header.word0 << "(" << std::hex << header.word0 << std::dec << ")\n"
         << "  WORD1: " << header.word1 << "(" << std::hex << header.word1 << std::dec << ")\n"
         << "    Version:                " << int(header.version) << "\n"
         << "    L1 Trigger Message:     " << int(header.triggermessageL1) << "\n"
         << "    EventID1:               " << header.triggerBC << "\n"
         << "  Offset and size (WORD2): " << header.word2 << "(" << std::hex << header.word2 << std::dec << ")\n"
         << "    Offset:                 " << header.offsetToNext << "\n"
         << "    Size:                   " << header.memorySize << "\n"
         << "  Package counter and link ID (WORD3): (" << std::hex << header.word3 << std::dec << ")\n"
         << "    Packet counter:         " << int(header.packetCounter) << "\n"
         << "    Link ID:                " << int(header.linkID) << "\n"
         << "  Status and mini eventID: " << header.word4 << " (" << std::hex << header.word4 << std::dec << ")\n"
         << "    Status:                 " << header.status << "\n"
         << "    Mini EventID:           " << header.triggerOrbit << "\n"
         /*
         << "  Trigger Classes: ( " << std::hex << header.words5[0] << " " << header.words5[1] << " " << header.words5[2] << std::dec << ")\n"
         << "    First 50:               " << std::bitset<sizeof(uint64_t)*8>(header.triggerType) << "\n"
         << "    Second 50:              " << std::bitset<sizeof(uint64_t)*8>(header.triggerTypeNext50) << "\n"
         << "  ROI: (" << std::hex << header.words5[3] << " " << header.words5[4] << std::dec << ")\n"
         << "    ROI:                    " << std::bitset<sizeof(uint64_t)*8>(header.roi) << "\n"
         */
         << "End Header" << std::endl;
  return stream;
}

std::istream& o2::emcal::operator>>(std::istream& stream, o2::emcal::RAWDataHeader& header)
{
  //std::cout << "called, 10 words" << std::endl;
  uint32_t message[10];
  auto address = reinterpret_cast<char*>(message);
  for (int i = 0; i < 10; i++) {
    stream.read(address + i * sizeof(uint32_t) / sizeof(char), sizeof(message[i]));
    //std::cout << "Word " << i << ":  " << std::hex << message[i] << std::dec << std::endl;
  }
  header.word0 = message[0];
  header.word1 = message[1];
  header.word2 = message[2];
  header.word3 = message[3];
  header.word4 = message[4];
  header.word5 = message[5];
  header.word6 = message[6];
  header.word7 = message[7];
  header.word8 = message[8];
  header.word9 = message[9];
  return stream;
}