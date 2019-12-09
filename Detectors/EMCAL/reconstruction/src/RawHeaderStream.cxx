// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <fstream>
#include <iostream>
#include "EMCALReconstruction/RawHeaderStream.h"

std::istream& o2::emcal::operator>>(std::istream& stream, o2::header::RAWDataHeaderV4& header)
{
  //std::cout << "called, 10 words" << std::endl;
  using wordtype = uint64_t;
  constexpr int RAWHEADERWORDS = sizeof(header) / sizeof(wordtype);
  wordtype message[RAWHEADERWORDS];
  auto address = reinterpret_cast<char*>(message);
  for (int i = 0; i < RAWHEADERWORDS; i++) {
    stream.read(address + i * sizeof(wordtype) / sizeof(char), sizeof(message[i]));
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
  return stream;
}

std::istream& o2::emcal::operator>>(std::istream& stream, o2::header::RAWDataHeaderV5& header)
{
  //std::cout << "called, 10 words" << std::endl;
  using wordtype = uint64_t;
  constexpr int RAWHEADERWORDS = sizeof(header) / sizeof(wordtype);
  wordtype message[RAWHEADERWORDS];
  auto address = reinterpret_cast<char*>(message);
  for (int i = 0; i < RAWHEADERWORDS; i++) {
    stream.read(address + i * sizeof(wordtype) / sizeof(char), sizeof(message[i]));
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
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::header::RAWDataHeaderV4& header)
{
  stream << "Raw data header V4:\n"
         << "  Word0 " << header.word0 << " (0x" << std::hex << header.word0 << std::dec << ")\n"
         << "    Version:         " << header.version << "\n"
         << "    Header size:     " << header.headerSize << "\n"
         << "    Block length:    " << header.blockLength << "\n"
         << "    FEE ID:          " << header.feeId << "\n"
         << "    Priority:        " << header.priority << "\n"
         << "  Word1 " << header.word1 << " (0x" << std::hex << header.word1 << std::dec << ")\n"
         << "    Offset to next:  " << header.offsetToNext << "\n"
         << "    Payload size (B):" << header.memorySize << "\n"
         << "    Packet counter:  " << static_cast<int>(header.packetCounter) << "\n"
         << "    Link ID:         " << static_cast<int>(header.linkID) << "\n"
         << "    Card ID:         " << header.cruID << "\n"
         << "    Endpoint:        " << static_cast<int>(header.endPointID) << "\n"
         << "  Word2 " << header.word2 << " (0x" << std::hex << header.word2 << std::dec << ")\n"
         << "    Trigger orbit:   " << header.triggerOrbit << "\n"
         << "    Heartbeat orbit: " << header.heartbeatOrbit << "\n"
         << "  Word3 " << header.word3 << " (0x" << std::hex << header.word3 << std::dec << ")\n"
         << "  Word4 " << header.word4 << " (0x" << std::hex << header.word4 << std::dec << ")\n"
         << "    Trigger BC:      " << header.triggerBC << "\n"
         << "    Heartbeat BC:    " << header.heartbeatBC << "\n"
         << "    Trigger Type:    " << header.triggerType << "\n"
         << "  Word5 " << header.word5 << " (0x" << std::hex << header.word5 << std::dec << ")\n"
         << "  Word6 " << header.word6 << " (0x" << std::hex << header.word6 << std::dec << ")\n"
         << "    Detector Field:  " << header.detectorField << "\n"
         << "    PAR:             " << header.par << "\n"
         << "    STOP:            " << header.stop << "\n"
         << "    Page count:      " << header.pageCnt << "\n"
         << "  Word7 " << header.word7 << " (0x" << std::hex << header.word7 << std::dec << ")\n"
         << "End header\n";
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::header::RAWDataHeaderV5& header)
{
  stream << "Raw data header V5:\n"
         << "   Word0 " << header.word0 << " (0x" << std::hex << header.word0 << std::dec << ")\n"
         << "   Word1 " << header.word1 << " (0x" << std::hex << header.word1 << std::dec << ")\n"
         << "   Word2 " << header.word2 << " (0x" << std::hex << header.word2 << std::dec << ")\n"
         << "   Word3 " << header.word3 << " (0x" << std::hex << header.word3 << std::dec << ")\n"
         << "   Word4 " << header.word4 << " (0x" << std::hex << header.word4 << std::dec << ")\n"
         << "   Word5 " << header.word5 << " (0x" << std::hex << header.word5 << std::dec << ")\n"
         << "   Word6 " << header.word6 << " (0x" << std::hex << header.word6 << std::dec << ")\n"
         << "   Word7 " << header.word7 << " (0x" << std::hex << header.word7 << std::dec << ")\n"
         << "End header\n";
  return stream;
}