// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <iomanip>
#include <iostream>
#include "DataFormatsTRD/RawData.h"

namespace o2
{

namespace trd
{

uint32_t unpacklinkinfo(const HalfCRUHeader& cruhead, const uint32_t link, const bool data = true)
{
  // cruhead is the incoming header to pass
  // link is the link you are requesting information on, 0-14
  //
  uint32_t info = 0;
  if (link > 14)
    return 0xffffffff;

  char* words;
  if (link < 8)
    words = (char*)&cruhead.word02[0];
  if (link < 15)
    words = (char*)&cruhead.word57[0];
  uint32_t byteoffset = link * 3; // each link [size+erroflags] is 24 bits, so 3 bytes.
  if (data)
    info = ((words[byteoffset + 1]) << 8) + (words[byteoffset + 2]);
  else
    info = words[byteoffset];

  return info;
}

uint32_t getlinkerrorflag(const HalfCRUHeader& cruhead, const uint32_t link)
{
  // link is the link you are requesting information on, 0-14
  uint32_t errorflags = 0;
  if (link < 8) {
    //dealing with word0-2
    errorflags = cruhead.linksA[link].errorflags;
  } else {
    if (link < 16) {
      errorflags = cruhead.linksB[link - 8 + 1].errorflags; // link 0 [actually 9] is in fact the end part of reserved.

    } else
      std::cout << "error link=" << link << " not in range 0-14" << std::endl;
  }
  return errorflags;
}

uint32_t getlinkdatasize(const HalfCRUHeader& cruhead, const uint32_t link)
{
  // link is the link you are requesting information on, 0-14
  //return number 32 byte blocks for the link 3x64bit ints.
  uint32_t size = 0;
  if (link < 8) {
    size = (cruhead.linksA[link].size);
  } else {
    if (link < 16) { // link 0 is part of reserved
      size = cruhead.linksB[link - 8 + 1].size;

    } else
      std::cout << "error link=" << link << " not in range 0-14" << std::endl;
  }
  return size;
}

uint32_t getlinkerrorflags(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linkerrorflags)
{
  // retrieve all the link error flags for this half cru
  for (uint32_t link = 0; link < 15; link++) {
    linkerrorflags[link] = getlinkerrorflag(cruheader, link);
  }
  return 0;
}
uint32_t getlinkdatasizes(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linksizes)
{
  // retrieve all the link sizes for this half cru
  for (uint32_t link = 0; link < 15; link++) {
    linksizes[link] = getlinkdatasize(cruheader, link);
  }
  return 0;
}

//
//  Printing methods to dump and display the various structures above in pretty format or hexdump
//  printNameOfStruct(const NameOfStruct& nameofstruct);
//  dumpNameOfStruct(const NameOfStruct& nameofstruct);
//  std::ostrea& operator<<(std::ostream& stream, const NameOfStruct& nameofstruct);
//

std::ostream& operator<<(std::ostream& stream, const TrackletHCHeader halfchamberheader)
{
  stream << "TrackletHCHeader : Raw:0x" << std::hex << halfchamberheader.word << " " << halfchamberheader.format << " ;; " << halfchamberheader.MCLK << " :: " << halfchamberheader.one << " :: " << halfchamberheader.HCID << std::endl;
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const TrackletMCMData& tracklet)
{
  // make a pretty output of the tracklet.
  stream << "TrackletMCMData: Raw:0x" << std::hex << tracklet.word << " pos=" << tracklet.pos << "::slope=" << tracklet.slope << "::pid=" << tracklet.pid << "::checkbit=" << tracklet.checkbit << std::endl;
  return stream;
}
void printTrackletMCMData(o2::trd::TrackletMCMData& tracklet)
{
  LOGF(INFO, "TrackletMCMData: Raw:0x%08x pos:%d slope:%d pid:0x%08x checkbit:0x%02x", tracklet.word, tracklet.pos, tracklet.slope, tracklet.pid, tracklet.checkbit);
}

void printTrackletMCMHeader(o2::trd::TrackletMCMHeader& mcmhead)
{
  LOGF(INFO, "MCMRawHeader: Raw:0x%08x 1:%d padrow: 0x%02x col: 0x%01x pid2 0x%02x pid1: 0x%02x pid0: 0x%02x 1:%d", mcmhead.word, mcmhead.onea, mcmhead.padrow, mcmhead.col, mcmhead.pid2, mcmhead.pid1, mcmhead.pid0, mcmhead.oneb);
}

std::ostream& operator<<(std::ostream& stream, const TrackletMCMHeader& mcmhead)
{
  // make a pretty output of the mcm header.
  // stream << "MCMRawHeader:" << mcmhead.checkbits << "::" << (mcmhead.pid&0xfff000000) << ":"<<  (mcmhead.pid&0xfff000) << ":"<< (mcmhead.pid&0xfff) << std::endl;
  stream << "MCMRawHeader: Raw:0x" << std::hex << mcmhead.word << " " << mcmhead.onea << "::" << mcmhead.pid2 << ":" << mcmhead.pid1 << ":" << mcmhead.pid0 << "::" << mcmhead.oneb << std::endl;
  return stream;
}

void printHalfChamber(o2::trd::TrackletHCHeader& halfchamber)
{
  LOGF(INFO, "TrackletHCHeader: Raw:0x%08x HCID : 0x%0x MCLK: 0x%0x Format: 0x%0x Always1:0x%0x", halfchamber.HCID, halfchamber.MCLK, halfchamber.format, halfchamber.one);
}

void dumpHalfChamber(o2::trd::TrackletHCHeader& halfchamber)
{
  LOGF(INFO, "HalfChamber : 0x%08x", halfchamber.word);
}

void printHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::array<uint32_t, 15> sizes;
  std::array<uint32_t, 15> errorflags;
  getlinkdatasizes(halfcru, sizes);
  getlinkerrorflags(halfcru, errorflags);
  LOGF(INFO, "V:%d BC:%d SB:%d EType:%d", halfcru.HeaderVersion, halfcru.BunchCrossing, halfcru.StopBit, halfcru.EventType);
  for (int i = 0; i < 15; i++)
    LOGF(INFO, "Link %d size: %ul eflag: 0x%02x", i, sizes[i], errorflags[i]);
}

void dumpHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::array<uint32_t, 16> raw{};
  memcpy(&raw[0], &halfcru, sizeof(halfcru));
  for (int i = 0; i < 4; i++) {
    int index = 4 * i;
    LOGF(INFO, "[1/2CRUHeader %d] 0x%08x 0x%08x 0x%08x 0x%08x", i, raw[index + 3], raw[index + 2], raw[index + 1], raw[index + 0]);
  }
}

std::ostream& operator<<(std::ostream& stream, const HalfCRUHeader& halfcru) // make a pretty output of the header.
{
  stream << std::hex;
  stream << "EventType : " << halfcru.EventType << std::endl;
  stream << "StopBit : " << halfcru.StopBit << std::endl;
  stream << "BunchCrossing : " << halfcru.BunchCrossing << std::endl;
  stream << "HeaderVersion : " << halfcru.HeaderVersion << std::endl;
  stream << "link  sizes : ";
  for (int link = 0; link < 15; link++)
    stream << link << ":" << std::hex << std::setw(4) << getlinkdatasize(halfcru, link) << ",";
  stream << std::endl;
  stream << "link  errorflags : ";
  for (int link = 0; link < 15; link++)
    stream << link << ":" << std::hex << std::setw(2) << getlinkerrorflag(halfcru, link) << ",";
  stream << std::endl;
  stream << "0x" << halfcru.word02[0] << " 0x" << halfcru.word02[1] << " 0x" << halfcru.word02[2] << " 0x" << halfcru.word3 << " 0x" << halfcru.word4 << " 0x" << halfcru.word57[0] << " 0x" << halfcru.word57[1] << " 0x" << halfcru.word57[2] << std::endl;
  return stream;
}

} // namespace trd
} // namespace o2
