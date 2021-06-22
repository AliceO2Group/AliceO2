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
#include <iomanip>
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/LinkRecord.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{

namespace trd
{

void buildTrackletHCHeader(TrackletHCHeader& header, int sector, int stack, int layer, int side, int chipclock, int format)
{
  header.MCLK = chipclock;
  header.format = format;
  header.one = 1;
  header.supermodule = sector;
  header.stack = stack;
  header.layer = layer;
  header.side = side;
}

void buildTrackletHCHeaderd(TrackletHCHeader& header, int detector, int rob, int chipclock, int format)
{
  int sector = (detector % (constants::NLAYER * constants::NSTACK));
  int stack = (detector % constants::NLAYER);
  int layer = ((detector % (constants::NLAYER * constants::NSTACK)) / constants::NLAYER);
  int side = rob % 2;
  buildTrackletHCHeader(header, sector, stack, layer, side, chipclock, format);
}

uint32_t getHCIDFromTrackletHCHeader(const TrackletHCHeader& header)
{
  return header.layer * 2 + header.stack * constants::NLAYER * 2 + header.supermodule * constants::NLAYER * constants::NSTACK * 2 + header.side;
}

// same method alternate input simpler to send a word pointer as const
uint32_t getHCIDFromTrackletHCHeader(const uint32_t& headerword)
{
  TrackletHCHeader header;
  header.word = headerword;
  return header.layer * 2 + header.stack * constants::NLAYER * 2 + header.supermodule * constants::NLAYER * constants::NSTACK * 2 + header.side;
}

uint16_t buildTRDFeeID(int supermodule, int side, int endpoint)
{
  TRDFeeID feeid;
  feeid.supermodule = supermodule;
  feeid.side = side;
  feeid.endpoint = endpoint;
  return feeid.word;
}

void buildTrackletMCMData(TrackletMCMData& trackletword, const uint slope, const uint pos, const uint q0, const uint q1, const uint q2)
{
  trackletword.slope = slope;
  trackletword.pos = pos;
  trackletword.pid = q0 | ((q1 & 0xff) << 8); //q2 sits with a bit of q1 in the header pid word.
  trackletword.checkbit = 1;
}

uint32_t getlinkerrorflag(const HalfCRUHeader& cruhead, const uint32_t link)
{
  // link is the link you are requesting information on, 0-14
  uint32_t errorflag = 0;
  //dealing with word0-2
  errorflag = cruhead.errorflags[link].errorflag;
  return errorflag;
}

uint32_t getlinkdatasize(const HalfCRUHeader& cruhead, const uint32_t link)
{
  // link is the link you are requesting information on, 0-14
  //return number 32 byte blocks for the link 3x64bit ints.
  uint32_t size = 0;
  size = cruhead.datasizes[link].size;
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
};

uint32_t getQFromRaw(const o2::trd::TrackletMCMHeader* header, const o2::trd::TrackletMCMData* data, int pidindex, int trackletindex)
{
  uint32_t pid = 0;
  uint32_t qa, qb;
  //PID VERSION 1
  //frist part of pid is in the TrackletMCMHeader
  switch (trackletindex) {
    case 0:
      qa = header->pid0;
      break;
    case 1:
      qa = header->pid1;
      break;
    case 2:
      qa = header->pid2;
      break;
    default:
      LOG(warn) << " unknown trackletindex to getQFromRaw : " << pidindex;
      break;
  }
  //second part of pid is in the TrackletMCMData
  qb = data->pid;
  switch (pidindex) {
    case 0:
      pid = qa & 0xffc >> 2;
      break;
    case 1:
      pid = ((qa & 0x3) << 5) | (qb >> 6);
      break;
    case 2:
      pid = qb & 0x3f;
      break;
    default:
      LOG(warn) << " unknown pid index of : " << pidindex;
      break;
  }
  //PID VERSION 2
  /*
 switch(pidindex) {
     case 0 : pid=qa&0xffc>>2;break;
     case 1 : pid=((qa&0x3)<<5)|(qb>>6);break;
     case 2 : pid=qb&0x3f;break;
     default : LOG(warn) << " unknown pid index of : " << pidindex;
 }
 */
  return pid;
}

uint32_t setHalfCRUHeader(HalfCRUHeader& cruhead, int crurdhversion, int bunchcrossing, int stopbits, int endpoint, int eventtype, int feeid, int cruid)
{
  cruhead.BunchCrossing = bunchcrossing;
  cruhead.StopBit = stopbits;
  cruhead.EndPoint = endpoint;
  cruhead.EventType = eventtype;
  cruhead.HeaderVersion = crurdhversion;
  //cruhead.FeeID = feeid;
  //cruhead.CRUID = cruid;
  return 0;
}

uint32_t setHalfCRUHeaderLinkData(HalfCRUHeader& cruhead, int link, int size, int errors)
{
  cruhead.datasizes[link].size = size;
  cruhead.errorflags[link].errorflag = errors;
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
  stream << "TrackletHCHeader : Raw:0x" << std::hex << halfchamberheader.word << " "
         << halfchamberheader.format << " ;; " << halfchamberheader.MCLK << " :: "
         << halfchamberheader.one << " :: (" << halfchamberheader.supermodule << ","
         << halfchamberheader.stack << "," << halfchamberheader.layer << ") on side :"
         << halfchamberheader.side << std::endl;
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const TrackletMCMData& tracklet)
{
  // make a pretty output of the tracklet.
  stream << "TrackletMCMData: Raw:0x" << std::hex << tracklet.word << " pos=" << tracklet.pos
         << "::slope=" << tracklet.slope << "::pid=" << tracklet.pid << "::checkbit="
         << tracklet.checkbit << std::endl;
  return stream;
}
void printTrackletMCMData(o2::trd::TrackletMCMData& tracklet)
{
  LOGF(INFO, "TrackletMCMData: Raw:0x%08x pos:%d slope:%d pid:0x%08x checkbit:0x%02x",
       tracklet.word, tracklet.pos, tracklet.slope, tracklet.pid, tracklet.checkbit);
}

void printTrackletMCMHeader(o2::trd::TrackletMCMHeader& mcmhead)
{
  LOG(info) << " about to print mcm raw header";
  LOGF(INFO, "MCMRawHeader: Raw:0x%08x 1:%d padrow: 0x%02x col: 0x%01x pid2 0x%02x pid1: 0x%02x pid0: 0x%02x 1:%d",
       mcmhead.word, mcmhead.onea, mcmhead.padrow, mcmhead.col,
       mcmhead.pid2, mcmhead.pid1, mcmhead.pid0, mcmhead.oneb);
  LOG(info) << " printed mcm raw header";
}

std::ostream& operator<<(std::ostream& stream, const TrackletMCMHeader& mcmhead)
{
  // make a pretty output of the mcm header.
  stream << "TrackletMCMRawHeader: Raw:0x" << std::hex << mcmhead.word << " " << mcmhead.onea << "::"
         << mcmhead.pid2 << ":" << mcmhead.pid1 << ":" << mcmhead.pid0 << "::"
         << mcmhead.oneb << std::endl;
  return stream;
}

void printHalfChamber(o2::trd::TrackletHCHeader& halfchamber)
{
  LOGF(INFO, "TrackletHCHeader: Raw:0x%08x SM : %d stack %d layer %d side : %d MCLK: 0x%0x Format: 0x%0x Always1:0x%0x",
       halfchamber.supermodule, halfchamber.stack, halfchamber.layer, halfchamber.side, halfchamber.MCLK, halfchamber.format, halfchamber.one);
}

void printDigitMCMHeader(o2::trd::DigitMCMHeader& mcmhead)
{
  LOGF(INFO, "DigitMCMRawHeader: Raw:0x%08x res(0xc):0x%02x mcm: 0x%03x rob: 0x%03x eventcount 0x%05x year(>2007?): 0x%02x ",
       mcmhead.word, mcmhead.res, mcmhead.mcm, mcmhead.rob, mcmhead.eventcount,
       mcmhead.yearflag);
}

void dumpHalfChamber(o2::trd::TrackletHCHeader const& halfchamber)
{
  LOGF(INFO, "HalfChamber : 0x%08x", halfchamber.word);
}

void printHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::array<uint32_t, 15> sizes;
  std::array<uint32_t, 15> errorflags;
  getlinkdatasizes(halfcru, sizes);
  getlinkerrorflags(halfcru, errorflags);
  LOGF(INFO, "V:%d BC:%d SB:%d EType:%d", halfcru.HeaderVersion, halfcru.BunchCrossing,
       halfcru.StopBit, halfcru.EventType);
  for (int i = 0; i < 15; i++) {
    LOGF(INFO, "Link %d size: %ul eflag: 0x%02x", i, sizes[i], errorflags[i]);
  }
}

void dumpHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::array<uint64_t, 8> raw{};
  memcpy(&raw[0], &halfcru, sizeof(halfcru));
  for (int i = 0; i < 2; i++) {
    int index = 4 * i;
    LOGF(INFO, "[1/2CRUHeader %d] 0x%08x 0x%08x 0x%08x 0x%08x", i, raw[index + 3], raw[index + 2],
         raw[index + 1], raw[index + 0]);
  }
}

void clearHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  halfcru.word0 = 0;
  halfcru.word12[0] = 0;
  halfcru.word12[1] = 0;
  halfcru.word3 = 0;
  halfcru.word47[0] = 0;
  halfcru.word47[1] = 0;
  halfcru.word47[2] = 0;
  halfcru.word47[3] = 0;
}
std::ostream& operator<<(std::ostream& stream, const HalfCRUHeader& halfcru)
{ // make a pretty output of the header.
  stream << std::hex;
  stream << "EventType : " << halfcru.EventType << std::endl;
  stream << "StopBit : " << halfcru.StopBit << std::endl;
  stream << "BunchCrossing : " << halfcru.BunchCrossing << std::endl;
  stream << "HeaderVersion : " << halfcru.HeaderVersion << std::endl;
  stream << "link  sizes : ";
  for (int link = 0; link < 15; link++) {
    stream << link << ":" << std::hex << std::setw(4) << getlinkdatasize(halfcru, link) << ",";
  }
  stream << std::endl;
  stream << "link  errorflags : ";
  for (int link = 0; link < 15; link++) {
    stream << link << ":" << std::hex << std::setw(2) << getlinkerrorflag(halfcru, link) << ",";
  }
  stream << std::endl;
  stream << "0x" << std::hex << halfcru.word0 << " 0x" << halfcru.word12[0] << " 0x" << halfcru.word12[1] << " 0x" << halfcru.word3 << " 0x" << halfcru.word47[0] << " 0x" << halfcru.word47[1] << " 0x" << halfcru.word47[2] << " 0x" << halfcru.word47[3] << std::endl;
  return stream;
}

bool trackletMCMHeaderSanityCheck(o2::trd::TrackletMCMHeader& header)
{
  // a bit limited to what we can check.
  bool goodheader = true;
  if (header.onea != 1) {
    goodheader = false;
  }
  if (header.oneb != 1) {
    goodheader = false;
  }
  // if we have 3rd tracklet (pid2!=0) then we must have all the others as well.
  if ((header.pid2 != 0) && (header.pid1 == 0 || header.pid0 == 0)) {
    goodheader = false;
  }
  // sim for 2 tracklets.
  if ((header.pid1 != 0) && (header.pid0 == 0)) {
    goodheader = false;
  }

  return goodheader;
}

bool trackletHCHeaderSanityCheck(o2::trd::TrackletHCHeader& header)
{
  bool goodheader = true;
  if (header.one != 1) {
    goodheader = false;
  }
  if (header.supermodule > 17) {
    goodheader = false;
  }
  //if(header.format != )  only certain format versions are permitted come back an fill in if needed.
  return goodheader;
}

bool digitMCMHeaderSanityCheck(o2::trd::DigitMCMHeader* header)
{
  // a bit limited to what we can check.
  bool goodheader = true;
  if (header->res != 0xc) {
    goodheader = false;
  }
  if (header->yearflag == 0) { //we only have data after 2007 now in run3.
    goodheader = false;
  }

  return goodheader;
}

void printDigitHCHeader(o2::trd::DigitHCHeader& header)
{
  LOGF(INFO, "Digit HalfChamber Header\n Raw:0x%08x 0x%08x reserve:%01x side:%01x stack:0x%02x layer:0x%02x supermod:0x%02x numberHCW:0x%02x minor:0x%03x major:0x%03x version:0x%01x reserve:0x%02x pretriggercount=0x%02x pretriggerphase=0x%02x bunchxing:0x%05x number of timebins : 0x%03x\n",
       header.word0, header.word1, header.res0, header.side, header.stack, header.layer, header.supermodule, header.numberHCW, header.minor, header.major, header.version, header.res1, header.ptrigcount, header.ptrigphase, header.bunchcrossing, header.numtimebins);
}

DigitMCMADCMask buildBlankADCMask()
{
  //set the default values for the mask.
  DigitMCMADCMask mask;
  mask.c = 0x1f;
  mask.n = 0x3;
  mask.j = 0xc;
  // actual mask will beset somewhere else, the above values are *always* that.
  return mask;
}

int getNumberofTracklets(o2::trd::TrackletMCMHeader& header)
{
  int headertrackletcount = 0;
  if (header.pid0 == 0xff) {
    LOG(warn) << "! we have an MCM Tracklet Header with the first pid zero implying no tracklets! header to follow:";
    LOG(warn) << header;
    return 0;
  } else {
    if (header.pid2 != 0xff) {
      // 3 tracklets
      headertrackletcount = 3;
      if (header.pid1 == 0xff || header.pid0 == 0xff) {
        LOG(warn) << "! we have an MCM Tracklet Header with the pid2!=0 but pid1 or pid0 is ! header to follow:";
        LOG(warn) << header;
      }
    } else {
      if (header.pid1 != 0xff) {
        // 2 tracklets
        headertrackletcount = 2;
        if (header.pid0 == 0xff) {
          LOG(warn) << "! we have an MCM Tracklet Header with the pid1!=0 but pid0 is ! header to follow:";
          LOG(warn) << header;
        }
      } else {
        if (header.pid0 != 0xff) {
          // 1 tracklet
          headertrackletcount = 1;
        } else {
          LOG(warn) << "! we have an MCM Tracklet Header with the pidx==0xff  we should not be here due to first if statement though! header to follow:";
          LOG(warn) << header;
        }
      }
    }
  }
  return headertrackletcount;
}

void setNumberOfTrackletsInHeader(o2::trd::TrackletMCMHeader& header, int numberoftracklets)
{

  //header.word |= 0xff<< (1+numberoftracklets*8);
  switch (numberoftracklets) {
    case 0:
      LOG(error) << " tracklet header but no tracklets???";
      header.pid0 = 0xff;
      header.pid1 = 0xff;
      header.pid2 = 0xff;
      break;
    case 1:
      header.pid1 = 0xff;
      header.pid2 = 0xff;
      break;
    case 2:
      header.pid2 = 0xff;
      break;
    case 3:
      break;
    default:
      LOG(error) << "we have more than 3 tracklets for an mcm. This should never happen: tracklet count=" << numberoftracklets;
  }
  //  LOG(info) << " setting header tracklet number " << numberoftracklets << " header pid0 pid1 pid2 :" << std::hex << header.word << " " << header.pid0 << " " << header.pid1 << " " << header.pid2;
}

int nextmcmadc(unsigned int& bp, int channel)
{
  //given a bitpattern (adcmask) find next channel with in the mask starting from the current channel;
  while ((bp & (1 << channel)) == 0) {
    channel++;
    if (channel == 21) {
      break;
    }
  }
  bp &= ~(1UL << (channel));
  return channel; // zero based
}

} // namespace trd
} // namespace o2
