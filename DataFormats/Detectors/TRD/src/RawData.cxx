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

#include <iomanip>
#include <iostream>
#include <bitset>
#include "fairlogger/Logger.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/LinkRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/HelperMethods.h"

using namespace o2::trd::constants;

namespace o2
{

namespace trd
{

bool LinkToHCIDMapping::isOK() const
{
  for (int linkIn = 0; linkIn < MAXHALFCHAMBER; ++linkIn) {
    int hcid = getHCID(linkIn);
    if (linkIn != getLink(hcid)) {
      return false;
    }
  }
  return true;
}

// linkA and linkB refer to the global ORI index and not to the half-chamber ID
void LinkToHCIDMapping::swapLinks(int linkA, int linkB)
{
  int hcidA = HelperMethods::getHCIDFromLinkID(linkA);
  int hcidB = HelperMethods::getHCIDFromLinkID(linkB);
  linkIDToHCID.erase(linkA);
  linkIDToHCID.insert({linkA, hcidB});
  linkIDToHCID.erase(linkB);
  linkIDToHCID.insert({linkB, hcidA});
  hcIDToLinkID.erase(hcidA);
  hcIDToLinkID.insert({hcidA, linkB});
  hcIDToLinkID.erase(hcidB);
  hcIDToLinkID.insert({hcidB, linkA});
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
         << (int)halfchamberheader.format << " ;; " << (int)halfchamberheader.MCLK << " :: "
         << halfchamberheader.one << " :: (" << (int)halfchamberheader.supermodule << ","
         << (int)halfchamberheader.stack << "," << (int)halfchamberheader.layer << ") on side :"
         << (int)halfchamberheader.side << std::endl;
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

std::ostream& operator<<(std::ostream& stream, const TrackletMCMHeader& mcmhead)
{
  // make a pretty output of the mcm header.
  stream << "TrackletMCMRawHeader: Raw:0x" << std::hex << mcmhead.word << " " << mcmhead.onea << "::"
         << mcmhead.pid2 << ":" << mcmhead.pid1 << ":" << mcmhead.pid0 << "::"
         << mcmhead.oneb << std::endl;
  return stream;
}

//functions updated/checked/new for new raw reader. above methods left for cross checking what changes have occured.
// construct a tracklet half chamber header according to the tdp and assembler found in

//HalfCRUHeader first :
//
//this only sets the first 64 bit word of the half cru header.
uint32_t setHalfCRUHeaderFirstWord(HalfCRUHeader& cruhead, int crurdhversion, int bunchcrossing, int stopbits, int endpoint, int eventtype, int feeid, int cruid)
{
  cruhead.word0 = 0;
  cruhead.BunchCrossing = bunchcrossing;
  cruhead.StopBit = stopbits;
  cruhead.EndPoint = endpoint;
  cruhead.EventType = eventtype;
  cruhead.HeaderVersion = crurdhversion;
  //This is undefiend behaviour if the rest of cruhead has not been set to zero ...
  //TODO check where this is called from
  return 0;
}

void setHalfCRUHeaderLinkSizeAndFlags(HalfCRUHeader& cruhead, int link, int size, int errors)
{
  cruhead.datasizes[link].size = size;
  cruhead.errorflags[link].errorflag = errors;
}

uint32_t getHalfCRULinkErrorFlag(const HalfCRUHeader& cruhead, const uint32_t link)
{
  // link is the link you are requesting information on, 0-14
  uint32_t errorflag = 0;
  //dealing with word0-2
  errorflag = cruhead.errorflags[link].errorflag;
  return errorflag;
}

uint32_t getHalfCRULinkDataSize(const HalfCRUHeader& cruhead, const uint32_t link)
{
  // link is the link you are requesting information on, 0-14
  //return number 32 byte blocks for the link 3x64bit ints.
  return cruhead.datasizes[link].size;
}

void getHalfCRULinkErrorFlags(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linkerrorflags)
{
  // retrieve all the link error flags for this half cru
  for (uint32_t link = 0; link < 15; link++) {
    linkerrorflags[link] = getHalfCRULinkErrorFlag(cruheader, link);
  }
}

void getHalfCRULinkDataSizes(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linksizes)
{
  // retrieve all the link error flags for this half cru
  for (uint32_t link = 0; link < 15; link++) {
    linksizes[link] = getHalfCRULinkDataSize(cruheader, link);
  }
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
    stream << link << ":" << std::hex << std::setw(4) << getHalfCRULinkDataSize(halfcru, link) << ",";
  }
  stream << std::endl;
  stream << "link  errorflags : ";
  for (int link = 0; link < 15; link++) {
    stream << link << ":" << std::hex << std::setw(2) << getHalfCRULinkErrorFlag(halfcru, link) << ",";
  }
  stream << std::endl;
  stream << "0x" << std::hex << halfcru.word0 << " 0x" << halfcru.word12[0] << " 0x" << halfcru.word12[1] << " 0x" << halfcru.word3 << " 0x" << halfcru.word47[0] << " 0x" << halfcru.word47[1] << " 0x" << halfcru.word47[2] << " 0x" << halfcru.word47[3] << std::endl;
  return stream;
}

//Tracklet HC Header

void constructTrackletHCHeader(TrackletHCHeader& header, int hcid, int chipclock, int format)
{
  int detector = hcid / 2;
  int sector = HelperMethods::getSector(detector);
  int stack = HelperMethods::getStack(detector);
  int layer = HelperMethods::getLayer(detector);
  int side = hcid % 2;
  header.word = 0;
  header.format = format;
  header.supermodule = ~sector;
  header.stack = ~stack;
  header.layer = ~layer;
  header.side = ~side;
  header.MCLK = chipclock;
  header.one = 1;
}

uint16_t constructTRDFeeID(int supermodule, int side, int endpoint)
{
  TRDFeeID feeid;
  feeid.word = 0;
  feeid.supermodule = supermodule;
  feeid.side = side;
  feeid.endpoint = endpoint;
  feeid.unused1 = 0;
  feeid.unused2 = 0;
  return feeid.word;
}

DigitMCMADCMask constructBlankADCMask()
{
  //set the default values for the mask.
  DigitMCMADCMask mask;
  mask.word = 0;
  mask.c = 0x1f;
  mask.n = 0x1;
  mask.j = 0xc;
  mask.adcmask = 0;
  // actual mask will beset somewhere else, the above values are *always* that.
  return mask;
}

void printTrackletHCHeader(const o2::trd::TrackletHCHeader& halfchamber)
{
  LOGF(info, "TrackletHCHeader: Raw:0x%08x SM : %d stack %d layer %d side : %d MCLK: 0x%0x Format: 0x%0x Always1:0x%0x",
       halfchamber.word, (int)(~halfchamber.supermodule) & 0x1f, (int)(~halfchamber.stack) & 0x7, (int)(~halfchamber.layer) & 0x7, (int)(~halfchamber.side) & 0x1, (int)halfchamber.MCLK, (int)halfchamber.format, (int)halfchamber.one);
}

void printTrackletMCMData(const o2::trd::TrackletMCMData& tracklet)
{
  LOGF(info, "TrackletMCMData: Raw:0x%08x pos:%d slope:%d pid:0x%03x checkbit:0x%02x",
       tracklet.word, tracklet.pos, tracklet.slope, tracklet.pid, tracklet.checkbit);
}
void printTrackletMCMHeader(const o2::trd::TrackletMCMHeader& mcmhead)
{
  LOGF(info, "MCMRawHeader: Raw:0x%08x 1:%d padrow: 0x%02x col: 0x%01x pid2 0x%02x pid1: 0x%02x pid0: 0x%02x 1:%d",
       mcmhead.word, mcmhead.onea, mcmhead.padrow, mcmhead.col,
       mcmhead.pid2, mcmhead.pid1, mcmhead.pid0, mcmhead.oneb);
}

void printHalfCRUHeader(const o2::trd::HalfCRUHeader& halfcru)
{
  std::array<uint32_t, 15> sizes;
  std::array<uint32_t, 15> errorflags;
  getHalfCRULinkDataSizes(halfcru, sizes);
  getHalfCRULinkErrorFlags(halfcru, errorflags);
  LOGF(info, "V:%d BC:%d SB:%d EType:%d", halfcru.HeaderVersion, halfcru.BunchCrossing,
       halfcru.StopBit, halfcru.EventType);
  for (int i = 0; i < 15; i++) {
    LOGF(info, "Link %d size: %lu eflag: 0x%02x", i, sizes[i], errorflags[i]);
  }
  LOG(info) << "Raw: " << std::hex << halfcru.word0 << " " << halfcru.word12[0] << " " << halfcru.word12[1] << " " << halfcru.word3 << " " << halfcru.word47[0] << " " << halfcru.word47[1] << " " << halfcru.word47[2] << " " << halfcru.word47[3];
}

void clearHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::memset(&halfcru, 0, sizeof(o2::trd::HalfCRUHeader));
}

bool halfCRUHeaderSanityCheck(const o2::trd::HalfCRUHeader& header)
{
  // check the sizes for less than max value
  // check the errors for either < 0x3, for now (may 2022) there is only no error, 1, or 2.
  //
  for (int link = 0; link < 15; ++link) {
    if (header.datasizes[link].size > MAXDATAPERLINK256) {
      return false;
    }
    if (header.errorflags[link].errorflag > MAXCRUERRORVALUE) {
      return false;
    }
  }
  if (header.EndPoint > 1) {
    // end point can only be zero or 1, for each of the 2 pci end points in the cru
    return false;
  }
  return true;
}

bool sanityCheckTrackletHCHeader(const o2::trd::TrackletHCHeader& header)
{
  if (header.one != 1) {
    return false;
  }
  if (((~header.supermodule) & 0x1f) >= NSECTOR) {
    return false;
  }
  if (((~header.stack) & 0x7) >= NSTACK) {
    return false;
  }
  if (((~header.layer) & 0x7) >= NLAYER) {
    return false;
  }
  return true;
}

bool sanityCheckTrackletMCMHeader(const o2::trd::TrackletMCMHeader& header)
{
  // a bit limited to what we can check.
  if (header.onea != 1) {
    return false;
  }
  if (header.oneb != 1) {
    return false;
  }
  return true;
}

bool sanityCheckDigitMCMHeader(const o2::trd::DigitMCMHeader& header)
{
  // a bit limited to what we can check.
  if (header.res != 0xc) {
    return false;
  }
  if (header.yearflag == 0) { // we only have data after 2007 now in run3.
    return false;
  }
  return true;
}

bool sanityCheckDigitMCMADCMask(const o2::trd::DigitMCMADCMask& mask)
{
  if (mask.n != 0x1) {
    return false;
  }
  if (mask.j != 0xc) {
    return false;
  }
  unsigned int counter = (~mask.c) & 0x1f;
  std::bitset<NADCMCM> headerMask(mask.adcmask);
  return (counter == headerMask.count());
}

void incrementADCMask(DigitMCMADCMask& mask, int channel)
{
  mask.adcmask |= 1UL << channel;
  int bitcount = (~mask.c) & 0x1f;
  bitcount++;
  mask.c = ~((bitcount)&0x1f);
}

void printDigitMCMHeader(const o2::trd::DigitMCMHeader& digitmcmhead)
{
  LOGF(info, "DigitMCMHeader: Raw:0x%08x, res: 0x%02x mcm: 0x%x rob: 0x%x eventcount 0x%05x year(>2007?): 0x%x ",
       digitmcmhead.word, digitmcmhead.res, digitmcmhead.mcm, digitmcmhead.rob, digitmcmhead.eventcount,
       digitmcmhead.yearflag);
}

void printDigitMCMData(const o2::trd::DigitMCMData& digitmcmdata)
{
  LOGF(info, "DigitMCMRawData: Raw:0x%08x res:0x%x x: 0x%03x y: 0x%03x z 0x%03x ",
       digitmcmdata.word, digitmcmdata.f, digitmcmdata.x, digitmcmdata.y, digitmcmdata.z);
}
void printDigitMCMADCMask(const o2::trd::DigitMCMADCMask& digitmcmadcmask)
{
  LOGF(info, "DigitMCMADCMask: Raw:0x%08x j(0xc):0x%01x mask: 0x%05x count: 0x%02x n(0x1) 0x%01x ",
       digitmcmadcmask.word, digitmcmadcmask.j, digitmcmadcmask.adcmask, digitmcmadcmask.c, digitmcmadcmask.n);
}

int getDigitHCHeaderWordType(uint32_t word)
{
  // all digit HC headers end with the bit pattern 01
  // the bits 5..2 are used to distinguish the different header types
  // the bit patterns 0b11001 and 0b110101 are outside the valid range
  // for the header type 1
  if ((word & 0x3f) == 0b110001) {
    return 2;
  }
  if ((word & 0x3f) == 0b110101) {
    return 3;
  }
  if ((word & 0x3) == 0b01) {
    return 1;
  }
  return -1;
}
// this method just exists to make the printDigitHCHeader simpler to read.
void printDigitHCHeaders(o2::trd::DigitHCHeader& header, uint32_t headers[3], int index, int offset, bool good)
{
  switch (index) {
    case -1:
      LOGF(info, "Digit HalfChamber Header: Raw:0x%08x reserve:0x%01x side:0x%01x stack:0x%02x layer:0x%02x supermod:0x%02x numberHCW:0x%02x minor:0x%03x major:0x%03x version(>2007):0x%01x",
           header.word, header.res, header.side, header.stack, header.layer, header.supermodule,
           header.numberHCW, header.minor, header.major, header.version);
      break;
    case 0:
      o2::trd::DigitHCHeader1 header1;
      header1.word = headers[offset];
      LOGF(info, "%s Digit HalfChamber Header1 Raw:0x%08x reserve:0x%02x pretriggercount=0x%02x pretriggerphase=0x%02x bunchxing:0x%05x number of timebins : 0x%03x", (good) ? "" : "*Corrupt*", header1.word, header1.res, header1.ptrigcount, header1.ptrigphase, header1.bunchcrossing, header1.numtimebins);
      break;
    case 1:
      o2::trd::DigitHCHeader2 header2;
      header2.word = headers[offset];
      LOGF(info, "%s Digit HalfChamber Header2 Raw:0x%08x reserve:0x%08x PedestalFilter:0x%01x GainFilter:0x%01x TailFilter:0x%01x CrosstalkFilter:0x%01x Non-linFilter:0x%01x RawDataBypassFilter:0x%01x DigitFilterCommonAdditive:0x%02x ", (good) ? "" : "*Corrupt*", header2.word, header2.res, header2.dfilter, header2.rfilter, header2.nlfilter, header2.xtfilter, header2.tfilter, header2.gfilter, header2.pfilter);
      break;
    case 2:
      o2::trd::DigitHCHeader3 header3;
      header3.word = headers[offset];
      LOGF(info, "%s Digit HalfChamber Header3: Raw:0x%08x reserve:0x%08x readout program revision:0x%08x assembler program version:0x%01x", (good) ? "" : "*Corrupt*", header3.word, header3.res, header3.svnrver, header3.svnver);
      break;
  }
}

void printDigitHCHeader(o2::trd::DigitHCHeader& header, uint32_t headers[3])
{
  printDigitHCHeaders(header, headers, -1, 0, true);
  int index;
  //for the currently 3 implemented other header words, they can come in any order, and are identified by their reserved portion
  for (int countheaderwords = 0; countheaderwords < header.numberHCW; ++countheaderwords) {
    switch (getDigitHCHeaderWordType(headers[countheaderwords])) {
      case 1:
        DigitHCHeader1 header1;
        header1.word = headers[countheaderwords];
        index = 0;
        if (header1.res != 0x1) {
          printDigitHCHeaders(header, headers, index, countheaderwords, false);
        } else {
          printDigitHCHeaders(header, headers, index, countheaderwords, true);
        }
        break;
      case 2:
        DigitHCHeader2 header2;
        header2.word = headers[countheaderwords];
        index = 1;
        if (header2.res != 0b110001) {
          printDigitHCHeaders(header, headers, index, countheaderwords, false);
        } else {
          printDigitHCHeaders(header, headers, index, countheaderwords, true);
        }
        break;
      case 3:
        DigitHCHeader3 header3;
        header3.word = headers[countheaderwords];
        index = 2;
        if (header3.res != 0b110101) {
          printDigitHCHeaders(header, headers, index, countheaderwords, false);
        } else {
          printDigitHCHeaders(header, headers, index, countheaderwords, true);
        }
        break;
    }
  }
}

} // namespace trd
} // namespace o2
