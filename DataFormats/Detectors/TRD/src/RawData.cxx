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
#include <iomanip>
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/LinkRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/Tracklet64.h"

namespace o2
{

namespace trd
{

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

std::ostream& operator<<(std::ostream& stream, const TrackletMCMHeader& mcmhead)
{
  // make a pretty output of the mcm header.
  stream << "TrackletMCMRawHeader: Raw:0x" << std::hex << mcmhead.word << " " << mcmhead.onea << "::"
         << mcmhead.pid2 << ":" << mcmhead.pid1 << ":" << mcmhead.pid0 << "::"
         << mcmhead.oneb << std::endl;
  return stream;
}

void dumpHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::array<uint64_t, 8> raw{};
  memcpy(&raw[0], &halfcru, sizeof(halfcru));
  for (int i = 0; i < 2; i++) {
    int index = 4 * i;
    LOGF(info, "[1/2CRUHeader %d] 0x%08x 0x%08x 0x%08x 0x%08x", i, raw[index + 3], raw[index + 2],
         raw[index + 1], raw[index + 0]);
  }
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

void constructTrackletHCHeader(TrackletHCHeader& header, int sector, int stack, int layer, int side, int chipclock, int format)
{
  header.word = 0;

  header.format = format;
  header.supermodule = ~sector;
  header.stack = ~stack;
  header.layer = ~layer;
  header.side = ~side;
  header.MCLK = chipclock;
  header.one = 0;
}

void constructTrackletHCHeaderd(TrackletHCHeader& header, int detector, int rob, int chipclock, int format)
{
  int sector = (detector % (constants::NLAYER * constants::NSTACK));
  int stack = (detector % constants::NLAYER);
  int layer = ((detector % (constants::NLAYER * constants::NSTACK)) / constants::NLAYER);
  int side = rob % 2;
  constructTrackletHCHeader(header, sector, stack, layer, side, chipclock, format);
}

uint32_t getHCIDFromTrackletHCHeader(const TrackletHCHeader& header)
{
  return header.layer * 2 + header.stack * constants::NLAYER * 2 + header.supermodule * constants::NLAYER * constants::NSTACK * 2 + header.side;
}

uint32_t getHCIDFromTrackletHCHeader(const uint32_t& headerword)
{
  TrackletHCHeader header;
  header.word = headerword;
  return header.layer * 2 + header.stack * constants::NLAYER * 2 + header.supermodule * constants::NLAYER * constants::NSTACK * 2 + header.side;
}

uint32_t getChargeFromRawHeaders(const o2::trd::TrackletHCHeader& hcheader, const o2::trd::TrackletMCMHeader* header, const std::array<o2::trd::TrackletMCMData, 3>& data, int pidindex, int trackletindex)
{
  uint32_t pid = 0;
  uint32_t highPID, lowPID; // highPID holds the 8 bits from the mcmheader, and lowPID holds the 12 bits from mcmdata
  uint32_t datatype = (hcheader.format) >> 2;
  switch (datatype) {
    case 0: //Cosmic
            // LOG(warn) << "This is a problem cosmic format tracklets ";
      //break;
    case 1: //TPT
      //LOG(warn) << "This is a problem  TPT format tracklets ";
      //break;
    case 2: //DIS
      //LOG(warn) << "This is a problem  DIS format tracklets ";
      //break;
      LOG(warn) << "This could be a problem  format tracklets is 0x" << std::hex << datatype << " thcheader:0x" << std::hex << hcheader.word << " tmcmheader:0x" << header->word;
    case 3:
      //PID VERSION 1
      //PID is 20 bits, 8 bits in mcmheader and 12 bits in mcmdata word
      //frist part of pid (highPID) is in the TrackletMCMHeader
      //highPID is 7 bits Q2, 1 bit Q1 OR ... 2 bit offset, 6 bits Q2.
      switch (trackletindex) {
        case 0:
          highPID = header->pid0;
          break;
        case 1:
          highPID = header->pid1;
          break;
        case 2:
          highPID = header->pid2;
          break;
        default:
          LOG(warn) << " unknown trackletindex of " << trackletindex << " to " << __func__ << " : " << pidindex;
          break;
      }
      lowPID = data[trackletindex].pid;
      //lowPID is 6 bits Q0 and 6 bits of Q1
      int pidword = (highPID << 12) + lowPID;       // the entire original 20 bit pid in the trap chips
      int dynamicq = hcheader.format & 0x1;         // last bit of format (lsb) defines the version of tracklet charge calculation
      uint32_t pidoffset = ((pidword >> 18) & 0x3); //<<6; // used for dynamic ranged charge windows, may or may not be used below.
      //pidword is here to make this code more readible and less error prone.
      switch (pidindex) {
        case 2: //Q2
          if (!dynamicq) {
            pid = (pidword >> 13) & 0x7f; // 7 bits at the top of all of pid (MSB)
          } else {
            pid = (pidword >> 12) & 0x3f; // 6 bits of Q2 and a shift
            pid |= pidoffset << 6;
            // LOG(info) << "Q2 pid : " << std::hex << pid << " pidoffset: "  << pidoffset;
          }
          break;
        case 1: //Q1
          if (!dynamicq) {
            pid = (pidword >> 6) & 0x7f; // 7 bits Q1 above the 6 bits of Q0
          } else {
            pid = (pidword >> 6) & 0x3f; // 6 bits of Q1 and a shift
            pid |= pidoffset << 6;
            //LOG(info) << "Q1 pid : " << std::hex << pid << " pidoffset: "  << pidoffset;;
          }
          break;
        case 0: //Q0
          if (!dynamicq) {
            pid = pidword & 0x3f; // 6 least significant bits
          } else {
            pid = pidword & 0x3f; // 6 bits of Q0
            pid |= pidoffset << 6;
            // LOG(info) << "Q0 pid : " << std::hex << pid << " pidoffset: "  << pidoffset;
          }
          break;
        default:
          LOG(warn) << " unknown pid index of : " << pidindex;
          break;
      }
  } // end of case of various formats.
  return pid;
}

//Tracklet MCM Header

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

void constructTrackletMCMData(TrackletMCMData& trackletword, const int format, const uint slope, const uint pos, const uint q0, const uint q1, const uint q2)
{
  trackletword.word = 0;
  // create a tracklet word as it would be sent from the FEE
  // slope and position have the 8-th bit flipped each
  trackletword.word = 0;
  trackletword.slope = slope ^ 0x80;
  trackletword.pos = pos ^ 0x80;
  trackletword.checkbit = 0;
  if (format & 0x1) {
    //length of q1 and q2 are 6 with a 2 bit offset.
    //What happens when q2 and q1 dont share the same offset (upper 2 bits ?) ?
    LOG(error) << "This tracklet format has not been tested yet";
    trackletword.pid = (q0 & 0x3f) & ((q1 & 0x1) << 6);
    //TODO check the 2 bit offset is still valid ... ??
  } else {
    trackletword.pid = (q0 & 0x3f) & ((q1 & 0x1) << 6);
  }
  //q2 sits with upper 2 bits of q1 in the header pid word, hence the 0x1f so 5 bits are used here.
}

void constructTrackletMCMData(TrackletMCMData& trackletword, const Tracklet64& tracklet)
{
  constructTrackletMCMData(trackletword, tracklet.getFormat(), tracklet.getSlope(), tracklet.getPosition(), tracklet.getQ0(), tracklet.getQ1(), tracklet.getQ2());
}

DigitMCMADCMask constructBlankADCMask()
{
  //set the default values for the mask.
  DigitMCMADCMask mask;
  mask.word = 0;
  mask.c = 0x1f;
  mask.n = 0x1;
  mask.j = 0xc;
  // actual mask will beset somewhere else, the above values are *always* that.
  return mask;
}

void printTrackletHCHeader(o2::trd::TrackletHCHeader& halfchamber)
{
  LOGF(info, "TrackletHCHeader: Raw:0x%08x SM : %d stack %d layer %d side : %d MCLK: 0x%0x Format: 0x%0x Always1:0x%0x",
       halfchamber.supermodule, halfchamber.stack, halfchamber.layer, halfchamber.side, halfchamber.MCLK, halfchamber.format, halfchamber.one);
}

void printTrackletMCMData(o2::trd::TrackletMCMData& tracklet)
{
  LOGF(info, "TrackletMCMData: Raw:0x%08x pos:%d slope:%d pid:0x%08x checkbit:0x%02x",
       tracklet.word, tracklet.pos, tracklet.slope, tracklet.pid, tracklet.checkbit);
}
void printTrackletMCMHeader(o2::trd::TrackletMCMHeader& mcmhead)
{
  LOG(info) << " about to print mcm raw header";
  LOGF(info, "MCMRawHeader: Raw:0x%08x 1:%d padrow: 0x%02x col: 0x%01x pid2 0x%02x pid1: 0x%02x pid0: 0x%02x 1:%d",
       mcmhead.word, mcmhead.onea, mcmhead.padrow, mcmhead.col,
       mcmhead.pid2, mcmhead.pid1, mcmhead.pid0, mcmhead.oneb);
  LOG(info) << " printed mcm raw header";
}

void printHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::array<uint32_t, 15> sizes;
  std::array<uint32_t, 15> errorflags;
  getHalfCRULinkDataSizes(halfcru, sizes);
  getHalfCRULinkErrorFlags(halfcru, errorflags);
  LOGF(info, "V:%d BC:%d SB:%d EType:%d", halfcru.HeaderVersion, halfcru.BunchCrossing,
       halfcru.StopBit, halfcru.EventType);
  for (int i = 0; i < 15; i++) {
    LOGF(info, "Link %d size: %ul eflag: 0x%02x", i, sizes[i], errorflags[i]);
  }
  LOG(info) << "Raw: " << std::hex << halfcru.word0 << " " << halfcru.word12[0] << " " << halfcru.word12[1] << " " << halfcru.word3 << " " << halfcru.word47[0] << " " << halfcru.word47[1] << " " << halfcru.word47[2] << " " << halfcru.word47[3];
}

void clearHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::memset(&halfcru, 0, sizeof(o2::trd::HalfCRUHeader));
}

bool sanityCheckTrackletMCMData(o2::trd::TrackletMCMData& data)
{
  bool gooddata = true;
  if (data.checkbit == 0) {
    gooddata = false;
  }
  /*
  if(data.slope< o2::trd::constants::MinTrackletSlope || data.slope> o2::trd::MaxTrackletSlope){
    gooddata = false;
  }
  if(data.pos<  o2::trd::constants::MinTrackletPos|| data.pos> o2::trd::constants::MinTrackletPos){
    gooddata = false;
  }
  if(data.pid){

  }
  */
  return gooddata;
}

bool halfCRUHeaderSanityCheck(o2::trd::HalfCRUHeader& header, std::array<uint32_t, 15>& lengths, std::array<uint32_t, 15>& eflags)
{
  // check the sizes for less than max value
  // check the errors for either < 0x3, for now (may 2022) there is only no error, 1, or 2.
  //
  bool goodheader = true;
  for (int lengthindex = 0; lengthindex < 15; ++lengthindex) {
    if (lengths[lengthindex] > o2::trd::constants::MAXDATAPERLINK256) {
      // something has gone insane.
      //LOG(info) << "AAA dumping half cru as : half cru link length > max possible! : " << lengths[lengthindex] << " ?? " << o2::trd::constants::MAXDATAPERLINK256;
      goodheader = false;
    }
  }
  for (int eflagindex = 0; eflagindex < 15; ++eflagindex) {
    if (eflags[eflagindex] > o2::trd::constants::MAXCRUERRORVALUE) {
      // something has gone insane.
      // LOG(info) << "AAA dumping half cru as : half cru link eflag > max possible! : " << std::hex << eflags[eflagindex] << " ?? " << o2::trd::constants::MAXCRUERRORVALUE;
      goodheader = false;
    }
    if (header.EndPoint > 1) {
      // end point can only be zero or 1, for ach of the 2 pci end points in the cru
      goodheader = false;
    }
    //LOG(info) << "Header sanity check is : " << goodheader;
    goodheader = false;
  }

  return goodheader;
}

bool sanityCheckTrackletMCMHeader(o2::trd::TrackletMCMHeader* header)
{
  // a bit limited to what we can check.
  bool goodheader = true;
  if (header->onea != 1) {
    goodheader = false;
  }
  if (header->oneb != 1) {
    goodheader = false;
  }
  // if we have 3rd tracklet (pid2!=0) then we must have all the others as well.
  if ((header->pid2 != 0xff) && (header->pid1 == 0xff || header->pid0 == 0xff)) {
    goodheader = false;
  }
  // sim for 2 tracklets.
  if ((header->pid1 != 0xff) && (header->pid0 == 0xff)) {
    goodheader = false;
  }
  return goodheader;
}

bool sanityCheckTrackletHCHeader(o2::trd::TrackletHCHeader& header)
{
  bool goodheader = true;
  if (header.one != -1) {
    goodheader = false;
  }
  if ((~header.supermodule) > 17) {
    goodheader = false;
  }
  if ((~header.layer) > 6) {
    goodheader = false;
  }
  if ((~header.stack) > 5) {
    goodheader = false;
  }
  if (header.one != 0) {
    goodheader = false;
  }
  int trackletmode = (header.format >> 2) & 0x3;
  bool dynamicqrange = false;
  switch (trackletmode) {
    case 0: //Cosmic
      break;
    case 1: //TPT
      break;
    case 2: //DIS
      break;
    case 3: //3Q Mode
      if ((header.format & 0x1) == 0x1) {
        dynamicqrange = true;
      }
      //LOG(info) << "Tracklet sector:layer:stack " << ((~header.supermodule)&0x1f) <<":" << ((~header.layer)&0x7) << ":" << ((~header.stack)&0x7) << " Tracklet format is : " << std::hex << header.format << " last bit : " << (header.format&0x1) << " trackletmode : " << trackletmode;
      break;
  }
  return goodheader;
}

bool sanityCheckDigitMCMHeader(o2::trd::DigitMCMHeader* header)
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

bool sanityCheckDigitMCMADCMask(o2::trd::DigitMCMADCMask& mask, int numberofbitsset)
{
  bool goodadcmask = true;
  uint32_t count = (unsigned int)mask.c;
  count = (~count) & 0x1f;
  if (count != numberofbitsset) {
    goodadcmask=false;
    LOG(warn) << "***DigitMCMADCMask bad bit count maskcount:" << ~mask.c << "::" << mask.c << " bitscounting:" << numberofbitsset << " bp: 0x" << std::hex << mask.adcmask;
  }
  if (mask.n != 0x1) {
    goodadcmask = false;
  }
  if (mask.j != 0xc) {
    goodadcmask = false;
  }
  return goodadcmask;
}

bool sanityCheckDigitMCMWord(o2::trd::DigitMCMData* word, int adcchannel)
{
  bool gooddata = true;
  // DigitMCMWord0x3 is odd 10 for odd adc channels and 11 for even, counted as the first of the 3.
  switch (word->c) {
    case 3: // even adc channnel
      if (adcchannel % 2 == 0) {
        gooddata = true;
      } else {
        gooddata = false;
      }
      break;
    case 2: // odd adc channel
      if (adcchannel % 2 == 1) {
        gooddata = true;
      } else {
        gooddata = false;
      }
      break;
    case 1: // error
      gooddata = false;
      break;
    case 0: // error
      gooddata = false;
      break;
      // no default all cases taken care of
  }
  return gooddata;
}

void printDigitMCMHeader(o2::trd::DigitMCMHeader& digitmcmhead)
{
  LOGF(info, "DigitMCMRawHeader: Raw:0x%08x res(0xc):0x%02x mcm: 0x%03x rob: 0x%03x eventcount 0x%05x year(>2007?): 0x%02x ",
       digitmcmhead.word, digitmcmhead.res, digitmcmhead.mcm, digitmcmhead.rob, digitmcmhead.eventcount,
       digitmcmhead.yearflag);
}

void printDigitMCMData(o2::trd::DigitMCMData& digitmcmdata)
{
  LOGF(info, "DigitMCMRawData: Raw:0x%08x res(0xc):0x%02x x: 0x%04x y: 0x%04x z 0x%04x ",
       digitmcmdata.word, digitmcmdata.c, digitmcmdata.x, digitmcmdata.y, digitmcmdata.z);
}

int getDigitHCHeaderWordType(uint32_t word)
{
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
  int countheaderwords = header.numberHCW;
  int index;
  //for the currently 3 implemeented other header words, they can come in any order, and are identified by their reserved portion
  for (int countheaderwords = 0; countheaderwords < header.numberHCW; ++countheaderwords) {
    switch (getDigitHCHeaderWordType(headers[countheaderwords])) {
      case 1:
        DigitHCHeader1 header1;
        header1.word = headers[countheaderwords];
        if (header1.res != 0x1) {
          printDigitHCHeaders(header, headers, index, countheaderwords, false);
        } else {
          printDigitHCHeaders(header, headers, index, countheaderwords, true);
        }
        break;
      case 2:
        DigitHCHeader2 header2;
        header2.word = headers[countheaderwords];
        if (header2.res != 0b110001) {
          printDigitHCHeaders(header, headers, index, countheaderwords, false);
        } else {
          printDigitHCHeaders(header, headers, index, countheaderwords, true);
        }
        break;
      case 3:
        DigitHCHeader3 header3;
        header3.word = headers[countheaderwords];
        if (header3.res != 0b110101) {
          printDigitHCHeaders(header, headers, index, countheaderwords, false);
        } else {
          printDigitHCHeaders(header, headers, index, countheaderwords, true);
        }
        break;
    }
  }
}

int getNumberOfTrackletsFromHeader(o2::trd::TrackletMCMHeader* header, bool verbose)
{
  int headertrackletcount = 0;
  if (header->pid0 == 0xff) {
    headertrackletcount = 0;
  } else {
    if (header->pid2 != 0xff) {
      // 3 tracklets
      headertrackletcount = 3;
      if (header->pid1 == 0xff || header->pid0 == 0xff) {
        if (verbose) {
          LOG(warn) << __func__ << " 3 tracklets but first 2 are empty ??" << header;
        }
      }
    } else {
      if (header->pid1 != 0xff) {
        // 2 tracklets
        headertrackletcount = 2;
        if (header->pid0 == 0xff) {
          if (verbose) {
            LOG(warn) << __func__ << " 2 tracklets but first is empty ??" << header;
          }
        }
      } else {
        if (header->pid0 != 0xff) {
          // 1 tracklet
          headertrackletcount = 1;
        } else {
          if (verbose) {
            LOG(warn) << __func__ << " 1 tracklet but first is empty ?? so should be no tracklets " << header;
          }
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
      LOG(error) << "tracklet header but no tracklets???";
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
      LOG(error) << "we have more than 3 tracklets for an mcm in " << __func__ << ". This should never happen: tracklet count=" << numberoftracklets;
  }
}

int getNextMCMADCfromBP(uint32_t& bp, int channel)
{
  //given a bitpattern (adcmask) find next channel with in the mask starting from the current channel.
  //channels are read from right to left, lsb to msb. channel zero is position 0 in the bit pattern.
  if (bp == 0) {
    return 22;
  }
  int position = channel;
  int m = 1 << channel;
  while (!(bp & m)) {
    m = m << 1;
    position++;
    if (position > 21) {
      break;
    }
  }
  bp &= ~(1UL << (position));
  return position;
}

} // namespace trd
} // namespace o2
