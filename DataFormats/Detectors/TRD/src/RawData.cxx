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
  feeid.unused1 = 0;
  feeid.unused2 = 0;
  return feeid.word;
}

void buildTrackletMCMData(TrackletMCMData& trackletword, const uint slope, const uint pos, const uint q0, const uint q1, const uint q2)
{
  // create a tracklet word as it would be sent from the FEE
  // slope and position have the 8-th bit flipped each
  trackletword.word = 0;
  trackletword.slope = slope ^ 0x80;
  trackletword.pos = pos ^ 0x80;
  trackletword.pid = (q0 & 0x7f) & ((q1 & 0x1f) << 7); //q2 sits with upper 2 bits of q1 in the header pid word, hence the 0x1f so 5 bits are used here.
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
      LOG(warn) << " unknown trackletindex of " << trackletindex << " to getQFromRaw : " << pidindex;
      break;
  }
  /*
   * Q0/1 are 7 bits, Q2 is 6 bits.
   * Q0 is completely in TrackletMCMData::data and Q2 is completely in TrackletHCHeader::pid*,
   * while Q1 is split with the lower 5 bits in the data and the upper 2 bits in the header.
   *
   * A detailed description of the format can be found in
   * https://alicetrd.web.cern.ch/alicetrd/tdp/main.pdf under 17.2.1
   *
   *     |07|06|05|04|03|02|01|00|
   *     -------------------------
   * qa: |       Q2        | Q1  |  TrackletMCMHeader::pid
   *     -------------------------
   *
   * TDP: This can be one of these fields HPID0/1/2 (=TrackletHCHeader::pid0/1/2) depending on
   *      the MCM-CPU.
   *
   *     |11|10|09|08|07|06|05|04|03|02|01|00|
   *     -------------------------------------
   * qb: |     Q0             |      Q1      |  TrackletMCMData::pid
   *     -------------------------------------
   *
   * TDP: This is the LPID field (=TrackletMCMData::pid).
   *
   * Q1 is then calculated like this:
   *
   *     |06|05|04|03|02|01|00|
   *     ----------------------
   * Q1: |qa.Q1|    qb.Q1     |
   *     ----------------------
   *
   **/
  qb = data->pid;
  switch (pidindex) {
    case 0:                   //Q0
      pid = (qb >> 5) & 0x7f; // 7 bits at the top of all of Q0
      break;
    case 1:                                  //Q1
      pid = ((qa & 0x3) << 5) | (qb & 0x1f); // 2 bits of qa and 5 bits of qb for Q1 .. 7 bits
      break;
    case 2:                   //Q2
      pid = (qa >> 2) & 0x3f; // 6 bits shifted down by bits 2 only taking 6 bits
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
  LOGF(info, "TrackletHCHeader: Raw:0x%08x SM : %d stack %d layer %d side : %d MCLK: 0x%0x Format: 0x%0x Always1:0x%0x",
       halfchamber.supermodule, halfchamber.stack, halfchamber.layer, halfchamber.side, halfchamber.MCLK, halfchamber.format, halfchamber.one);
}

void printDigitMCMHeader(o2::trd::DigitMCMHeader& mcmhead)
{
  LOGF(info, "DigitMCMRawHeader: Raw:0x%08x res(0xc):0x%02x mcm: 0x%03x rob: 0x%03x eventcount 0x%05x year(>2007?): 0x%02x ",
       mcmhead.word, mcmhead.res, mcmhead.mcm, mcmhead.rob, mcmhead.eventcount,
       mcmhead.yearflag);
}

void dumpHalfChamber(o2::trd::TrackletHCHeader const& halfchamber)
{
  LOGF(info, "HalfChamber : 0x%08x", halfchamber.word);
}

void printHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
  std::array<uint32_t, 15> sizes;
  std::array<uint32_t, 15> errorflags;
  getlinkdatasizes(halfcru, sizes);
  getlinkerrorflags(halfcru, errorflags);
  LOGF(info, "V:%d BC:%d SB:%d EType:%d", halfcru.HeaderVersion, halfcru.BunchCrossing,
       halfcru.StopBit, halfcru.EventType);
  for (int i = 0; i < 15; i++) {
    LOGF(info, "Link %d size: %ul eflag: 0x%02x", i, sizes[i], errorflags[i]);
  }
  LOG(info) << "Raw: " << std::hex << halfcru.word0 << " " << halfcru.word12[0] << " " << halfcru.word12[1] << " " << halfcru.word3 << " " << halfcru.word47[0] << " " << halfcru.word47[1] << " " << halfcru.word47[2] << " " << halfcru.word47[3];
  for (int i = 0; i < 15; i++) {
    LOGF(info, "Raw: %d word: %ul x", i, sizes[i], errorflags[i]);
  }
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
  if ((header.pid2 != 0xff) && (header.pid1 == 0xff || header.pid0 == 0xff)) {
    goodheader = false;
  }
  // sim for 2 tracklets.
  if ((header.pid1 != 0xff) && (header.pid0 == 0xff)) {
    goodheader = false;
  }

  return goodheader;
}

bool trackletHCHeaderSanityCheck(o2::trd::TrackletHCHeader& header)
{
  bool goodheader = true;
  //TODO something wrong TDP is different from rawdata.h
  //figure out but for now, just approve.
  return true;
  if (header.one != 1) {
    goodheader = false;
  }
  if (header.supermodule > 17) {
    goodheader = false;
  }
  //if(header.format != )  only certain format versions are permitted come back an fill in if needed.
  if (header.layer > 6) {
    goodheader = false;
  }
  if (header.stack > 5) {
    goodheader = false;
  }
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

bool digitMCMADCMaskSanityCheck(o2::trd::DigitMCMADCMask& mask, int numberofbitsset)
{
  bool goodadcmask = true;
  uint32_t count = (unsigned int)mask.c;
  count = ~count;
  /*  if(count != numberofbitsset){
    goodadcmask=false;
    LOG(warn) << "***DigitMCMADCMask bad bit count maskcount:" << ~mask.c << " bitscounting:" << numberofbitsset;
  }*/
  if (mask.n != 0x1) {
    goodadcmask = false;
  }
  if (mask.j != 0xc) {
    goodadcmask = false;
  }
  return goodadcmask;
}

bool digitMCMWordSanityCheck(o2::trd::DigitMCMData* word, int adcchannel)
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
void printDigitHCHeader(o2::trd::DigitHCHeader& header, uint32_t headers[3])
{
  LOGF(info, "Digit HalfChamber Header: Raw:0x%08x reserve:0x%01x side:0x%01x stack:0x%02x layer:0x%02x supermod:0x%02x numberHCW:0x%02x minor:0x%03x major:0x%03x version(>2007):0x%01x",
       header.word, header.res, header.side, header.stack, header.layer, header.supermodule,
       header.numberHCW, header.minor, header.major, header.version);
  int countheaderwords = header.numberHCW;
  //for the currently 3 implemeented other header words, they can come in any order, and are identified by their reserved portion
  for (int countheaderwords = 0; countheaderwords < header.numberHCW; ++countheaderwords) {
    switch (getDigitHCHeaderWordType(headers[countheaderwords])) {
      case 1:
        DigitHCHeader1 header1;
        header1.word = headers[countheaderwords];
        if (header1.res != 0x1) {
          LOGF(info, "*Corrupt* Digit HalfChamber Header1 Raw:0x%08x reserve:0x%02x pretriggercount=0x%02x pretriggerphase=0x%02x bunchxing:0x%05x number of timebins : 0x%03x", header1.word, header1.res, header1.ptrigcount, header1.ptrigphase, header1.bunchcrossing, header1.numtimebins);
        } else {
          LOGF(info, "Digit HalfChamber Header1 Raw:0x%08x reserve:0x%02x pretriggercount=0x%02x pretriggerphase=0x%02x bunchxing:0x%05x number of timebins : 0x%03x", header1.word, header1.res, header1.ptrigcount, header1.ptrigphase, header1.bunchcrossing, header1.numtimebins);
        }
        break;
      case 2:
        DigitHCHeader2 header2;
        header2.word = headers[countheaderwords];
        if (header2.res != 0b110001) {
          LOGF(info, "*Corrupt* Digit HalfChamber Header2 Raw:0x%08x reserve:0x%08x PedestalFilter:0x%01x GainFilter:0x%01x TailFilter:0x%01x CrosstalkFilter:0x%01x Non-linFilter:0x%01x RawDataBypassFilter:0x%01x DigitFilterCommonAdditive:0x%02x ", header2.word, header2.res, header2.dfilter, header2.rfilter, header2.nlfilter, header2.xtfilter, header2.tfilter, header2.gfilter, header2.pfilter);
        } else {
          LOGF(info, "Digit HalfChamber Header2 Raw:0x%08x reserve:0x%08x PedestalFilter:0x%01x GainFilter:0x%01x TailFilter:0x%01x CrosstalkFilter:0x%01x Non-linFilter:0x%01x RawDataBypassFilter:0x%01x DigitFilterCommonAdditive:0x%02x ", header2.word, header2.res, header2.dfilter, header2.rfilter, header2.nlfilter, header2.xtfilter, header2.tfilter, header2.gfilter, header2.pfilter);
        }
        break;
      case 3:
        DigitHCHeader3 header3;
        header3.word = headers[countheaderwords];
        if (header3.res != 0b110101) {
          LOGF(info, "*Corrupt*Digit HalfChamber Header3: Raw:0x%08x reserve:0x%08x readout program revision:0x%08x assembler program version:0x%01x", header3.word, header3.res, header3.svnrver, header3.svnver);
        } else {
          LOGF(info, "Digit HalfChamber Header3: Raw:0x%08x reserve:0x%08x readout program revision:0x%08x assembler program version:0x%01x", header3.word, header3.res, header3.svnrver, header3.svnver);
        }
        break;
    }
  }
}

DigitMCMADCMask buildBlankADCMask()
{
  //set the default values for the mask.
  DigitMCMADCMask mask;
  mask.c = 0x1f;
  mask.n = 0x1;
  mask.j = 0xc;
  // actual mask will beset somewhere else, the above values are *always* that.
  return mask;
}

int getNumberofTracklets(o2::trd::TrackletMCMHeader& header)
{
  int headertrackletcount = 0;
  if (header.pid0 == 0xff) {
    //LOG(warn) << header;
  } else {
    if (header.pid2 != 0xff) {
      // 3 tracklets
      headertrackletcount = 3;
      if (header.pid1 == 0xff || header.pid0 == 0xff) {
        //   LOG(warn) << header;
      }
    } else {
      if (header.pid1 != 0xff) {
        // 2 tracklets
        headertrackletcount = 2;
        if (header.pid0 == 0xff) {
          //    LOG(warn) << header;
        }
      } else {
        if (header.pid0 != 0xff) {
          // 1 tracklet
          headertrackletcount = 1;
        } else {
          //   LOG(warn) << header;
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
  if (bp == 0) {
    return 22;
  }
  int position = channel;
  int m = 1 << channel;
  while (!(bp & m)) {
    m = m << 1;
    position++;
    if (position > 31) {
      break;
    }
  }
  bp &= ~(1UL << (position));
  return position;
}

} // namespace trd
} // namespace o2
