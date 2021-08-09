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

/// @file   CruRawReader.h
/// @brief  TRD raw data translator

#include "DetectorsRaw/RDHUtils.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Headers/RDHAny.h"
#include "TRDReconstruction/CruRawReader.h"
#include "TRDBase/FeeParam.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "TRDReconstruction/DigitsParser.h"
#include "TRDReconstruction/TrackletsParser.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Output.h"
#include "Framework/InputRecordWalker.h"

#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <numeric>
#include <iostream>

namespace o2::trd
{

bool CruRawReader::skipRDH()
{
  // check rdh for being empty or only padding words.
  if (o2::raw::RDHUtils::getMemorySize(mOpenRDH) == o2::raw::RDHUtils::getHeaderSize(mOpenRDH)) {
    //empty rdh so we want to avoid parsing it for cru data.
    LOG(info) << " skipping rdh (empty) with packetcounter of: " << std::hex << o2::raw::RDHUtils::getPacketCounter(mOpenRDH);
    return true;
  } else {

    if (mHBFPayload[0] == o2::trd::constants::CRUPADDING32 && mHBFPayload[0] == o2::trd::constants::CRUPADDING32) {
      //event only contains paddings words.
      LOG(info) << " skipping rdh (padding) with packetcounter of: " << std::hex << o2::raw::RDHUtils::getPacketCounter(mOpenRDH);
      // mDataPointer+= o2::raw::RDHUtils::getOffsetToNext()/4;
      auto rdh = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
      mDataPointer += o2::raw::RDHUtils::getOffsetToNext(rdh) / 4;
      //mDataPointer=reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh));
      return true;
      return true;
    } else {
      return false;
    }
  }
}

void CruRawReader::OutputHalfCruRawData()
{
  LOG(info) << "Full 1/2 CRU dump begin  **************************  FEEID:0x" << std::hex << mFEEID.word;
  for (int z = 0; z < 15; ++z) {
    LOG(info) << "link " << z << " length : " << mCurrentHalfCRULinkLengths[z] << " (256bit rows)";
  }
  int linkcount = 0;
  uint64_t linkzsum = 0;
  int bufferoffset = 0;
  uint64_t totalhalfcrulength = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
                                                mCurrentHalfCRULinkLengths.end(),
                                                decltype(mCurrentHalfCRULinkLengths)::value_type(0));
  totalhalfcrulength *= 8; //convert from 256 bits to 32 bits.
  LOGP(info, "CRH bufferoffset:{0:06d} :: {1:08x} {2:08x}  {3:08x} {4:08x} {5:08x} {6:08x} {7:08x} {8:08x} ", bufferoffset, HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 1]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 2]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 3]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 4]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 5]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 6]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 7]));
  bufferoffset += 8;
  LOGP(info, "CRH bufferoffset:{0:06d} :: {1:08x} {2:08x}  {3:08x} {4:08x} {5:08x} {6:08x} {7:08x} {8:08x} ", bufferoffset, HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 1]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 2]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 3]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 4]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 5]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 6]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 7]));
  int outputoffset = 2;
  bufferoffset += 8;
  int olengthoffset0 = 0;
  while (olengthoffset0 < totalhalfcrulength) {
    bufferoffset = olengthoffset0 + 16;
    if (mCurrentHalfCRULinkLengths[linkcount] == 0) {
      //output nothing, but state link empty
      LOG(info) << "empty link link:" << linkcount;
      linkcount++;
    } else {
      LOGP(info, "0x{0:06x} :: {1:08x} {2:08x}  {3:08x} {4:08x} {5:08x} {6:08x} {7:08x} {8:08x} ", olengthoffset0, HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 1]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 2]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 3]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 4]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 5]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 6]), HelperMethods::swapByteOrderreturn(mHBFPayload[bufferoffset + 7]));
      olengthoffset0 += 8; // advance a whole cru word of 256 bits or 8 32 bit words
      bufferoffset += 8;   // advance a whole cru word of 256 bits or 8 32 bit words
      if (olengthoffset0 == mCurrentHalfCRULinkLengths[linkcount] * 8 + linkzsum * 8) {
        LOG(info) << "end of link :" << linkcount;
        linkzsum += mCurrentHalfCRULinkLengths[linkcount];
        linkcount++;
      }
    }
  }
  LOG(info) << "CRU Next 16 words just for info";
  for (int bufferoffset = totalhalfcrulength; bufferoffset < totalhalfcrulength + 16; bufferoffset += 8) {
    LOGP(info, "0x{0:06x} :: {1:08x} {2:08x}  {3:08x} {4:08x} {5:08x} {6:08x} {7:08x} {8:08x} ", bufferoffset, mHBFPayload[bufferoffset], mHBFPayload[bufferoffset + 1], mHBFPayload[bufferoffset + 2], mHBFPayload[bufferoffset + 3], mHBFPayload[bufferoffset + 4], mHBFPayload[bufferoffset + 5], mHBFPayload[bufferoffset + 6], mHBFPayload[bufferoffset + 7]);
  }
  LOG(info) << "Full 1/2 CRU dump end ***************";
}

bool CruRawReader::processHBFs(int datasizealreadyread, bool verbose)
{
  if (mVerbose) {
    LOG(info) << "PROCESS HBF starting at " << std::hex << (void*)mDataPointer;
  }
  mDataRDH = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
  mOpenRDH = reinterpret_cast<o2::header::RDHAny*>((char*)mDataPointer);
  auto rdh = mDataRDH;
  auto preceedingrdh = rdh;
  uint64_t totaldataread = 0;
  mState = CRUStateHalfCRUHeader;
  uint32_t currentsaveddatacount = 0;
  mTotalHBFPayLoad = 0;
  int loopcount = 0;
  // loop until RDH stop header
  while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
    //o2::raw::RDHUtils::printRDH(rdh);
    if (mVerbose) {
      LOG(info) << "--- RDH open/continue detected loopcount :" << loopcount;
      LOG(info) << " rdh first word 0x" << std::hex << (uint32_t)*mDataPointer;
      for (int i = 0; i < 64; ++i) {
        LOG(info) << std::hex << " 0x" << *(mDataPointer + i);
      }
      LOG(info) << "---------------------- parsing that rdh";
    }
    preceedingrdh = rdh;
    auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
    auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
    auto rdhpayload = memorySize - headerSize;
    mFEEID.word = o2::raw::RDHUtils::getFEEID(rdh);       //TODO change this and just carry around the curreht RDH
    mCRUEndpoint = o2::raw::RDHUtils::getEndPointID(rdh); // the upper or lower half of the currently parsed cru 0-14 or 15-29
    mCRUID = o2::raw::RDHUtils::getCRUID(rdh);
    auto packetCount = o2::raw::RDHUtils::getPacketCounter(rdh);
    o2::InteractionRecord a = o2::raw::RDHUtils::getTriggerIR(rdh);
    mIR = a;
    mDataEndPointer = (const uint32_t*)((char*)rdh + offsetToNext);
    // copy the contents of the current rdh into the buffer to be parsed
    std::memcpy((char*)&mHBFPayload[0] + currentsaveddatacount, reinterpret_cast<const char*>(rdh) + headerSize, rdhpayload);
    mTotalHBFPayLoad += rdhpayload;
    currentsaveddatacount += rdhpayload;
    totaldataread += offsetToNext;
    // move to next rdh
    rdh = reinterpret_cast<const o2::header::RDHAny*>(reinterpret_cast<const char*>(rdh) + offsetToNext);
    if ((char*)(rdh) < (char*)&mHBFPayload[0] + mDataBufferSize) {
      //if (reinterpret_cast<const o2::header::RDHAny*>(rdh) < (char*)&mHBFPayload[0] + mDataBufferSize) {
      // we can still copy into this buffer.
    } else {
      LOG(warn) << "next rdh exceeds the bounds of the cru payload buffer";
      if (mVerbose) {
        LOG(info) << "rdh position  is out of bounds of the buffer";
        o2::raw::RDHUtils::printRDH(rdh);
      }
      return false; //-1;
    }
  }
  //increment the data pointer by the size of the stop rdh.
  mDataPointer = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh));
  // at this point the entire HBF data payload is sitting in mHBFPayload and the total data count is mTotalHBFPayLoad
  int counthalfcru = 0;
  mHBFoffset32 = 0;

  while ((mHBFoffset32 < ((mTotalHBFPayLoad) / 4))) { // the blank event of eeeeee at the end
    if (mVerbose) {
      LOG(info) << "Looping over cruheaders in HBF, loop count " << counthalfcru << " current offset is" << mHBFoffset32 << " total payload is " << mTotalHBFPayLoad / 4 << "  raw :" << mTotalHBFPayLoad;
    }
    int halfcruprocess = processHalfCRU(mHBFoffset32);
    if (mVerbose) {
      switch (halfcruprocess) {
        case -1:
          LOG(info) << "ignored rdh event ";
          break;
        case 0:
          LOG(fatal) << "figure out what now";
          break;
        case 1:
          LOG(info) << "all good parsing half cru";
          break;
        default:
          return true;
      }
    }
    //take care of the case where there is an "empty" rdh containing all 0xeeeeeeee as payload.
    if (mTotalHBFPayLoad / 4 - mHBFoffset32 == 8 && mHBFPayload[mHBFoffset32 + 7] == o2::trd::constants::CRUPADDING32) {
      mHBFoffset32 += 8;
    }
    counthalfcru++;
    if (counthalfcru == 1) {
      break;
    }
  } // loop of halfcru's while there is still data in the heart beat frame.
  if (totaldataread > 0) {
    mDatareadfromhbf = totaldataread;
  }
  return true; //totaldataread;
}

int CruRawReader::processHalfCRU(int cruhbfstartoffset)
{
  if (mVerbose) {
    LOG(info) << "************************ parsing HALFCRU starting at " << cruhbfstartoffset;
  }
  //It will clean this code up *alot*
  // process a halfcru
  uint32_t currentlinkindex = 0;
  uint32_t currentlinkoffset = 0;
  uint32_t currentlinksize = 0;
  uint32_t currentlinksize32 = 0;
  uint32_t linksizeAccum32 = 0;
  uint32_t sumtrackletwords = 0;
  uint32_t sumdigitwords = 0;
  uint32_t sumlinklengths = 0;
  int trackletwordsread = 0; // this will read up to the tracnklet end marker.
  int mDigitWordsRead = 0;
  int mDigitWordsRejected = 0;
  uint32_t cruwordsread = 9;
  //reject halfcru if it starts with padding words.
  //this should only hit that instance where the cru payload is a "blank event" of o2::trd::constants::CRUPADDING32
  if (mHBFPayload[cruhbfstartoffset] == o2::trd::constants::CRUPADDING32 && mHBFPayload[cruhbfstartoffset + 1] == o2::trd::constants::CRUPADDING32) {
    if (mVerbose) {
      LOG(info) << "blank rdh payload";
    }
    return -1;
  }
  if (mTotalHBFPayLoad == 0) {
    //empty payload
    return -1;
  }
  // well then read the halfcruheader.
  memcpy((char*)&mCurrentHalfCRUHeader, (void*)(&mHBFPayload[cruhbfstartoffset]), sizeof(mCurrentHalfCRUHeader)); //TODO remove the copy just use pointer dereferencing, doubt it will improve the speed much though.

  o2::trd::getlinkdatasizes(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths);
  o2::trd::getlinkerrorflags(mCurrentHalfCRUHeader, mCurrentHalfCRULinkErrorFlags);
  mTotalHalfCRUDataLength256 = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
                                               mCurrentHalfCRULinkLengths.end(),
                                               decltype(mCurrentHalfCRULinkLengths)::value_type(0));
  mTotalHalfCRUDataLength = mTotalHalfCRUDataLength256 * 32; //convert to bytes.
  int mTotalHalfCRUDataLength32 = mTotalHalfCRUDataLength256 * 8; //convert to bytes.
                                                                  //check for cru errors :
                                                                  //  if (mHeaderVerbose) {
  int linkerrorcounter = 0;
  if (mHeaderVerbose) {
    LOG(info) << "link errors";
    for (auto& linkerror : mCurrentHalfCRULinkErrorFlags) {
      if (linkerror != 0) {
        LOG(info) << "E link error FEEID:" << mFEEID.word << " CRUID:" << mCRUID << " Endpoint:" << mCRUEndpoint
                  << " on linkcount:" << linkerrorcounter++ << " errorval:0x" << std::hex << linkerror;
      }
    }
  }
  std::array<uint32_t, 1024>::iterator currentlinkstart = mHBFPayload.begin() + cruhbfstartoffset;
  if (mHeaderVerbose) { //TODO put the following if statement into a method to simplify reading
    OutputHalfCruRawData();
  }
  std::array<uint32_t, 1024>::iterator linkstart, linkend;
  int dataoffsetstart32 = sizeof(mCurrentHalfCRUHeader) / 4 + cruhbfstartoffset; // in uint32
  //CHECK 1 does rdh endpoint match cru header end point.
  if (mCRUEndpoint != mCurrentHalfCRUHeader.EndPoint) {
    LOG(warn) << " Endpoint mismatch : CRU Half chamber header endpoint = " << mCurrentHalfCRUHeader.EndPoint << " rdh end point = " << mCRUEndpoint;
    //disaster dump the rest of this hbf
    return 42;
    if (mVerbose) {
      LOG(info) << "******* LINK # " << currentlinkindex;
    }
  }

  // verify cru header vs rdh header
  //FEEID has supermodule/layer/stack/side in it.
  //CRU has
  mHBFoffset32 += sizeof(mCurrentHalfCRUHeader) / 4;
  linkstart = mHBFPayload.begin() + dataoffsetstart32;
  linkend = mHBFPayload.begin() + dataoffsetstart32;
  //loop over links
  for (currentlinkindex = 0; currentlinkindex < constants::NLINKSPERHALFCRU; currentlinkindex++) {
    auto linktimerstart = std::chrono::high_resolution_clock::now(); // measure total processing time
    currentlinksize = mCurrentHalfCRULinkLengths[currentlinkindex];
    currentlinksize32 = currentlinksize * 8; //x8 to go from 256 bits to 32 bit;
    linkstart = mHBFPayload.begin() + dataoffsetstart32 + linksizeAccum32;
    linkend = linkstart + currentlinksize32;
    uint64_t linkzsum = 0;
    int dioffset = dataoffsetstart32 + linksizeAccum32;
    if (dioffset % 8 != 0) {
      LOG(error) << " we are not 256 bit aligned ... this should never happen";
    }
    if (mHeaderVerbose) {
      LOG(info) << "Cru link :" << currentlinkindex << " raw dump before processing begin linkstart:" << std::hex << linkstart << " to " << linkend;
      for (int dumpoffset = dataoffsetstart32 + linksizeAccum32; dumpoffset < dataoffsetstart32 + linksizeAccum32 + currentlinksize32; dumpoffset += 8) {
        LOGP(info, "0x{0:06x} :: {1:08x} {2:08x}  {3:08x} {4:08x} {5:08x} {6:08x} {7:08x} {8:08x} ", dumpoffset, HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 1]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 2]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 3]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 4]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 5]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 6]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 7]));
      }
      LOG(info) << "Cru link :" << currentlinkindex << " raw dump before processing end";
    }
    linksizeAccum32 += currentlinksize32;
    int supermodule = mFEEID.supermodule;
    int endpoint = mFEEID.endpoint;
    int side = mFEEID.side;
    //stack layer and side map to ori
    int stack, layer, halfchamberside;
    int oriindex = currentlinkindex + constants::NLINKSPERHALFCRU * endpoint; // endpoint denotes the pci side, upper or lower for the pair of 15 fibres.
    FeeParam::unpackORI(oriindex, side, stack, layer, halfchamberside);
    int currentdetector = stack * constants::NLAYER + layer + supermodule * constants::NLAYER * constants::NSTACK;
    if (mDataVerbose) {
      LOG(info) << "******* LINK # " << currentlinkindex << " and  starting at " << mHBFoffset32 << " unpackORI(" << oriindex << "," << side << "," << stack << "," << layer << "," << halfchamberside << ") and an FEEID:" << std::hex << mFEEID.word << " det:" << std::dec << currentdetector;
      LOG(info) << "******* LINK # " << currentlinkindex << " an FEEID:" << std::hex << mFEEID.word << " det:" << std::dec << currentdetector << " Error Flags : " << mCurrentHalfCRULinkErrorFlags[currentlinkindex];
    }
    if (linkstart != linkend) { // if link is not empty
      bool cleardigits = false; //linkstart and linkend already have the multiple cruheaderoffsets built in
      trackletwordsread = mTrackletsParser.Parse(&mHBFPayload, linkstart, linkend, mFEEID, halfchamberside, currentdetector, stack, layer, cleardigits, mByteSwap, mTrackletHCHeaderState, mVerbose, mHeaderVerbose, mDataVerbose); // this will read up to the tracklet end marker.
      if (mVerbose) {
        LOG(info) << "trackletwordsread:" << trackletwordsread << "  mem copy with offset of : " << cruhbfstartoffset << " parsing with linkstart: " << linkstart << " ending at : " << linkend;
      }
      linkstart += trackletwordsread;
      //now we have a tracklethcheader and a digithcheader.
      mHBFoffset32 += trackletwordsread;
      mTotalTrackletsFound += mTrackletsParser.getTrackletsFound();
      //now read the digit half chamber header
      DigitHCHeader digitHCHeader;
      uint32_t dhcheader0 = mHBFPayload[mHBFoffset32++];
      uint32_t dhcheader1 = mHBFPayload[mHBFoffset32++];
      if (mByteSwap) {
        // byte swap if needed.
        o2::trd::HelperMethods::swapByteOrder(dhcheader0);
        o2::trd::HelperMethods::swapByteOrder(dhcheader1);
      }
      digitHCHeader.word0 = dhcheader0;
      digitHCHeader.word1 = dhcheader1;
      if (mHeaderVerbose) {
        LOG(info) << "*** HCHHeader : 0x" << std::hex << digitHCHeader.word0 << " 0x" << digitHCHeader.word1;
        printDigitHCHeader(digitHCHeader);
      }
      if (digitHCHeader.word0 == 0x0 || digitHCHeader.word1 == 0x0) {
        LOG(warn) << "Missing DigitHCHeader, read digit end marker of zeros";
        printDigitHCHeader(digitHCHeader);
      }
      //move over the DigitHCHeader mHBFoffset32 has already been moved in the reading.
      linkstart += 2;
      if (digitHCHeader.major == 0x47) {
        // config event so ignore for now and bail out of parsing.
        LOG(warn) << " HCHeader major version is 0x47 bailing out of parsing this as its a config event";
        //advance data pointers to the end;
        linkstart = linkend;
        mHBFoffset32 = dataoffsetstart32 + currentlinksize; // go to the end of the link
      } else {
        mDigitWordsRead = 0;
        //linkstart and linkend already have the multiple cruheaderoffsets built in
        mDigitWordsRead = mDigitsParser.Parse(&mHBFPayload, linkstart, linkend, currentdetector, stack, layer, digitHCHeader, mFEEID, currentlinkindex, cleardigits, mByteSwap, mVerbose, mHeaderVerbose, mDataVerbose);
        mDigitWordsRejected = mDigitsParser.getDumpedDataCount();
        if (mDigitsParser.getDumpedDataCount() != 0) {
          LOG(info) << "FEEID: " << mFEEID.word << " LINK #" << oriindex << " bad datacount:" << mDigitsParser.getDataWordsParsed() << "::" << mDigitsParser.getDumpedDataCount();
        } else {
          LOG(info) << "FEEID: " << mFEEID.word << " LINK #" << oriindex << " good datacount:" << mDigitsParser.getDataWordsParsed() << "::" << mDigitsParser.getDumpedDataCount();
        }
        if (mDigitWordsRead != std::distance(linkstart, linkend)) {
          //we have the data corruption problem of a pile of stuff at the end of a link, jump over it.
          if (mFixDigitEndCorruption) {
            mDigitWordsRead = std::distance(linkstart, linkend);
          } else {
            LOG(warn) << "read digits but data still left on the link digitwordsread:" << mDigitWordsRead << " and link length:" << std::distance(linkstart, linkend);
          }
        }
        mTotalDigitsFound += mDigitsParser.getDigitsFound();
        if (mVerbose) {
          LOG(info) << "mDigitWordsRead : " << mDigitWordsRead << " mem copy with offset of : " << cruhbfstartoffset << " parsing digits with linkstart: " << linkstart << " ending at : " << linkend;
        }
        sumlinklengths += mCurrentHalfCRULinkLengths[currentlinkindex];
        sumtrackletwords += trackletwordsread;
        sumdigitwords += mDigitWordsRead;
        mHBFoffset32 += mDigitWordsRead + mDigitWordsRejected; // all 3 in 32bit units
        mTotalDigitWordsRead = mDigitWordsRead;
        mTotalDigitWordsRejected = mDigitWordsRejected;
      }
    } else {
      if (mVerbose) {
        LOG(info) << "link start and end are the same, link appears to be empty for link currentlinkdex";
      }
    }
  } //for loop over link index.
  // we have read in all the digits and tracklets for this event.
  //digits and tracklets are sitting inside the parsing classes.
  //extract the vectors and copy them to tracklets and digits here, building the indexing(triggerrecords)
  //as this is for a single cru half chamber header all the tracklets and digits are for the same trigger defined by the bc and orbit in the rdh which we hold in mIR
  mIR.bc = mCurrentHalfCRUHeader.BunchCrossing; // correct mIR to have the physics trigger bunchcrossing *NOT* the heartbeat trigger bunch crossing.

  mEventRecords.addTracklets(mIR, mTrackletsParser.getTracklets());
  if (mVerbose) {
    LOG(info) << "inserting tracklets from parser of size : " << mTrackletsParser.getTracklets().size() << " mEventRecordsTracklets is now :" << mEventRecords.sumTracklets();
  }
  mTrackletsParser.clear();
  mEventRecords.addDigits(mIR, std::begin(mDigitsParser.getDigits()), std::end(mDigitsParser.getDigits()));
  if (mVerbose) {
    LOG(info) << "inserting digits from parser of size : " << mDigitsParser.getDigits().size();
  }
  mDigitsParser.clear();
  if (mVerbose) {
    LOG(info) << "Event digits after eventi # : " << mEventRecords.sumDigits() << " having added : via sum=" << mDigitsParser.getDigits().size() << " digitsfound is " << mDigitsParser.getDigitsFound();
  }
  int lasttrigger = 0, lastdigit = 0, lasttracklet = 0;

  //if we get here all is ok.
  return 1;
}

bool CruRawReader::buildCRUPayLoad()
{
  // copy data for the current half cru, and when we eventually get to the end of the payload return 1
  // to say we are done.
  int cruid = 0;
  int additionalBytes = -1;
  int crudatasize = -1;
  LOG(info) << "--- Build CRU Payload, added " << additionalBytes << " bytes to CRU "
            << cruid << " with new size " << crudatasize;
  return true;
}

bool CruRawReader::processCRULink()
{
  /* process a CRU Link 15 per half cru */
  //  checkFeeID(); // check the link we are working with corresponds with the FeeID we have in the current rdh.
  //  uint32_t slotId = GET_TRMDATAHEADER_SLOTID(*mDataPointer);
  return false;
}

void CruRawReader::resetCounters()
{
  TRDStatCountersPerEvent.mLinkErrorFlag.fill(0);
  mEventCounter = 0;
  mFatalCounter = 0;
  mErrorCounter = 0;
}

void CruRawReader::checkSummary()
{
  char chname[2] = {'a', 'b'};

  LOG(info) << "--- SUMMARY COUNTERS: " << mEventCounter << " events "
            << " | " << mFatalCounter << " decode fatals "
            << " | " << mErrorCounter << " decode errors ";
}

bool CruRawReader::run()
{
  uint32_t dowhilecount = 0;
  uint64_t totaldataread = 0;
  rewind();
  uint64_t mTotalDigitWordsRead = 0;
  uint64_t mTotalDigitWordsRejected = 0;
  uint32_t* bufferptr;
  bufferptr = (uint32_t*)mDataBuffer;
  do {
    if (mDataVerbose) {
      LOG(info) << " mDataBuffer :" << (void*)mDataBuffer << " and offset to start on is :" << totaldataread;
    }
    mDatareadfromhbf = 0;
    processHBFs(totaldataread, mVerbose);
    totaldataread += mDatareadfromhbf;
  } while (((char*)mDataPointer - mDataBuffer) < mDataBufferSize);

  return false;
};

void CruRawReader::getParsedObjects(std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits, std::vector<TriggerRecord>& triggers)
{
  int digitcountsum = 0;
  int trackletcountsum = 0;
  mEventRecords.unpackData(triggers, tracklets, digits);
}

void CruRawReader::getParsedObjectsandClear(std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits, std::vector<TriggerRecord>& triggers)
{
  getParsedObjects(tracklets, digits, triggers);
  clearall();
}

//write the output data directly to the given DataAllocator from the datareader task.
void CruRawReader::buildDPLOutputs(o2::framework::ProcessingContext& pc, bool displaytracklets)
{
  mEventRecords.sendData(pc, displaytracklets);
  clearall(); // having now written the messages clear for next.
}

} // namespace o2::trd
