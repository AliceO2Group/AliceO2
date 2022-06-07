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
    if (mVerbose) {
      LOG(info) << " skipping rdh (empty) packetcount of: " << std::hex << o2::raw::RDHUtils::getPacketCounter(mOpenRDH);
    }
    return true;
  } else {

    if (mHBFPayload[0] == o2::trd::constants::CRUPADDING32 && mHBFPayload[0] == o2::trd::constants::CRUPADDING32) {
      //event only contains paddings words.
      if (mVerbose) {
        LOG(info) << " skipping rdh (padding) with packetcounter of: " << std::hex << o2::raw::RDHUtils::getPacketCounter(mOpenRDH);
      }
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
  uint32_t linkzsum = 0;
  int bufferoffset = 0;
  uint32_t totalhalfcrulength = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
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

void CruRawReader::dumpRDHAndNextHeader(const o2::header::RDHAny* rdh)
{
  LOG(info) << "######################### Dumping RDH incoming buffer ##########################";
  //  o2::raw::RDHUtils::printRDH(rdh);
  LOG(info) << "Now for the buffer breakdown";
  auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
  for (int i = 0; i < offsetToNext / 4; ++i) {
    LOG(info) << "ii:" << i << " 0x" << std::hex << *((uint32_t*)rdh + i); //*(uint32_t*)(reinterpret_cast<const uint32_t*>(rdh) + i);
  }
  LOG(info) << "Next 8....";
  for (int i = 0; i < 8; ++i) {
    LOG(info) << "iii:" << i << " 0x" << std::hex << *((uint32_t*)rdh + i + offsetToNext); //*(uint32_t*)(reinterpret_cast<const uint32_t*>(rdh) + i+offsetToNext);
  }
  LOG(info) << "######################### Finished RDH incoming buffer ##########################";
}

bool CruRawReader::processHBFs(int datasizealreadyread, bool verbose)
{
  if (mHeaderVerbose) {
    LOG(info) << "PROCESS HBF starting at " << std::hex << (void*)mDataPointer << " already read in : " << datasizealreadyread;
  }
  mDataRDH = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
  mOpenRDH = reinterpret_cast<const o2::header::RDHAny*>((const char*)mDataPointer);
  auto rdh = mDataRDH;
  auto preceedingrdh = rdh;
  uint32_t totaldataread = 0;
  if (mHeaderVerbose) {
    LOG(info) << " mem : " << o2::raw::RDHUtils::getMemorySize(rdh) << " headersize : " << o2::raw::RDHUtils::getHeaderSize(rdh);
  }
  mState = CRUStateHalfCRUHeader;
  uint32_t currentsaveddatacount = 0;
  mTotalHBFPayLoad = 0;
  int loopcount = 0;
  int counthalfcru = 0;
  mHBFoffset32 = 0;
  // loop until RDH stop header
  while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
    if (mHeaderVerbose) {
      LOG(info) << "----------------------------------------------";
      LOG(info) << " rdh first word 0x" << std::hex << (uint32_t)*mDataPointer;
      LOG(info) << " rdh first word is sitting at 0x" << std::hex << (void*)mDataPointer;
      o2::raw::RDHUtils::printRDH(rdh);
      dumpRDHAndNextHeader(rdh);
    }
    preceedingrdh = rdh;
    auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
    if (memorySize == 0) {
      LOG(warn) << "rdh memory size is zero";
      break; // get out of here if the rdh says it has nothing.
    }
    auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
    auto rdhpayload = memorySize - headerSize;
    mFEEID.word = o2::raw::RDHUtils::getFEEID(rdh);       //TODO change this and just carry around the curreht RDH
    mCRUEndpoint = o2::raw::RDHUtils::getEndPointID(rdh); // the upper or lower half of the currently parsed cru 0-14 or 15-29
    mCRUID = o2::raw::RDHUtils::getCRUID(rdh);
    mIR = o2::raw::RDHUtils::getTriggerIR(rdh);
    int packetCount = o2::raw::RDHUtils::getPacketCounter(rdh);
    //mDataEndPointer = (uint32_t*)((char*)rdh + offsetToNext);
    if (mOptions[TRDM1Debug]) {
      LOG(info) << "mFEEID:" << mFEEID.word << " mCRUEndpoint:" << mCRUEndpoint << " mCRUID:" << mCRUID << " packetCount:" << packetCount << "rdhpayload:" << rdhpayload << " offsettonext:" << offsetToNext << " dmDataEndPointer(after move to rdh+offsetToNext): 0x" << std::hex << (void*)mDataEndPointer << " rdh is currently at 0x" << std::hex << (void*)rdh;
    }
    // copy the contents of the current rdh into the buffer to be parsed
    std::memcpy((char*)&mHBFPayload[0] + currentsaveddatacount, ((char*)rdh) + headerSize, rdhpayload);
    mTotalHBFPayLoad += rdhpayload;
    currentsaveddatacount += rdhpayload;
    totaldataread += offsetToNext;
    // move to next rdh
    auto oldRDH = rdh;
    rdh = reinterpret_cast<const o2::header::RDHAny*>(reinterpret_cast<const char*>(rdh) + offsetToNext);
    //increment the data pointer by the size of the stop rdh.
    mDataPointer = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh));

    if (mHeaderVerbose) {
      if (!o2::raw::RDHUtils::getStop(rdh)) {
        LOG(info) << "Next rdh is not a stop, and has a header size of " << o2::raw::RDHUtils::getHeaderSize(rdh) << " and memsize of : " << o2::raw::RDHUtils::getMemorySize(rdh);
        LOG(info) << "rdh 0x" << (void*)rdh << " bufsize:" << mDataBufferSize << " payload start: 0x" << (void*)&mHBFPayload[0] << " mHBFoffset32 " << std::dec << mHBFoffset32;
        LOGP(info, " rdh::: {0:08x} {1:08x} {2:08x}  {3:08x} {4:08x} {5:08x} {6:08x} {7:08x} ", *((uint32_t*)rdh), *((uint32_t*)rdh + 1), *((uint32_t*)rdh + 2), *((uint32_t*)rdh + 3), *((uint32_t*)rdh + 4), *((uint32_t*)rdh + 5), *((uint32_t*)rdh + 6), *((uint32_t*)rdh + 7), *((uint32_t*)rdh + 8));
        //      o2::raw::RDHUtils::printRDH(rdh);
      } else {
        LOG(info) << "Next rdh is a stop, and we have moved to it.";
      }
    }

    if (o2::raw::RDHUtils::getStop(rdh) || o2::raw::RDHUtils::getOffsetToNext(rdh) < mDataBufferSize - mHBFoffset32) {
      // we can still copy into this buffer.
    } else {
      if (mMaxWarnPrinted > 0) {
        LOG(warn) << "rdh bounds fail offsetToNext:" << offsetToNext << " rdh 0x" << (void*)rdh << " bufsize:" << mDataBufferSize << " payload start: 0x" << (void*)&mHBFPayload[0] << " mHBFoffset32 " << std::dec << mHBFoffset32;
        checkNoWarn();
      }
        if (mVerbose) {
          LOG(info) << "rdh bounds fail offsetToNext:" << offsetToNext << " rdh 0x" << (void*)rdh << " bufsize:" << mDataBufferSize << " payload start: 0x" << (void*)&mHBFPayload[0] << " mHBFoffset32 " << std::dec << mHBFoffset32;
          //o2::raw::RDHUtils::printRDH(rdh);
        }
        if (mVerbose || mOptions[TRDM1Debug]) {
          LOG(warn) << "returning from processHBFs with a false";
          LOG(warn) << "rdh in question is : ";
          //o2::raw::RDHUtils::printRDH(rdh);
        }
      return false; //-1;
    }
  }

  // at this point the entire HBF data payload is sitting in mHBFPayload and the total data count is mTotalHBFPayLoad
  while ((mHBFoffset32 < ((mTotalHBFPayLoad) / 4))) {
    if (mHeaderVerbose) {
      LOG(info) << "Looping over cruheaders in HBF, loop count " << counthalfcru << " current offset is" << mHBFoffset32 << " total payload is " << mTotalHBFPayLoad / 4 << "  raw :" << mTotalHBFPayLoad;
    }
    int halfcruprocess = processHalfCRU(mHBFoffset32);
    if (mVerbose) {
      switch (halfcruprocess) {
        case -1:
          LOG(info) << "ignored rdh event ";
          break;
        case 0:
          LOG(error) << "figure out what now";
          break;
        case 1:
          LOG(info) << "all good parsing half cru";
          LOG(info) << " mHBFoffset32:" << mHBFoffset32 << " and mTotalHBFPayload/4 : " << mTotalHBFPayLoad / 4;
          break;
        case 2:
          LOG(info) << "all good parsing half cru was blank double 0xe event";
          break;
          //        default:
          //          return true;
      }
    }
    //take care of the case where there is an "empty" rdh containing all 0xeeeeeeee as payload.
    //    if (mTotalHBFPayLoad / 4 - mHBFoffset32 == 8 && mHBFPayload[mHBFoffset32 + 7] == o2::trd::constants::CRUPADDING32) {
    //      mHBFoffset32 += 8;
    //    }
    if (halfcruprocess == -2) {
      //dump rest of this rdh payload, something screwed up.
      mHBFoffset32 = mTotalHBFPayLoad / 4;
    }
    counthalfcru++;
  } // loop of halfcru's while there is still data in the heart beat frame.
  if (totaldataread > 0) {
    mDatareadfromhbf = totaldataread;
  }
  return true; //totaldataread;
}

int CruRawReader::checkTrackletHCHeader()
{
  // index 0 is rdh data, index 1 is ori calculated data
  auto currentsector = mTrackletHCHeader.supermodule;
  auto currentlayer = mTrackletHCHeader.layer;
  auto currentstack = mTrackletHCHeader.stack;
  auto currentside = mTrackletHCHeader.side;
  if (!mOptions[TRDIgnoreTrackletHCHeaderBit]) { // we take half chamber header as authoritive
    return 0;
  }
  return 0; // for now always ignore, something is wrong with it, yet to be determined, tdp and header dont match.  FIXME!
}

int CruRawReader::checkDigitHCHeader()
{
  // index 0 is rdh data, index 1 is ori calculated data
  auto currentsector = mDigitHCHeader.supermodule;
  auto currentlayer = mDigitHCHeader.layer;
  auto currentstack = mDigitHCHeader.stack;
  auto currentside = mDigitHCHeader.side;
  //check rdh info vs half chamber header
  if (!mOptions[TRDIgnoreDigitHCHeaderBit]) { // we take half chamber header as authoritive
    // can use digithcheader for cross checking the sector/stack/layer
    if (currentstack != mStack[0] || currentstack != mStack[1]) {
      //stack mismatch
      //count these
      //mEventRecord.ErrorStats[TRDParsingDigitStackMismatch]++;
      increment2dHist(TRDParsingDigitStackMismatch, mFEEID.supermodule * 2 + mHalfChamberSide[0], mStack[0], mLayer[0]);
    }
    if (currentlayer != mLayer[0] || currentlayer != mLayer[1]) {
      //layer mismatch
      //count these
      //mEventRecord.ErrorStats[TRDParsingDigitLayerMisMatch]++;
      increment2dHist(TRDParsingDigitLayerMismatch, mFEEID.supermodule * 2 + mHalfChamberSide[0], mStack[0], mLayer[0]);
    }
    if (currentsector != mSector[0] || currentsector != mSector[1]) {
      //sector mismatch, mDetector comes in from a construction via the feeid and ori.
      //count these
      //mEventRecord.ErrorStats[TRDParsingDigitSectorMisMatch]++;
      increment2dHist(TRDParsingDigitSectorMismatch, mFEEID.supermodule * 2 + mHalfChamberSide[0], mStack[0], mLayer[0]);
    }
    mSector[2] = currentsector; //from hc header treating it as authoritative
    mLayer[2] = currentlayer;
    mStack[2] = currentstack;
    mDetector[2] = mLayer[2] + mStack[2] * constants::NLAYER + mSector[2] * constants::NLAYER * constants::NSTACK;
    return 2;
  } else { // ignore the halfcahmber headers contents so use the rdh
    //take mDetector, layer and stack from the rdh/cru, we have those already assigned on entry to here
    return 0; // the index in mSector/mStack etc.
  }
}

int CruRawReader::parseDigitHCHeader()
{
  //mHBFoffset is the current offset into the current buffer,
  //
  uint32_t dhcheader = mHBFPayload[mHBFoffset32++];
  std::array<uint32_t, 4> headers{0};
  if (mByteSwap) {
    // byte swap if needed.
    o2::trd::HelperMethods::swapByteOrder(dhcheader);
  }
  mDigitHCHeader.word = dhcheader;
  if (mDigitHCHeader.major == 0 && mDigitHCHeader.minor == 0 && mDigitHCHeader.numberHCW == 0) {
    //hack this data into something resembling usable.
    mDigitHCHeader.major = mHalfChamberMajor;
    mDigitHCHeader.minor = 42; // to keep me entertained
    mDigitHCHeader.numberHCW = mHalfChamberWords;
    if (mHalfChamberWords == 0 || mHalfChamberMajor == 0) {
      //LOG(warn) << "we have a messed up halfchamber header and you have only set the halfchamber command line option to zero, hex dump of data and revisit what it should be.";
      // already in histograms
    }
  }

  int additionalHeaderWords = mDigitHCHeader.numberHCW;
  if (additionalHeaderWords >= 3) {
    increment2dHist(TRDParsingDigitHeaderCountGT3, mFEEID.supermodule * 2 + mHalfChamberSide[0], mStack[0], mLayer[0]);
    //TODO graph this and stats it
    if (mMaxErrsPrinted > 0) {
      LOG(alarm) << "Error parsing DigitHCHeader, too many additional words count=" << additionalHeaderWords;
      printDigitHCHeader(mDigitHCHeader, &headers[0]);
      checkNoErr();
    }
    return -1;
  }
  for (int headerwordcount = 0; headerwordcount < additionalHeaderWords; ++headerwordcount) {
    headers[headerwordcount] = mHBFPayload[mHBFoffset32++];
    if (mByteSwap) {
      // byte swap if needed.
      o2::trd::HelperMethods::swapByteOrder(headers[headerwordcount]);
    }
    switch (getDigitHCHeaderWordType(headers[headerwordcount])) {
      case 1: // header header1;
        mDigitHCHeader1.word = headers[headerwordcount];
        if (mDigitHCHeader1.res != 0x1) {
          //LOG(alarm) << "Digit HC Header 1 reserved : " << std::hex << mDigitHCHeader1.res << " raw: 0x" << mDigitHCHeader1.word;
          increment2dHist(TRDParsingDigitHeaderWrong1, mFEEID.supermodule * 2 + mHalfChamberSide[0], mStack[0], mLayer[0]);
        }
        break;
      case 2: // header header2;
        mDigitHCHeader2.word = headers[headerwordcount];
        if (mDigitHCHeader2.res != 0b110001) {
          // LOG(alarm) << "Digit HC Header 2 reserved : " << std::hex << mDigitHCHeader2.res << " raw: 0x" << mDigitHCHeader2.word;
          increment2dHist(TRDParsingDigitHeaderWrong2, mFEEID.supermodule * 2 + mHalfChamberSide[0], mStack[0], mLayer[0]);
        }
        break;
      case 3: // header header3;
        mDigitHCHeader3.word = headers[headerwordcount];
        if (mDigitHCHeader3.res != 0b110101) {
          // LOG(alarm) << "Digit HC Header 3 reserved : " << std::hex << mDigitHCHeader3.res << " raw: 0x" << mDigitHCHeader3.word;
          increment2dHist(TRDParsingDigitHeaderWrong3, mFEEID.supermodule * 2 + mHalfChamberSide[0], mStack[0], mLayer[0]);
        }
        break;
      default:
        //LOG(alarm) << "Error parsing DigitHCHeader at word:" << headerwordcount << " looking at 0x:" << std::hex << mHBFPayload[mHBFoffset32 - 1];
        increment2dHist(TRDParsingDigitHeaderWrong4, mFEEID.supermodule * 2 + mHalfChamberSide[0], mStack[0], mLayer[0]);
    }
  }
  if (mHeaderVerbose) {
    printDigitHCHeader(mDigitHCHeader, &headers[0]);
  }
  return 1;
}

void CruRawReader::updateLinkErrorGraphs(int currentlinkindex, int supermodule_half, int stack_layer)
{
  mEventRecords.incLinkErrorFlags(mDetector[0], mHalfChamberSide[0], stack_layer, mCurrentHalfCRULinkErrorFlags[currentlinkindex]);
  if (mRootOutput) {
    if (mCurrentHalfCRULinkErrorFlags[currentlinkindex] == 0) {
      ((TH2F*)mLinkErrors->At(0))->Fill(supermodule_half, stack_layer);
    }
    if (mCurrentHalfCRULinkErrorFlags[currentlinkindex] & 0x1) {
      ((TH2F*)mLinkErrors->At(1))->Fill(supermodule_half, stack_layer);
    }
    if (mCurrentHalfCRULinkErrorFlags[currentlinkindex] & 0x2) {
      ((TH2F*)mLinkErrors->At(2))->Fill(supermodule_half, stack_layer);
    }
    if (mCurrentHalfCRULinkErrorFlags[currentlinkindex] > 0) {
      ((TH2F*)mLinkErrors->At(3))->Fill(supermodule_half, stack_layer);
    }
    if (mCurrentHalfCRULinkLengths[currentlinkindex] > 0) {
      ((TH2F*)mLinkErrors->At(4))->Fill(supermodule_half, stack_layer);
      //mEventRecords.incLinkWords(mDetector[0], mHalfChamberSide[0], stack_layer, mCurrentHalfCRULinkLengths[currentlinkindex]);
    }
    if (mCurrentHalfCRULinkLengths[currentlinkindex] == 0) {
      ((TH2F*)mLinkErrors->At(5))->Fill(supermodule_half, stack_layer);
      mEventRecords.incLinkNoData(mDetector[0], mHalfChamberSide[0], stack_layer);
    }
  }
}

int CruRawReader::processHalfCRU(int cruhbfstartoffset)
{
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
  mDigitWordsRead = 0;
  mDigitWordsRejected = 0;
  mTrackletWordsRead = 0;
  mTrackletWordsRejected = 0;
  uint32_t cruwordsread = 9;
  //reject halfcru if it starts with padding words.
  //this should only hit that instance where the cru payload is a "blank event" of o2::trd::constants::CRUPADDING32
  if (mHBFPayload[cruhbfstartoffset] == o2::trd::constants::CRUPADDING32 && mHBFPayload[cruhbfstartoffset + 1] == o2::trd::constants::CRUPADDING32) {
    if (mVerbose) {
      LOG(info) << "blank rdh payload data at " << cruhbfstartoffset << ": 0x " << std::hex << mHBFPayload[cruhbfstartoffset] << " and 0x" << mHBFPayload[cruhbfstartoffset + 1];
    }
    mHBFoffset32 += 2;
    return 2;
  }
  if (mTotalHBFPayLoad == 0) {
    //empty payload
    return -1;
  }
  auto crustart = std::chrono::high_resolution_clock::now();
  // well then read the halfcruheader.
  memcpy((char*)&mCurrentHalfCRUHeader, (void*)(&mHBFPayload[cruhbfstartoffset]), sizeof(mCurrentHalfCRUHeader)); //TODO remove the copy just use pointer dereferencing, doubt it will improve the speed much though.

  o2::trd::getlinkdatasizes(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths);
  o2::trd::getlinkerrorflags(mCurrentHalfCRUHeader, mCurrentHalfCRULinkErrorFlags);
  mTotalHalfCRUDataLength256 = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
                                               mCurrentHalfCRULinkLengths.end(),
                                               decltype(mCurrentHalfCRULinkLengths)::value_type(0));
  mTotalHalfCRUDataLength = mTotalHalfCRUDataLength256 * 32; //convert to bytes.
  int mTotalHalfCRUDataLength32 = mTotalHalfCRUDataLength256 * 8; //convert to bytes.

  //can this half cru length fit into the available space of the rdh accumulated payload
  if (mTotalHalfCRUDataLength32 > mTotalHBFPayLoad - mHBFoffset32) {
    if (mMaxErrsPrinted > 0) {
      LOG(error) << "Next HalfCRU header says it contains more data than in the rdh payloads! " << mTotalHalfCRUDataLength32 << " < " << mTotalHBFPayLoad << "-" << mHBFoffset32;
      checkNoErr();
    }
    return -2;
  }
  if (halfCRUHeaderSanityCheck(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths, mCurrentHalfCRULinkErrorFlags)) {
    if (mMaxErrsPrinted > 0) {
      LOG(error) << "HalfCRU header failed sanity check";
      checkNoErr();
    }
    return -2;
  }

  //get eventrecord for event we are looking at
  mIR.bc = mCurrentHalfCRUHeader.BunchCrossing; // correct mIR to have the physics trigger bunchcrossing *NOT* the heartbeat trigger bunch crossing.
  InteractionRecord trdir(mIR);
  mCurrentEvent = &mEventRecords.getEventRecord(trdir);
  //check for cru errors :
  int linkerrorcounter = 0;
  for (auto& linkerror : mCurrentHalfCRULinkErrorFlags) {
    if (linkerror != 0) {
      if (mHeaderVerbose) {
        LOG(info) << "E link error FEEID:" << mFEEID.word << " CRUID:" << mCRUID << " Endpoint:" << mCRUEndpoint
                  << " on linkcount:" << linkerrorcounter++ << " errorval:0x" << std::hex << linkerror;
      }
    }
  }

  std::array<uint32_t, 1024>::iterator currentlinkstart = mHBFPayload.begin() + cruhbfstartoffset;
  if (mHeaderVerbose) {
    printHalfCRUHeader(mCurrentHalfCRUHeader);
    OutputHalfCruRawData();
  }
  std::array<uint32_t, 1024>::iterator linkstart, linkend;
  int dataoffsetstart32 = sizeof(mCurrentHalfCRUHeader) / 4 + cruhbfstartoffset; // in uint32
  //CHECK 1 does rdh endpoint match cru header end point.
  if (mCRUEndpoint != mCurrentHalfCRUHeader.EndPoint) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << " Endpoint mismatch : CRU Half chamber header endpoint = " << mCurrentHalfCRUHeader.EndPoint << " rdh end point = " << mCRUEndpoint;
      checkNoWarn();
    }
    if (mVerbose) {
      LOG(info) << "******* LINK # " << currentlinkindex;
    }
    //disaster dump the rest of this hbf
    return -2;
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
    mSector[0] = mFEEID.supermodule;
    mEndPoint[0] = mFEEID.endpoint;
    mSide[0] = mFEEID.side; // side of detector A/C
    int hbfoffsetatstartoflink = mHBFoffset32;
    //stack layer and side map to ori
    int oriindex = currentlinkindex + constants::NLINKSPERHALFCRU * mEndPoint[0]; // endpoint denotes the pci side, upper or lower for the pair of 15 fibres.
    FeeParam::unpackORI(oriindex, mSide[0], mStack[1], mLayer[1], mHalfChamberSide[1]);
    //sadly not all the data is redundant, probably a good thing, so stack and layer and halfchamber side is derived from the ori.
    mLayer[0] = mLayer[1];
    mStack[0] = mStack[1];
    mHalfChamberSide[0] = mHalfChamberSide[1];
    mSector[1] = oriindex / 30;
    mSide[1] = mSide[0];
    mDetector[0] = mStack[0] * constants::NLAYER + mLayer[0] + mSector[0] * constants::NLAYER * constants::NSTACK;
    mDetector[1] = mStack[1] * constants::NLAYER + mLayer[1] + mSector[1] * constants::NLAYER * constants::NSTACK;
    int supermodule_half = mSector[0] * 2 + mHalfChamberSide[0]; // will just go with the rdh one here its only for the hack graphing purposes.
    float stack_layer;
    stack_layer = mStack[0] * constants::NLAYER + mLayer[0]; // similarly this is also only for graphing so just use the rdh ones for now.
    updateLinkErrorGraphs(currentlinkindex, supermodule_half, stack_layer);

    mEventRecords.incLinkErrorFlags(mFEEID.supermodule, mHalfChamberSide[0], stack_layer, mCurrentHalfCRULinkErrorFlags[currentlinkindex]);
    currentlinksize = mCurrentHalfCRULinkLengths[currentlinkindex];
    // first parameter is the base of the link in the, so either the first or second part of the cru hence the *15
    mCurrentEvent->setDataPerLink((mFEEID.supermodule * 2 + mHalfChamberSide[0]) * 30 + currentlinkindex, currentlinksize);
    currentlinksize32 = currentlinksize * 8; //x8 to go from 256 bits to 32 bit;
    linkstart = mHBFPayload.begin() + dataoffsetstart32 + linksizeAccum32;
    linkend = linkstart + currentlinksize32;
    if (currentlinksize == 0) {
      mEventRecords.incLinkNoData(mDetector[0], mHalfChamberSide[0], stack_layer);
    }
    uint32_t linkzsum = 0;
    int dioffset = dataoffsetstart32 + linksizeAccum32;
    if (dioffset % 8 != 0) {
      if (mMaxErrsPrinted > 0) {
        LOG(error) << " we are not 256 bit aligned ... this should never happen";
        checkNoErr();
      }
    }
    if (mHBFoffset32 != std::distance(mHBFPayload.begin(), linkstart)) {
      mHBFoffset32 = std::distance(mHBFPayload.begin(), linkstart);
    }
    if (mHeaderVerbose) {
      LOG(info) << "Cru link :" << currentlinkindex << " raw dump before processing begin linkstart:" << std::hex << linkstart << " to " << linkend << " mHBFoffset32=" << std::dec << mHBFoffset32 << " and distance from start is : " << std::distance(mHBFPayload.begin(), linkstart);
      for (int dumpoffset = dataoffsetstart32 + linksizeAccum32; dumpoffset < dataoffsetstart32 + linksizeAccum32 + currentlinksize32; dumpoffset += 8) {
        LOGP(info, "0x{0:06x} :: {1:08x} {2:08x}  {3:08x} {4:08x} {5:08x} {6:08x} {7:08x} {8:08x} ", dumpoffset, HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 1]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 2]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 3]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 4]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 5]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 6]), HelperMethods::swapByteOrderreturn(mHBFPayload[dumpoffset + 7]));
      }
    }
    linksizeAccum32 += currentlinksize32;
    if (mDataVerbose) {
      LOG(info) << "******* LINK # " << currentlinkindex << " and  starting at " << mHBFoffset32 << " unpackORI(" << oriindex << "," << mSide[1] << "," << mStack[1] << "," << mLayer[1] << "," << mHalfChamberSide[1] << ") and an FEEID:" << std::hex << mFEEID.word << " det:" << std::dec << mDetector[1];
      LOG(info) << "******* LINK # " << currentlinkindex << " an FEEID:" << std::hex << mFEEID.word << " det:" << std::dec << mDetector[1] << " Error Flags : " << mCurrentHalfCRULinkErrorFlags[currentlinkindex];
    }
    if (linkstart != linkend) { // if link is not empty
      bool cleardigits = false; //linkstart and linkend already have the multiple cruheaderoffsets built in
      auto trackletparsingstart = std::chrono::high_resolution_clock::now();
      if (mHeaderVerbose) {
        LOG(info) << "*** Tracklet Parser : starting at " << std::hex << linkstart << " at hbfoffset: " << std::dec << mHBFoffset32 << " linkhbf start pos:" << hbfoffsetatstartoflink;
      }
      // for now we are using 0 i.e. from rdh FIXME figure out which is authoritative between rdh and ori tracklethcheader if we have it enabled.
      mTrackletWordsRead = mTrackletsParser.Parse(&mHBFPayload, linkstart, linkend, mFEEID, mHalfChamberSide[0], mDetector[0], mStack[0], mLayer[0], mCurrentEvent, &mEventRecords, mOptions, cleardigits, mTrackletHCHeaderState); // this will read up to the tracklet end marker.
      mTrackletWordsRejected = mTrackletsParser.getDataWordsDumped();
      std::chrono::duration<double, std::micro> trackletparsingtime = std::chrono::high_resolution_clock::now() - trackletparsingstart;
      mCurrentEvent->incTrackletTime((double)std::chrono::duration_cast<std::chrono::microseconds>(trackletparsingtime).count());
      if (mRootOutput) {
        mTrackletTiming->Fill((int)std::chrono::duration_cast<std::chrono::microseconds>(trackletparsingtime).count());
      }
      if (mHeaderVerbose) {
        LOG(info) << "trackletwordsread:" << mTrackletWordsRead << " trackletwordsrejected:" << mTrackletWordsRejected << "  mem copy with offset of : " << cruhbfstartoffset << " parsing with linkstart: " << linkstart << " ending at : " << linkend;
      }
      linkstart += mTrackletWordsRead + mTrackletWordsRejected;
      //now we have a tracklethcheader and a digithcheader.

      mHBFoffset32 += mTrackletWordsRead + mTrackletWordsRejected;
      mTotalTrackletsFound += mTrackletsParser.getTrackletsFound();
      mTotalTrackletWordsRejected += mTrackletWordsRejected;
      mTotalTrackletWordsRead += mTrackletWordsRead;
      mCurrentEvent->incWordsRead(mTrackletWordsRead);
      mCurrentEvent->incWordsRejected(mTrackletWordsRejected);
      mEventRecords.incLinkWordsRead(mFEEID.supermodule, mHalfChamberSide[0], stack_layer, mTrackletWordsRead);
      mEventRecords.incLinkWordsRejected(mFEEID.supermodule, mHalfChamberSide[0], stack_layer, mTrackletWordsRejected);
      if (mTrackletsParser.getTrackletParsingState()) {
        mHBFoffset32 += std::distance(linkstart, linkend);
        linkstart = linkend; // bail out as tracklet parsing bombed out. We are essentially lost.
      }
      if (mHeaderVerbose) {
        LOG(info) << "*** Tracklet Parser : trackletwordsread:" << mTrackletWordsRead << " ending " << std::hex << linkstart << " at hbfoffset: " << std::dec << mHBFoffset32;
      }

      /****************
      ** DIGITS NOW ***
      *****************/
      // Check if we have a calibration trigger ergo we do actually have digits data. check if we are now at the end of the data due to bugs, i.e. if trackletparsing read padding words.
      if (linkstart != linkend && (mCurrentHalfCRUHeader.EventType == o2::trd::constants::ETYPECALIBRATIONTRIGGER || mOptions[TRDIgnore2StageTrigger])) { // calibration trigger
        if (mHeaderVerbose) {
          LOG(info) << "*** Digit Parsing : starting at " << std::hex << linkstart << " at hbfoffset: " << std::dec << mHBFoffset32 << " linkhbf start pos:" << hbfoffsetatstartoflink;
        }
        // linkstart advanced all the way to the end due to trackletparser parsing crupadding words (known bug or feature )
        auto hfboffsetbeforehcparse = mHBFoffset32;
        //now read the digit half chamber header
        auto hcparse = parseDigitHCHeader();
        mWhichData = checkDigitHCHeader();
        //move over the DigitHCHeader mHBFoffset32 has already been moved in the reading.
        if (mHBFoffset32 - hfboffsetbeforehcparse != 1 + mDigitHCHeader.numberHCW) {
          if (mMaxErrsPrinted > 0) {
            LOG(alarm) << "Seems data offset is out of sync with number of HC Headers words " << mHBFoffset32 << "-" << hfboffsetbeforehcparse << "!=" << 1 << "+" << mDigitHCHeader.numberHCW;
            checkNoErr();
          }
        }
        if (hcparse == -1) {
          if (mMaxWarnPrinted > 0) {
            LOG(alarm) << "Parsing Digit HCHeader returned a -1";
            checkNoWarn();
          }
        } else {
          linkstart += 1 + mDigitHCHeader.numberHCW;
        }
        mEventRecords.incMajorVersion(mDigitHCHeader.major); // 127 is max histogram goes to 256

        if (mDigitHCHeader.major == 0x47) {
          // config event so ignore for now and bail out of parsing.
          //advance data pointers to the end;
          linkstart = linkend;
          mHBFoffset32 = std::distance(mHBFPayload.begin(), linkend); //currentlinksize-mTrackletWordsRead-sizeof(digitHCHeader)/4; // advance to the end of the link
          mTotalDigitWordsRejected += std::distance(linkstart + mTrackletWordsRead + sizeof(DigitHCHeader) / 4, linkend);
        } else {
            mDigitWordsRead = 0;
            auto digitsparsingstart = std::chrono::high_resolution_clock::now();
            //linkstart and linkend already have the multiple cruheaderoffsets built in
            mDigitWordsRead = mDigitsParser.Parse(&mHBFPayload, linkstart, linkend, mDetector[mWhichData], mStack[mWhichData], mLayer[mWhichData], mHalfChamberSide[mWhichData], mDigitHCHeader, mFEEID, currentlinkindex, mCurrentEvent, &mEventRecords, mOptions, cleardigits);
            std::chrono::duration<double, std::micro> digitsparsingtime = std::chrono::high_resolution_clock::now() - digitsparsingstart;
            if (mRootOutput) {
              mDigitTiming->Fill((int)std::chrono::duration_cast<std::chrono::microseconds>(digitsparsingtime).count());
            }
            mCurrentEvent->incDigitTime((double)std::chrono::duration_cast<std::chrono::microseconds>(digitsparsingtime).count());
            mDigitWordsRejected = mDigitsParser.getDumpedDataCount();
            mCurrentEvent->incWordsRead(mDigitWordsRead);
            mCurrentEvent->incWordsRejected(mDigitWordsRejected);
            mEventRecords.incLinkWordsRead(mFEEID.supermodule, mHalfChamberSide[0], stack_layer, mDigitWordsRead);
            mEventRecords.incLinkWordsRejected(mFEEID.supermodule, mHalfChamberSide[0], stack_layer, mDigitWordsRejected);
            if (mHeaderVerbose) {
              if (mDigitsParser.getDumpedDataCount() != 0) {
                LOG(info) << "FEEID: " << mFEEID.word << " LINK #" << oriindex << " bad datacount:" << mDigitsParser.getDataWordsParsed() << "::" << mDigitsParser.getDumpedDataCount();
              } else {
                LOG(info) << "FEEID: " << mFEEID.word << " LINK #" << oriindex << " good datacount:" << mDigitsParser.getDataWordsParsed() << "::" << mDigitsParser.getDumpedDataCount();
              }
            }
            mDigitWordsRejected = mDigitsParser.getDumpedDataCount();
            if (mHeaderVerbose) {
              if (mDigitsParser.getDumpedDataCount() != 0) {
                LOG(info) << "FEEID: " << mFEEID.word << " LINK #" << oriindex << " bad datacount:" << mDigitsParser.getDataWordsParsed() << "::" << mDigitsParser.getDumpedDataCount();
              } else {
                LOG(info) << "FEEID: " << mFEEID.word << " LINK #" << oriindex << " good datacount:" << mDigitsParser.getDataWordsParsed() << "::" << mDigitsParser.getDumpedDataCount();
              }
              mEventRecords.incParsingError(TRDParsingDigitStackMismatch, mFEEID.supermodule, mHalfChamberSide[0], mStack[0] * constants::NLAYER + mLayer[0]);
            }
            if (mDigitWordsRead + mDigitWordsRejected != std::distance(linkstart, linkend)) {
              //we have the data corruption problem of a pile of stuff at the end of a link, jump over it.
              if (mFixDigitEndCorruption) {
                mDigitWordsRead = std::distance(linkstart, linkend);
              } else {
                increment2dHist(TRDParsingDigitDataStillOnLink, (mFEEID.supermodule * 2 + mHalfChamberSide[0]) * 30, mStack[0], mLayer[0]);
                mEventRecords.incParsingError(TRDParsingDigitDataStillOnLink, mFEEID.supermodule, mHalfChamberSide[0], mStack[0] * constants::NLAYER + mLayer[0]);
              }
            }
            mTotalDigitsFound += mDigitsParser.getDigitsFound();
            if (mDataVerbose) {
              LOG(info) << "mDigitWordsRead : " << mDigitWordsRead << " mem copy with offset of : " << cruhbfstartoffset << " parsing digits with linkstart: " << linkstart << " ending at : " << linkend << " linkhbf start pos:" << hbfoffsetatstartoflink;
            }
            mHBFoffset32 += mDigitWordsRead + mDigitWordsRejected; // all 3 in 32bit units
            mTotalDigitWordsRead += mDigitWordsRead;
            mTotalDigitWordsRejected += mDigitWordsRejected;
          sumlinklengths += mCurrentHalfCRULinkLengths[currentlinkindex];
          sumtrackletwords += mTrackletWordsRead;
          sumdigitwords += mDigitWordsRead;

          if (mDigitWordsRejected > 0) {
            if (mRootOutput) {
              ((TH2F*)mLinkErrors->At(6))->Fill(supermodule_half, stack_layer);
            }
          } else if (mRootOutput) {
            ((TH2F*)mLinkErrors->At(7))->Fill(supermodule_half, stack_layer);
          }
        }
      }
    } else {
      if (mCurrentHalfCRUHeader.EventType == o2::trd::constants::ETYPEPHYSICSTRIGGER) {
        mEventRecords.incMajorVersion(128); // 127 is max histogram goes to 256
      }
    }
  } //for loop over link index.
  // we have read in all the digits and tracklets for this event.
  //digits and tracklets are sitting inside the parsing classes.
  //extract the vectors and copy them to tracklets and digits here, building the indexing(triggerrecords)
  //as this is for a single cru half chamber header all the tracklets and digits are for the same trigger defined by the bc and orbit in the rdh which we hold in mIR

  int lasttrigger = 0, lastdigit = 0, lasttracklet = 0;
  std::chrono::duration<double, std::milli> cruparsingtime = std::chrono::high_resolution_clock::now() - crustart;
  if (mRootOutput) {
    mCruTime->Fill((int)std::chrono::duration_cast<std::chrono::milliseconds>(cruparsingtime).count());
  }
  mCurrentEvent->incTime(cruparsingtime.count());

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
  //mStatCountersPerEvent.mLinkErrorFlag.fill(0);
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
  uint32_t totaldataread = 0;
  rewind();
  mTotalDigitWordsRead = 0;
  mTotalDigitWordsRejected = 0;
  mTotalTrackletWordsRead = 0;
  mTotalTrackletWordsRejected = 0;
  uint32_t* bufferptr;
  bufferptr = (uint32_t*)mDataBuffer;
  do {
    if (mDataVerbose) {
      LOG(info) << " mDataBuffer :" << (void*)mDataBuffer << " and offset to start on is :" << totaldataread;
    }
    mDatareadfromhbf = 0;
    auto goodprocessing = processHBFs(totaldataread, mVerbose);
    totaldataread += mDatareadfromhbf;
    if (!goodprocessing) {
      //processHBFs returned false, get out of here ...
      LOG(error) << "Error processing heart beat frame ... good luck";
      break;
    }
    if (totaldataread == 0) {
      if (mMaxWarnPrinted > 0) {
        LOG(warn) << "EEE  we read zero data but bailing out of here for now.";
        checkNoWarn();
      }
      break;
    }
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
void CruRawReader::buildDPLOutputs(o2::framework::ProcessingContext& pc)
{
  mEventRecords.sendData(pc, mOptions[TRDGenerateStats]);
  clearall(); // having now written the messages clear for next.
}

void CruRawReader::checkNoWarn()
{
  if (!mVerbose && --mMaxWarnPrinted == 0) {
    LOG(alarm) << "Warnings limit reached, the following ones will be suppressed";
  }
}

void CruRawReader::checkNoErr()
{
  if (!mVerbose && --mMaxErrsPrinted == 0) {
    LOG(error) << "Errors limit reached, the following ones will be suppressed";
  }
}

} // namespace o2::trd
