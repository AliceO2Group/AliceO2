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
#include "DataFormatsCTP/TriggerOffsetsParam.h"

#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <iomanip>

using namespace o2::trd::constants;

namespace o2::trd
{

void CruRawReader::configure(int tracklethcheader, int halfchamberwords, int halfchambermajor, std::bitset<16> options)
{
  mTrackletHCHeaderState = tracklethcheader;
  mHalfChamberWords = halfchamberwords;
  mHalfChamberMajor = halfchambermajor;
  mOptions = options;
  mTimeBins = TIMEBINS; // set to value from constants incase the DigitHCHeader1 header is not present.
  mPreviousDigitHCHeadersvnver = 0xffffffff;
  mPreviousDigitHCHeadersvnrver = 0xffffffff;
}

void CruRawReader::incrementErrors(int error, int halfChamberId, std::string message)
{
  mEventRecords.incParsingError(error, halfChamberId);
  if (mOptions[TRDVerboseErrorsBit]) {
    LOG(info) << "PE: " << o2::trd::ParsingErrorsString[error] << " hcid :: " << halfChamberId;
    if (message != "") {
      LOG(info) << message;
    }
  }
}

void CruRawReader::outputLinkRawData(int link)
{
  std::array<uint32_t, 1024>::iterator linkstart, linkend, bufferoffset;
  uint32_t lengthoflinksbefore = 0;
  int linkindex = 0;
  for (linkindex = 0; linkindex < link; ++linkindex) {
    lengthoflinksbefore += mCurrentHalfCRULinkLengths[linkindex];
  }
  lengthoflinksbefore *= 8;                                 // to get to 32 bit word units
  lengthoflinksbefore += sizeof(mCurrentHalfCRUHeader) / 4; // 8 bit words to 32 bit words
  linkstart = mHBFPayload.begin() + mHalfCRUStartOffset + lengthoflinksbefore;
  linkend = linkstart + mCurrentHalfCRULinkLengths[link] * 8;
  //mHalfCRUStartOffset=cruhbfstartoffset;
  bufferoffset = linkstart;
  uint32_t wordcount = 0;
  while (bufferoffset < linkend) {
    std::stringstream outputstring;
    uint32_t stringlength = 8;
    if (linkend - bufferoffset < 8) {
      stringlength = linkend - bufferoffset;
    }
    for (int z = 0; z < stringlength; ++z) {
      outputstring << "0x" << std::hex << std::setw(8) << std::setfill('0') << *(bufferoffset + z) << " ";
    }
    bufferoffset += 8;
    LOG(info) << wordcount++ << ":" << outputstring.str();
  }
}

bool CruRawReader::checkRDH(const o2::header::RDHAny* rdh)
{
  // first check for FEEID from unconfigured CRU
  TRDFeeID feeid;
  feeid.word = o2::raw::RDHUtils::getFEEID(rdh);
  if (((feeid.word) >> 4) == 0xfff) { // error condition is 0xfff? as the end point is known to the cru, but the rest is configured.
    if (mMaxErrsPrinted > 0) {
      LOG(error) << "RDH check failed due to 0xfff. FLP not configured, call TRD on call. Whole feeid = " << std::hex << (unsigned int)feeid.word;
      checkNoErr();
    }
    incrementErrors(FEEIDIsFFFF, -1, fmt::format("failed due to 0xfff? : {} whole feeid: {} ", feeid.word, (unsigned int)feeid.word));
    return false;
  }
  if (feeid.supermodule > 17) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "Wrong supermodule number " << std::dec << (int)feeid.supermodule << " detected in RDH. Whole feeid : " << std::hex << (unsigned int)feeid.word;
      checkNoWarn();
    }
    incrementErrors(FEEIDBadSector, -1, fmt::format("Wrong supermodule number {} detected in RDH whole feeid: {}", (int)feeid.supermodule, (unsigned int)feeid.word));
    return false;
  }
  if (o2::raw::RDHUtils::getMemorySize(rdh) <= 0) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "Received RDH header with invalid memory size (<= 0) ";
      checkNoWarn();
    }
    incrementErrors(BadRDHMemSize, -1, fmt::format("Received RDH header with invalid memory size (<= 0) "));
    return false;
  }
  return true;
}

bool CruRawReader::compareRDH(const o2::header::RDHAny* rdhPrev, const o2::header::RDHAny* rdhCurr)
{
  if (o2::raw::RDHUtils::getFEEID(rdhPrev) != o2::raw::RDHUtils::getFEEID(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH FEEID are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(BadRDHFEEID);
    return false;
  }
  if (o2::raw::RDHUtils::getEndPointID(rdhPrev) != o2::raw::RDHUtils::getEndPointID(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  EndPointID are not identical in rdh.";
      checkNoWarn();
    }
    if (mOptions[TRDVerboseErrorsBit]) {
      LOG(error) << "ERDH  EndPointID are not identical in rdh.";
    }
    incrementErrors(BadRDHEndPoint);
    return false;
  }
  if (o2::raw::RDHUtils::getTriggerOrbit(rdhPrev) != o2::raw::RDHUtils::getTriggerOrbit(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  Orbit are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(BadRDHOrbit);
    return false;
  }
  if (o2::raw::RDHUtils::getCRUID(rdhPrev) != o2::raw::RDHUtils::getCRUID(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  CRUID are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(BadRDHCRUID);
    return false;
  }
  if (o2::raw::RDHUtils::getPacketCounter(rdhPrev) == o2::raw::RDHUtils::getPacketCounter(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  PacketCounters are not sequential in rdh.";
      checkNoWarn();
    }
    incrementErrors(BadRDHPacketCounter);
    return false;
  }
  return true;
}

int CruRawReader::processHBFs()
{
  const o2::header::RDHAny* rdh = reinterpret_cast<const o2::header::RDHAny*>(mCurrRdhPtr);
  auto rdhPrevious = rdh;
  bool firstRdh = true;
  uint32_t totalDataInputSize = 0;
  mTotalHBFPayLoad = 0;

  // loop until RDH stop header
  while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
    if (mOptions[TRDVerboseBit]) {
      LOG(info) << "Current RDH is as follows:";
      o2::raw::RDHUtils::printRDH(rdh);
    }
    if (!checkRDH(rdh)) {
      return -1;
    }
    if (!firstRdh && !compareRDH(rdhPrevious, rdh)) {
      // previous and current RDHs are inconsistent, this should not happen
      return -1;
    }
    rdhPrevious = rdh;
    firstRdh = false;
    auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
    auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
    auto rdhpayload = memorySize - headerSize;
    mFEEID.word = o2::raw::RDHUtils::getFEEID(rdh);       //TODO change this and just carry around the curreht RDH
    mCRUEndpoint = o2::raw::RDHUtils::getEndPointID(rdh); // the upper or lower half of the currently parsed cru 0-14 or 15-29
    mCRUID = o2::raw::RDHUtils::getCRUID(rdh);
    mIR = o2::raw::RDHUtils::getTriggerIR(rdh);

    // copy the contents of the current RDH into the buffer to be parsed, RDH payload is memory size minus header size
    std::memcpy((char*)&mHBFPayload[0] + mTotalHBFPayLoad, ((char*)rdh) + headerSize, rdhpayload);
    // copy the contents of the current rdh into the buffer to be parsed
    mTotalHBFPayLoad += rdhpayload;
    totalDataInputSize += offsetToNext;
    // move to next rdh
    rdh = reinterpret_cast<const o2::header::RDHAny*>(reinterpret_cast<const char*>(rdh) + offsetToNext);
    // increment the data pointer by the size of the next RDH.
    mCurrRdhPtr = reinterpret_cast<const char*>(rdh) + offsetToNext;

    if (mOptions[TRDVerboseWordBit]) {
      LOG(info) << "Next RDH is as follows:";
      o2::raw::RDHUtils::printRDH(rdh);
    }

    if (!o2::raw::RDHUtils::getStop(rdh) && offsetToNext >= mCurrRdhPtr - mDataBufferPtr) {
      // we can still copy into this buffer.
      if (mMaxWarnPrinted > 0) {
        LOGP(warn, "RDH offsetToNext = {} is larger than it can possibly be. Remaining data in the buffer = {}", offsetToNext, mCurrRdhPtr - mDataBufferPtr);
        checkNoWarn();
      }
      return -1;
    }
  }

  // at this point the entire HBF data payload is sitting in mHBFPayload and the total data count is mTotalHBFPayLoad
  int iteration = 0;
  mHBFoffset32 = 0;
  while ((mHBFoffset32 < ((mTotalHBFPayLoad) / 4))) {
    if (mOptions[TRDVerboseBit]) {
      LOGP(info, "Current half-CRU iteration {}, current offset in the HBF payload {}, total payload in number of 32-bit words {}", iteration, mHBFoffset32, mTotalHBFPayLoad / 4);
    }
    int halfcruprocess = processHalfCRU(iteration);

    if (halfcruprocess == -2) {
      //dump rest of this rdh payload, something screwed up.
      break;
    }
    iteration++;
  } // loop of halfcru's while there is still data in the heart beat frame.

  return totalDataInputSize;
}

void CruRawReader::checkDigitHCHeader(int halfChamberIdRef)
{
  // compare the half chamber ID from the digit HC header with the reference one obtained from the link ID
  int halfChamberIdHeader = mDigitHCHeader.supermodule * NHCPERSEC + mDigitHCHeader.stack * NLAYER * 2 + mDigitHCHeader.layer * 2 + mDigitHCHeader.side;

  if (!mOptions[TRDIgnoreDigitHCHeaderBit]) {
    if (halfChamberIdHeader != halfChamberIdRef) {
      incrementErrors(DigitHCHeaderMismatch, halfChamberIdRef, fmt::format("HCID mismatch in Digit HCHeader : halfChamberIdHeader: {} halfChamberId : {}", halfChamberIdHeader, halfChamberIdRef));
      if (mOptions[TRDVerboseErrorsBit]) {
      }
    }
  }
}

int CruRawReader::parseDigitHCHeader(int hcid)
{
  // mHBFoffset32 is the current offset into the current buffer,
  //
  mDigitHCHeader.word = mHBFPayload[mHBFoffset32++];
  std::array<uint32_t, 4> headers{0};
  if (mOptions[TRDByteSwapBit]) {
    // byte swap if needed.
    o2::trd::HelperMethods::swapByteOrder(mDigitHCHeader.word);
  }
  if (mDigitHCHeader.major == 0 && mDigitHCHeader.minor == 0 && mDigitHCHeader.numberHCW == 0) {
    //hack this data into something resembling usable.
    mDigitHCHeader.major = mHalfChamberMajor;
    mDigitHCHeader.minor = 42;
    mDigitHCHeader.numberHCW = mHalfChamberWords;
    if (mHalfChamberWords == 0 || mHalfChamberMajor == 0) {
      if (mMaxWarnPrinted > 0) {
        LOG(warn) << "halfchamber header is corrupted and you have only set the halfchamber command line option to zero, hex dump of data and revisit what it should be.";
        checkNoWarn();
      }
      // already in histograms
    }
  }

  int additionalHeaderWords = mDigitHCHeader.numberHCW;
  if (additionalHeaderWords >= 3) {
    incrementErrors(DigitHeaderCountGT3, hcid);
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "Error parsing DigitHCHeader, too many additional words count=" << additionalHeaderWords << " header:" << std::hex << mDigitHCHeader.word;
      //printDigitHCHeader(mDigitHCHeader, &headers[0]);
      checkNoWarn();
    }
    return -1;
  }
  std::bitset<3> headersfound;

  for (int headerwordcount = 0; headerwordcount < additionalHeaderWords; ++headerwordcount) {
    headers[headerwordcount] = mHBFPayload[mHBFoffset32++];
    if (mOptions[TRDByteSwapBit]) {
      // byte swap if needed.
      o2::trd::HelperMethods::swapByteOrder(headers[headerwordcount]);
    }
    switch (getDigitHCHeaderWordType(headers[headerwordcount])) {
      case 1: // header header1;
        if (headersfound.test(0)) {
          // we have a problem, we already have a Digit HC Header1, we are lost.
          if (mOptions[TRDVerboseErrorsBit]) {
            LOG(info) << "We have a >1 Digit HC Header 1  : " << std::hex << " raw: 0x" << headers[headerwordcount];
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHCHeader1Problem, hcid);
        }
        mDigitHCHeader1.word = headers[headerwordcount];
        headersfound.set(0);
        if (mDigitHCHeader1.res != 0x1) {
          if (mOptions[TRDVerboseErrorsBit]) {
            LOG(info) << "Digit HC Header 1 reserved : 0x" << std::hex << mDigitHCHeader1.res << " raw: 0x" << mDigitHCHeader1.word;
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHeaderWrong1, hcid);
        }
        if ((mDigitHCHeader1.numtimebins > TIMEBINS) || (mDigitHCHeader1.numtimebins < 3)) {
          if (mMaxWarnPrinted > 0) {
            LOG(warn) << "Time bins in Digit HC Header 1 is " << mDigitHCHeader1.numtimebins
                      << " this is absurd";
            checkNoWarn();
          }
          return -1;
        }
        mTimeBins = mDigitHCHeader1.numtimebins;
        break;
      case 2: // header header2;
        if (headersfound.test(1)) {
          // we have a problem, we already have a Digit HC Header2, we are hereby lost.
          if (mOptions[TRDVerboseErrorsBit]) {
            LOG(info) << "We have a >1 Digit HC Header 2  : " << std::hex << " raw: 0x" << headers[headerwordcount];
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHCHeader2Problem);
        }
        mDigitHCHeader2.word = headers[headerwordcount];
        headersfound.set(1);
        if (mDigitHCHeader2.res != 0b110001) {
          if (mOptions[TRDVerboseErrorsBit]) {
            LOG(info) << "Digit HC Header 2 reserved : " << std::hex << mDigitHCHeader2.res << " raw: 0x" << mDigitHCHeader2.word;
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHeaderWrong2, hcid);
        }
        break;
      case 3: // header header3;
        if (headersfound.test(2)) {
          // we have a problem, we already have a Digit HC Header2, we are hereby lost.
          if (mOptions[TRDVerboseErrorsBit]) {
            LOG(info) << "We have a >1 Digit HC Header 2  : " << std::hex << " raw: 0x" << headers[headerwordcount];
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHCHeader3Problem, hcid);
        }
        mDigitHCHeader3.word = headers[headerwordcount];
        headersfound.set(2);
        if (mDigitHCHeader3.res != 0b110101) {
          if (mOptions[TRDVerboseErrorsBit]) {
            LOG(info) << "Digit HC Header 3 reserved : " << std::hex << mDigitHCHeader3.res << " raw: 0x" << mDigitHCHeader3.word;
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHeaderWrong3, hcid);
        }
        if (mPreviousDigitHCHeadersvnver != 0xffffffff &&
            mPreviousDigitHCHeadersvnrver != 0xffffffff) {
          if ((mDigitHCHeader3.svnver != mPreviousDigitHCHeadersvnver) &&
              (mDigitHCHeader3.svnrver != mPreviousDigitHCHeadersvnrver)) {
            if (mMaxWarnPrinted > 0) {
              checkNoWarn();
            }
            if (mOptions[TRDVerboseErrorsBit]) {
              LOG(info) << "Digit HC Header 3 svn ver : " << std::hex << mDigitHCHeader3.svnver << " svn release ver : 0x" << mDigitHCHeader3.svnrver;
              printDigitHCHeader(mDigitHCHeader, headers.data());
            }
            incrementErrors(DigitHCHeaderSVNMismatch, hcid);
            return -1;
          } else {
            // this is the first time seeing a DigitHCHeader3
            mPreviousDigitHCHeadersvnver = mDigitHCHeader3.svnver;
            mPreviousDigitHCHeadersvnrver = mDigitHCHeader3.svnrver;
          }
        }
        break;
      default:
        if (mOptions[TRDVerboseErrorsBit]) {
          LOG(info) << " unknown error in switch staement for Digit HC Header";
          printDigitHCHeader(mDigitHCHeader, headers.data());
        }
        incrementErrors(DigitHeaderWrong4, hcid);
    }
  }
  if (mOptions[TRDVerboseBit]) {
    printDigitHCHeader(mDigitHCHeader, &headers[0]);
  }

  return 1;
}

int CruRawReader::processHalfCRU(int iteration)
{
  // process a halfcru
  mHalfCRUStartOffset = mHBFoffset32; // store the start of the halfcru we are currently on.
  // this should only hit that instance where the cru payload is a "blank event" of CRUPADDING32
  if (mHBFPayload[mHBFoffset32] == CRUPADDING32) {
    if (mOptions[TRDVerboseBit]) {
      LOG(info) << "blank rdh payload data at " << mHBFoffset32 << ": 0x " << std::hex << mHBFPayload[mHBFoffset32] << " and 0x" << mHBFPayload[mHBFoffset32 + 1];
    }
    int loopcount = 0;
    while (mHBFPayload[mHBFoffset32] == CRUPADDING32 && loopcount < 8) { // can only ever be an entire 256 bit word hence a limit of 8 here.
      // TODO: check with Guido if it could not actually be more padding words
      mHBFoffset32++;
      mWordsAccepted++;
      loopcount++;
    }
    return 2;
  }

  auto crustart = std::chrono::high_resolution_clock::now();

  memcpy(&mCurrentHalfCRUHeader, &(mHBFPayload[mHBFoffset32]), sizeof(HalfCRUHeader));
  mHBFoffset32 += sizeof(HalfCRUHeader) / 4; // advance past the header.
  if (mOptions[TRDVerboseWordBit]) {
    //output the cru half chamber header : raw/parsed
    dumpHalfCRUHeader(mCurrentHalfCRUHeader);
  }

  o2::trd::getHalfCRULinkDataSizes(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths);
  o2::trd::getHalfCRULinkErrorFlags(mCurrentHalfCRUHeader, mCurrentHalfCRULinkErrorFlags);
  uint32_t totalHalfCRUDataLength256 = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
                                                       mCurrentHalfCRULinkLengths.end(),
                                                       0U);
  uint32_t totalHalfCRUDataLength32 = totalHalfCRUDataLength256 * 8; // convert to 32-bit words

  // in the interests of descerning real corrupt halfcruheaders
  // from the sometimes garbage at the end of a half cru
  // if the first word is clearly garbage assume garbage and not a corrupt halfcruheader.
  if (iteration > 0) {
    if (mCurrentHalfCRUHeader.EndPoint != mPreviousHalfCRUHeader.EndPoint) {
      if (mMaxWarnPrinted > 0) {
        LOGF(info, "For current half-CRU index %i we have end point %i, while the previous end point was %i", iteration, mCurrentHalfCRUHeader.EndPoint, mPreviousHalfCRUHeader.EndPoint);
        checkNoWarn();
      }
      incrementErrors(HalfCRUCorrupt);
      mWordsRejected += totalHalfCRUDataLength32;
      return -2;
    }
    // event type can change wit in a
    if (mCurrentHalfCRUHeader.StopBit != mPreviousHalfCRUHeader.StopBit) {
      if (mMaxWarnPrinted > 0) {
        LOGF(info, "For current half-CRU index %i we have stop bit %i, while the previous stop bit was %i", iteration, mCurrentHalfCRUHeader.StopBit, mPreviousHalfCRUHeader.StopBit);
        checkNoWarn();
      }
      incrementErrors(HalfCRUCorrupt);
      mWordsRejected += totalHalfCRUDataLength32;
      return -2;
    }
  }
  mPreviousHalfCRUHeader = mCurrentHalfCRUHeader;

  //can this half cru length fit into the available space of the rdh accumulated payload
  if (totalHalfCRUDataLength32 > (mTotalHBFPayLoad / 4) - mHBFoffset32) {
    if (mMaxWarnPrinted > 0) {
      LOGP(warn, "HalfCRU header says it contains more data ({} 32-bit words) than is remaining in the payload ({} 32-bit words)", totalHalfCRUDataLength32, ((mTotalHBFPayLoad / 4) - mHBFoffset32));
      checkNoWarn();
    }
    incrementErrors(HalfCRUSumLength);
    mWordsRejected += (mTotalHBFPayLoad / 4) - mHBFoffset32;
    return -2;
  }
  if (!halfCRUHeaderSanityCheck(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths,
                                mCurrentHalfCRULinkErrorFlags)) {
    // let incrementErrors catch the undefined values of sector side stack and layer as if not
    // set it will go so zero in the method, however if set, it means this is the second half
    // cru header, and we have the values from the last one we read which
    // *SHOULD* be the same as this halfcruheader.
    incrementErrors(HalfCRUCorrupt, -1, fmt::format("HalfCRU header failed sanity check for FEEID with {:#x} ", (unsigned int)mFEEID.word));
    mWordsRejected += (mTotalHBFPayLoad / 4) - mHBFoffset32;
    return -2;
  }

  //get eventrecord for event we are looking at
  mIR.bc = mCurrentHalfCRUHeader.BunchCrossing; // correct mIR to have the physics trigger
                                                // bunchcrossing *NOT* the heartbeat trigger
                                                // bunch crossing.

  if (o2::ctp::TriggerOffsetsParam::Instance().LM_L0 > (int)mIR.bc) {
    // applying the configured BC shift would lead to negative BC, hence we reject this trigger
    // dump to the end of this cruhalfchamberheader
    // data to dump is totalHalfCRUDataLength32
    mHBFoffset32 += totalHalfCRUDataLength32;   // go to the end of this halfcruheader and payload.
    mWordsRejected += totalHalfCRUDataLength32; // add the rejected data to the accounting;
    if (mOptions[TRDVerboseErrorsBit]) {
    }
    incrementErrors(HalfCRUBadBC, -1, fmt::format("Bunchcrossing from previous orbit, is negative after shift and data is being rejected LM_L0:{} bc:{}", o2::ctp::TriggerOffsetsParam::Instance().LM_L0, (int)mIR.bc));
    return 0; // nothing particularly wrong with the data, we just dont want it, as a trigger problem
  } else {
    // apply CTP offset shift
    mIR.bc -= o2::ctp::TriggerOffsetsParam::Instance().LM_L0;
  }
  if (mOptions[TRDOnlyCalibrationTriggerBit] &&
      mCurrentHalfCRUHeader.EventType == o2::trd::constants::ETYPEPHYSICSTRIGGER) {
    mHBFoffset32 += totalHalfCRUDataLength32; // go to the end of this halfcruheader and payload
    return 1;                                 // we dont want the physics triggers to polute the
                                              // logging, possibly other reasons
  }
  InteractionRecord trdir(mIR);
  mCurrentEvent = &mEventRecords.getEventRecord(trdir);

  // check for cru errors :
  //  TODO make this check configurable? Or do something in case of error flags set?
  int linkerrorcounter = 0;
  for (auto& linkerror : mCurrentHalfCRULinkErrorFlags) {
    if (linkerror != 0) {
      if (mOptions[TRDVerboseBit]) {
        LOG(info) << "E link error FEEID:" << mFEEID.word << " CRUID:" << mCRUID << " Endpoint:" << mCRUEndpoint << " on linkcount:" << linkerrorcounter++ << " errorval:0x" << std::hex << linkerror;
      }
    }
  }

  if (mOptions[TRDVerboseHalfCruBit]) {
    dumpInputPayload();
  }

  //CHECK 1 does rdh endpoint match cru header end point.
  if (mCRUEndpoint != mCurrentHalfCRUHeader.EndPoint) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << " Endpoint mismatch : CRU Half chamber header endpoint = "
                << mCurrentHalfCRUHeader.EndPoint << " rdh end point = " << mCRUEndpoint;
      checkNoWarn();
    }
    //disaster dump the rest of this hbf
    return -2;
  }

  //loop over links
  uint32_t linksizeAccum32 = 0;     // accumulated size of all links in 32-bit words
  auto hbfOffsetTmp = mHBFoffset32; // store current position at the beginning of the half-CRU payload data
  for (int currentlinkindex = 0; currentlinkindex < NLINKSPERHALFCRU; currentlinkindex++) {
    int cruIdx = mFEEID.supermodule * 2 + mFEEID.side;                    // 2 CRUs per SM, side defining A/C-side CRU
    int halfCruIdx = cruIdx * 2 + mFEEID.endpoint;                        // endpoint (0 or 1) defines half-CRU
    int linkIdxGlobal = halfCruIdx * NLINKSPERHALFCRU + currentlinkindex; // global link ID [0..1079]
    int halfChamberId = HelperMethods::getHCIDFromLinkID(linkIdxGlobal);
    // TODO we keep detector, stack, layer, sector, side for now to be compatible to the current code state,
    // but halfChamberId contains everything we need to know... More cleanup to be done in second step
    int detectorId = halfChamberId / 2;
    int stack = HelperMethods::getStack(detectorId);
    int layer = HelperMethods::getLayer(detectorId);
    int sector = HelperMethods::getSector(detectorId);
    int side = halfChamberId % 2;
    int stack_layer = stack * NLAYER + layer; // similarly this is also only for graphing so just use the rdh ones for now.
    mEventRecords.incLinkErrorFlags(mFEEID.supermodule, side, stack_layer, mCurrentHalfCRULinkErrorFlags[currentlinkindex]);
    uint32_t currentlinksize32 = mCurrentHalfCRULinkLengths[currentlinkindex] * 8; // x8 to go from 256 bits to 32 bit;
    std::array<uint32_t, 1024>::iterator linkstart = mHBFPayload.begin() + mHBFoffset32;
    std::array<uint32_t, 1024>::iterator linkend = linkstart + currentlinksize32;
    linksizeAccum32 += currentlinksize32;
    if (currentlinksize32 == 0) {
      mEventRecords.incLinkNoData(detectorId, side, stack_layer);
    }
    if (mOptions[TRDVerboseErrorsBit]) {
      if (currentlinksize32 > 0) {
        LOGF(info, "Half-CRU link %i raw dump before parsing starts:", currentlinkindex);
        for (uint32_t dumpoffset = mHBFoffset32; dumpoffset < mHBFoffset32 + currentlinksize32; dumpoffset += 8) {
          LOGF(info, "0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x", mHBFPayload[dumpoffset], mHBFPayload[dumpoffset + 1], mHBFPayload[dumpoffset + 2], mHBFPayload[dumpoffset + 3], mHBFPayload[dumpoffset + 4], mHBFPayload[dumpoffset + 5], mHBFPayload[dumpoffset + 6], mHBFPayload[dumpoffset + 7]);
        }
      } else {
        LOGF(info, "Half-CRU link %i has zero link size", currentlinkindex);
      }
    }
    if (linkstart != linkend) { // if link is not empty
      auto trackletparsingstart = std::chrono::high_resolution_clock::now();
      if (mOptions[TRDVerboseErrorsBit]) {
        LOGF(info, "Tracklet parser starting at offset %u and processing up to %u words", mHBFoffset32, currentlinksize32);
      }
      // for now we are using 0 i.e. from rdh FIXME figure out which is authoritative between rdh and ori tracklethcheader if we have it enabled.
      int trackletWordsRead = mTrackletsParser.Parse(&mHBFPayload, linkstart, linkend, mFEEID, side, detectorId, stack, layer, mCurrentEvent, &mEventRecords, mOptions, mTrackletHCHeaderState); // this will read up to the tracklet end marker.
      if (mTrackletsParser.dumpLink()) {
        //dump the link that cause the error
        // the call to dumpLink resets the boolean to false;
        outputLinkRawData(currentlinkindex);
      }
      if (trackletWordsRead == -1) {
        //something went wrong bailout of here.
        if (mMaxErrsPrinted > 0) {
          LOG(warn) << "TrackletParser returned -1 for  LINK # " << currentlinkindex << " an FEEID:" << std::hex << mFEEID.word << " det:" << std::dec << detectorId << " is > the lenght stored in the cruhalfchamber header : " << mCurrentHalfCRULinkLengths[currentlinkindex];
          checkNoErr();
        }
        incrementErrors(TrackletsReturnedMinusOne, halfChamberId);
        return -2;
      }
      int trackletWordsDumped = mTrackletsParser.getDataWordsDumped();
      std::chrono::duration<double, std::micro> trackletparsingtime = std::chrono::high_resolution_clock::now() - trackletparsingstart;
      mCurrentEvent->incTrackletTime((double)std::chrono::duration_cast<std::chrono::microseconds>(trackletparsingtime).count());
      if (mOptions[TRDVerboseBit]) {
        LOG(info) << "trackletwordsread:" << trackletWordsRead << " trackletwordsrejected:" << trackletWordsDumped << " parsing with linkstart: " << linkstart << " ending at : " << linkend;
      }
      linkstart += trackletWordsRead + trackletWordsDumped;
      //now we have a tracklethcheader and a digithcheader.

      mHBFoffset32 += trackletWordsRead + trackletWordsDumped;
      mTotalTrackletsFound += mTrackletsParser.getTrackletsFound();
      mTotalTrackletWordsRejected += trackletWordsDumped;
      mTotalTrackletWordsRead += trackletWordsRead;
      mCurrentEvent->incWordsRead(trackletWordsRead);
      mCurrentEvent->incWordsRejected(trackletWordsDumped);
      mEventRecords.incLinkWordsRead(mFEEID.supermodule, side, stack_layer, trackletWordsRead);
      mEventRecords.incLinkWordsRejected(mFEEID.supermodule, side, stack_layer, trackletWordsDumped);
      if (mOptions[TRDVerboseBit]) {
        LOG(info) << "*** Tracklet Parser : trackletwordsread:" << trackletWordsRead << " ending "
                  << std::hex << linkstart << " at hbfoffset: " << std::dec << mHBFoffset32;
      }
      if (mTrackletsParser.getTrackletParsingState()) {
        LOGF(info, "Tracklet parser is in a bad state");
        mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
        continue; // move to next link of this half-CRU
      }

      /****************
      ** DIGITS NOW ***
      *****************/
      // Check if we have a calibration trigger ergo we do actually have digits data. check if we are now at the end of the data due to bugs, i.e. if trackletparsing read padding words.
      if (linkstart != linkend &&
          (mCurrentHalfCRUHeader.EventType == ETYPECALIBRATIONTRIGGER || mOptions[TRDIgnore2StageTrigger]) &&
          (mHBFPayload[mHBFoffset32] != CRUPADDING32)) {
        // calibration trigger and insure we dont come in here if we are on a padding word.
        if (mOptions[TRDVerboseBit]) {
          LOG(info) << "*** Digit Parsing : starting at " << std::hex << linkstart
                    << " at hbfoffset: " << std::dec << mHBFoffset32;
        }

        uint32_t hfboffsetbeforehcparse = mHBFoffset32;
        //now read the digit half chamber header
        auto hcparse = parseDigitHCHeader(halfChamberId);
        if (hcparse != 1) {
          //disaster dump the rest of this hbf
          return -2;
        }
        checkDigitHCHeader(halfChamberId);
        //move over the DigitHCHeader mHBFoffset32 has already been moved in the reading.
        if (mHBFoffset32 - hfboffsetbeforehcparse != 1U + mDigitHCHeader.numberHCW) {
          if (mMaxErrsPrinted > 0) {
            LOG(warn) << "Seems data offset is out of sync with number of HC Headers words "
                      << mHBFoffset32 << "-" << hfboffsetbeforehcparse << "!=" << 1 << "+"
                      << mDigitHCHeader.numberHCW;
            checkNoErr();
          }
        }
        if (hcparse == -1) {
          if (mMaxWarnPrinted > 0) {
            LOG(warn) << "Parsing Digit HCHeader returned a -1";
            checkNoWarn();
          }
        } else {
          linkstart += 1 + mDigitHCHeader.numberHCW;
        }
        mEventRecords.incMajorVersion(mDigitHCHeader.major); // 127 is max histogram goes to 256

        if (mDigitHCHeader.major == 0x47) {
          // config event so ignore for now and bail out of parsing.
          //advance data pointers to the end;
          mHBFoffset32 = hbfOffsetTmp + currentlinksize32;
          // linkstart points to beginning of digits data, need to add the HC header words to it
          mTotalDigitWordsRejected += std::distance(linkstart + 1U + mDigitHCHeader.numberHCW, linkend);
          LOG(info) << "Configuration event  ";
        } else {
          auto digitsparsingstart = std::chrono::high_resolution_clock::now();
          // linkstart and linkend already have the multiple cruheaderoffsets built in
          int digitWordsRead = mDigitsParser.Parse(&mHBFPayload, linkstart, linkend, detectorId, stack, layer, side, mDigitHCHeader, mTimeBins, mFEEID, currentlinkindex, mCurrentEvent, &mEventRecords, mOptions);
          std::chrono::duration<double, std::micro> digitsparsingtime = std::chrono::high_resolution_clock::now() - digitsparsingstart;
          mCurrentEvent->incDigitTime((double)std::chrono::duration_cast<std::chrono::microseconds>(digitsparsingtime).count());
          int digitWordsRejected = mDigitsParser.getDumpedDataCount();
          mCurrentEvent->incWordsRead(digitWordsRead);
          mCurrentEvent->incWordsRejected(digitWordsRejected);
          mEventRecords.incLinkWordsRead(mFEEID.supermodule, side, stack_layer, digitWordsRead);
          mEventRecords.incLinkWordsRejected(mFEEID.supermodule, side, stack_layer, digitWordsRejected);

          if (mOptions[TRDVerboseBit]) {
            LOGF(info, "FEEID: 0x%8x, LINK # %i, digit words parsed %i, digit words dumped %i", mFEEID.word, linkIdxGlobal, digitWordsRead, digitWordsRejected);
          }
          if (mDigitsParser.dumpLink()) {
            //dump the link that cause the error
            // the call to dumpLink resets the boolean to false;
            outputLinkRawData(currentlinkindex);
          }
          if (mOptions[TRDVerboseBit]) {
            LOG(info) << "FEEID: " << mFEEID.word << " LINK #" << linkIdxGlobal << " bad datacount:"
                      << mDigitsParser.getDataWordsParsed() << "::" << mDigitsParser.getDumpedDataCount();
          }
          if (digitWordsRead + digitWordsRejected != std::distance(linkstart, linkend)) {
            // we have the data corruption problem of a pile of stuff at the end of a link, jump over it.
            if (mFixDigitEndCorruption) {
              digitWordsRead = std::distance(linkstart, linkend);
            } else {
              incrementErrors(DigitDataStillOnLink, halfChamberId);
              if (mOptions[TRDVerboseErrorsBit]) {
                LOG(info) << "FEEID: " << mFEEID.word << " LINK #" << linkIdxGlobal << " data still on link ";
              }
            }
          }
          mTotalDigitsFound += mDigitsParser.getDigitsFound();
          mHBFoffset32 += digitWordsRead + digitWordsRejected; // all 3 in 32bit units
          mTotalDigitWordsRead += digitWordsRead;
          mTotalDigitWordsRejected += digitWordsRejected;
        }
      }
    } else {
      if (mCurrentHalfCRUHeader.EventType == ETYPEPHYSICSTRIGGER) {
        mEventRecords.incMajorVersion(128); // 127 is max histogram goes to 256
      }
    }
    if (mHBFoffset32 != hbfOffsetTmp + linksizeAccum32) {
      // if only tracklet data is available the tracklet parser advances to the tracklet end marker, but there can still be padding words on the link
      LOGF(debug, "After processing link %i the payload offset of %u is not the expected %u + %u", currentlinkindex, mHBFoffset32, hbfOffsetTmp, linksizeAccum32);
      mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
    }
  } // for loop over link index.
  if (mHBFoffset32 != hbfOffsetTmp + totalHalfCRUDataLength32) {
    LOGF(debug, "After processing half-CRU data the offset (%u) is not pointing to the expected position (%u + %u = %u)", mHBFoffset32, hbfOffsetTmp, totalHalfCRUDataLength32, totalHalfCRUDataLength32 + hbfOffsetTmp);
    mHBFoffset32 = hbfOffsetTmp + totalHalfCRUDataLength32;
  }
  // we have read in all the digits and tracklets for this event.
  //digits and tracklets are sitting inside the parsing classes.
  //extract the vectors and copy them to tracklets and digits here, building the indexing(triggerrecords)
  //as this is for a single cru half chamber header all the tracklets and digits are for the same trigger defined by the bc and orbit in the rdh which we hold in mIR

  std::chrono::duration<double, std::milli> cruparsingtime = std::chrono::high_resolution_clock::now() - crustart;
  mCurrentEvent->incTime(cruparsingtime.count());

  //if we get here all is ok.
  return 1;
}

void CruRawReader::dumpInputPayload() const
{
  // we print 8 32-bit words per line
  LOG(info) << "Dumping full input payload ----->";
  for (int iWord = 0; iWord < (mDataBufferSize / 4); iWord += 8) {
    LOGF(info, "Word %4i/%4i: 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x",
         iWord, mDataBufferSize / 4,
         *((uint32_t*)mDataBufferPtr + iWord), *((uint32_t*)mDataBufferPtr + iWord + 1), *((uint32_t*)mDataBufferPtr + iWord + 2), *((uint32_t*)mDataBufferPtr + iWord + 3),
         *((uint32_t*)mDataBufferPtr + iWord + 4), *((uint32_t*)mDataBufferPtr + iWord + 5), *((uint32_t*)mDataBufferPtr + iWord + 6), *((uint32_t*)mDataBufferPtr + iWord + 7));
  }
  LOG(info) << "<------ Done dumping full input payload";
}

void CruRawReader::run()
{
  if (mOptions[TRDVerboseErrorsBit]) {
    dumpInputPayload();
  }

  mCurrRdhPtr = mDataBufferPtr; // set the pointer to the current RDH to the beginning of the payload
  while ((mCurrRdhPtr - mDataBufferPtr) < mDataBufferSize) {

    int dataRead = processHBFs();

    if (dataRead < 0) {
      if (mMaxWarnPrinted > 0) {
        LOG(warn) << "Error processing heart beat frame ... dumping data, heart beat frame rejected";
        checkNoWarn();
      }
      break;
    } else if (dataRead == 0) {
      if (mMaxWarnPrinted > 0) {
        LOG(warn) << "EEE  we read zero data but bailing out of here for now.";
        checkNoWarn();
      }
      break;
    } else {
      LOG(debug) << "Done processing HBFs. Total input size was " << dataRead << " bytes (including all headers and padding words)";
    }
  }
};

//write the output data directly to the given DataAllocator from the datareader task.
void CruRawReader::buildDPLOutputs(o2::framework::ProcessingContext& pc)
{
  mEventRecords.sendData(pc, mOptions[TRDGenerateStats]);
  clearall(); // having now written the messages clear for next.
}

void CruRawReader::checkNoWarn()
{
  if (!mOptions[TRDVerboseErrorsBit]) {
    if (--mMaxWarnPrinted == 0) {
      LOG(alarm) << "Warnings limit reached, the following ones will be suppressed";
    }
  }
}

void CruRawReader::checkNoErr()
{
  if (!mOptions[TRDVerboseErrorsBit]) {
    if (--mMaxErrsPrinted == 0) {
      LOG(error) << "Errors limit reached, the following ones will be suppressed";
    }
  }
}

} // namespace o2::trd
