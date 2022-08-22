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

using namespace o2::trd::constants;

namespace o2::trd
{

void CruRawReader::configure(int tracklethcheader, int halfchamberwords, int halfchambermajor, std::bitset<16> options)
{
  mVerbose = options[TRDVerboseBit];
  mHeaderVerbose = options[TRDHeaderVerboseBit];
  mDataVerbose = options[TRDDataVerboseBit];
  mFixDigitEndCorruption = options[TRDFixDigitCorruptionBit];
  mTrackletHCHeaderState = tracklethcheader;
  mHalfChamberWords = halfchamberwords;
  mHalfChamberMajor = halfchambermajor;
  mOptions = options;
  mTimeBins = TIMEBINS; // set to value from constants incase the DigitHCHeader1 header is not present.
  mPreviousDigitHCHeadersvnver = 0xffffffff;
  mPreviousDigitHCHeadersvnrver = 0xffffffff;
}

void CruRawReader::incrementErrors(int hist, int sector, int side, int stack, int layer)
{
  if (sector > 17 || sector < -1) {
    sector = 0;
  }
  if (stack > 4 || stack < 0) {
    stack = 0;
  }
  if (layer > 5 || layer < 0) {
    layer = 0;
  }
  if (sector == -1) {
    sector = (unsigned int)mFEEID.supermodule;
    side = (unsigned int)mFEEID.side;
    layer = 0;
    stack = (unsigned int)mFEEID.endpoint;
    // encode the endpoint into the stack for the 2d plots. This is for those situations where you can not know stack/layer at the time of the error.
  }
  mEventRecords.incParsingError(hist, sector, side, stack * NLAYER + layer);
  if (mVerbose) {
    LOG(info) << "Parsing error: " << hist << " sector:" << sector << " side:" << side << " stack:" << stack << " layer:" << layer;
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
    incrementErrors(TRDFEEIDIsFFFF, 0, 0, 0, 0);
    return false;
  }
  if (feeid.supermodule > 17) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "Wrong supermodule number " << std::dec << (int)feeid.supermodule << " detected in RDH. Whole feeid : " << std::hex << (unsigned int)feeid.word;
      checkNoWarn();
    }
    incrementErrors(TRDFEEIDBadSector, 0, 0, 0, 0);
    return false;
  }
  if (o2::raw::RDHUtils::getMemorySize(rdh) <= 0) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "Received RDH header with invalid memory size (<= 0) ";
      checkNoWarn();
    }
    incrementErrors(TRDParsingBadRDHMemSize, 0, 0, 0, 0);
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
    incrementErrors(TRDParsingBadRDHFEEID, 0, 0, 0, 0);
    return false;
  }
  if (o2::raw::RDHUtils::getEndPointID(rdhPrev) != o2::raw::RDHUtils::getEndPointID(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  EndPointID are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(TRDParsingBadRDHEndPoint, 0, 0, 0, 0);
    return false;
  }
  if (o2::raw::RDHUtils::getTriggerOrbit(rdhPrev) != o2::raw::RDHUtils::getTriggerOrbit(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  Orbit are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(TRDParsingBadRDHOrbit, 0, 0, 0, 0);
    return false;
  }
  if (o2::raw::RDHUtils::getCRUID(rdhPrev) != o2::raw::RDHUtils::getCRUID(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  CRUID are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(TRDParsingBadRDHCRUID, 0, 0, 0, 0);
    return false;
  }
  if (o2::raw::RDHUtils::getPacketCounter(rdhPrev) == o2::raw::RDHUtils::getPacketCounter(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  PacketCounters are not sequential in rdh.";
      checkNoWarn();
    }
    incrementErrors(TRDParsingBadRDHPacketCounter, 0, 0, 0, 0);
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
    if (mHeaderVerbose) {
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
    mTotalHBFPayLoad += rdhpayload;
    totalDataInputSize += offsetToNext;
    // move to next rdh
    rdh = reinterpret_cast<const o2::header::RDHAny*>(reinterpret_cast<const char*>(rdh) + offsetToNext);
    // increment the data pointer by the size of the next RDH.
    mCurrRdhPtr = reinterpret_cast<const char*>(rdh) + offsetToNext;

    if (mHeaderVerbose) {
      LOG(info) << "Next RDH is as follows:";
      o2::raw::RDHUtils::printRDH(rdh);
    }

    if (!o2::raw::RDHUtils::getStop(rdh) && offsetToNext >= mCurrRdhPtr - mDataBufferPtr) {
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
    if (mHeaderVerbose) {
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
  if (halfChamberIdRef != halfChamberIdHeader) {
    LOGF(info, "HCID mismatch in DigitHCHeader detected for ref HCID %i", halfChamberIdRef);
  } else {
    LOGF(info, "HCID good for number %i", halfChamberIdRef);
  }

  if (!mOptions[TRDIgnoreDigitHCHeaderBit]) {
    if (halfChamberIdHeader != halfChamberIdRef) {
      incrementErrors(TRDParsingDigitHCHeaderMismatch, mFEEID.supermodule, halfChamberIdRef % 2, HelperMethods::getStack(halfChamberIdRef / 2), HelperMethods::getLayer(halfChamberIdRef / 2));
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
    mDigitHCHeader.minor = 42; // to keep me entertained
    mDigitHCHeader.numberHCW = mHalfChamberWords;
  }

  int additionalHeaderWords = mDigitHCHeader.numberHCW;
  if (additionalHeaderWords >= 3) {
    incrementErrors(TRDParsingDigitHeaderCountGT3, mFEEID.supermodule, hcid % 2, HelperMethods::getStack(hcid / 2), HelperMethods::getLayer(hcid / 2));
    //TODO graph this and stats it
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
          // we have a problem, we already have a Digit HC Header1, we are hereby lost, so as Monty Python said, .... run away , run away, run away.
          if (mMaxWarnPrinted > 0) {
            LOG(warn) << "We have a >1 Digit HC Header 1  : " << std::hex << " raw: 0x" << headers[headerwordcount];
            checkNoWarn();
          }
          incrementErrors(TRDParsingDigitHCHeader1);
        }
        mDigitHCHeader1.word = headers[headerwordcount];
        headersfound.set(0);
        if (mDigitHCHeader1.res != 0x1) {
          if (mMaxWarnPrinted > 0) {
            LOG(warn) << "Digit HC Header 1 reserved : 0x" << std::hex << mDigitHCHeader1.res << " raw: 0x" << mDigitHCHeader1.word;
            checkNoWarn();
          }
          incrementErrors(TRDParsingDigitHeaderWrong1);
        }
        if ((mDigitHCHeader1.numtimebins > TIMEBINS) || (mDigitHCHeader1.numtimebins < 3)) {
          if (mMaxWarnPrinted > 0) {
            LOG(warn) << "Time bins in Digit HC Header 1 is " << mDigitHCHeader1.numtimebins << " this is absurd";
            checkNoWarn();
          }
          return -1;
        }
        mTimeBins = mDigitHCHeader1.numtimebins;
        break;
      case 2: // header header2;
        if (headersfound.test(1)) {
          // we have a problem, we already have a Digit HC Header2, we are hereby lost, so as Monty Python said, .... run away , run away, run away.
          if (mMaxWarnPrinted > 0) {
            LOG(warn) << "We have a >1 Digit HC Header 2  : " << std::hex << " raw: 0x" << headers[headerwordcount];
            checkNoWarn();
          }
          incrementErrors(TRDParsingDigitHCHeader2);
          LOG(info) << "We have a >1 Digit HC Header 2 reserved : " << std::hex << headers[headerwordcount];
        }
        mDigitHCHeader2.word = headers[headerwordcount];
        headersfound.set(1);
        if (mDigitHCHeader2.res != 0b110001) {
          // LOG(warn) << "Digit HC Header 2 reserved : " << std::hex << mDigitHCHeader2.res << " raw: 0x" << mDigitHCHeader2.word;
          incrementErrors(TRDParsingDigitHeaderWrong2);
        }
        break;
      case 3: // header header3;
        if (headersfound.test(2)) {
          // we have a problem, we already have a Digit HC Header2, we are hereby lost, so as Monty Python said, .... run away , run away, run away.
          if (mMaxWarnPrinted > 0) {
            LOG(warn) << "We have a >1 Digit HC Header 2  : " << std::hex << " raw: 0x" << headers[headerwordcount];
            checkNoWarn();
          }
          incrementErrors(TRDParsingDigitHCHeader3);
        }
        mDigitHCHeader3.word = headers[headerwordcount];
        headersfound.set(2);
        if (mDigitHCHeader3.res != 0b110101) {
          // LOG(warn) << "Digit HC Header 3 reserved : " << std::hex << mDigitHCHeader3.res << " raw: 0x" << mDigitHCHeader3.word;
          incrementErrors(TRDParsingDigitHeaderWrong3);
        }
        if (mPreviousDigitHCHeadersvnver != 0xffffffff && mPreviousDigitHCHeadersvnrver != 0xffffffff) {
          if ((mDigitHCHeader3.svnver != mPreviousDigitHCHeadersvnver) && (mDigitHCHeader3.svnrver != mPreviousDigitHCHeadersvnrver)) {
            if (mMaxWarnPrinted > 0) {
              LOG(warn) << "Digit HC Header 3 svn ver : " << std::hex << mDigitHCHeader3.svnver << " svn release ver : 0x" << mDigitHCHeader3.svnrver;
              checkNoWarn();
            }
            incrementErrors(TRDParsingDigitHCHeaderSVNMismatch);
            return -1;
          } else {
            // this is the first time seeing a DigitHCHeader3
            mPreviousDigitHCHeadersvnver = mDigitHCHeader3.svnver;
            mPreviousDigitHCHeadersvnrver = mDigitHCHeader3.svnrver;
          }
        }
        break;
      default:
        //LOG(warn) << "Error parsing DigitHCHeader at word:" << headerwordcount << " looking at 0x:" << std::hex << mHBFPayload[mHBFoffset32 - 1];
        incrementErrors(TRDParsingDigitHeaderWrong4);
    }
  }
  if (mHeaderVerbose) {
    printDigitHCHeader(mDigitHCHeader, &headers[0]);
  }

  return 1;
}

int CruRawReader::processHalfCRU(int iteration)
{
  // process a halfcru

  // this should only hit that instance where the cru payload is a "blank event" of CRUPADDING32
  if (mHBFPayload[mHBFoffset32] == CRUPADDING32) {
    if (mVerbose) {
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
  // well then read the halfcruheader.
  memcpy(&mCurrentHalfCRUHeader, &(mHBFPayload[mHBFoffset32]), sizeof(HalfCRUHeader));
  mHBFoffset32 += sizeof(HalfCRUHeader) / 4; // advance past the header.

  o2::trd::getHalfCRULinkDataSizes(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths);
  o2::trd::getHalfCRULinkErrorFlags(mCurrentHalfCRUHeader, mCurrentHalfCRULinkErrorFlags);
  uint32_t totalHalfCRUDataLength256 = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
                                                       mCurrentHalfCRULinkLengths.end(),
                                                       0U);
  uint32_t totalHalfCRUDataLength32 = totalHalfCRUDataLength256 * 8; // convert to 32-bit words

  // in the interests of descerning real corrupt halfcruheaders from the sometimes garbage at the end of a half cru
  // if the first word is clearly garbage assume garbage and not a corrupt halfcruheader.
  if (iteration > 0) {
    if (mCurrentHalfCRUHeader.EndPoint != mPreviousHalfCRUHeader.EndPoint) {
      incrementErrors(TRDParsingHalfCRUCorrupt);
      LOGF(info, "For current half-CRU index %i we have end point %i, while the previous end point was %i", iteration, mCurrentHalfCRUHeader.EndPoint, mPreviousHalfCRUHeader.EndPoint);
      mWordsRejected += totalHalfCRUDataLength32;
      return -2;
    }
    if (mCurrentHalfCRUHeader.StopBit != mPreviousHalfCRUHeader.StopBit) {
      incrementErrors(TRDParsingHalfCRUCorrupt);
      LOGF(info, "For current half-CRU index %i we have stop bit %i, while the previous stop bit was %i", iteration, mCurrentHalfCRUHeader.StopBit, mPreviousHalfCRUHeader.StopBit);
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
    incrementErrors(TRDParsingHalfCRUSumLength);
    mWordsRejected += (mTotalHBFPayLoad / 4) - mHBFoffset32;
    return -2;
  }
  if (!halfCRUHeaderSanityCheck(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths, mCurrentHalfCRULinkErrorFlags)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "HalfCRU header failed sanity check for FEEID with  sector:side:endpoint: " << (unsigned int)mFEEID.supermodule << ":" << (unsigned int)mFEEID.side << ":" << (unsigned int)mFEEID.endpoint;
      checkNoWarn();
    }
    // let incrementErrors catch the undefined values of sector side stack and layer as if not set it will go so zero in the method, however if set, it means this is the second half cru header, and we have the values from the last one we read which
    // *SHOULD* be the same as this halfcruheader.
    incrementErrors(TRDParsingHalfCRUCorrupt);
    mWordsRejected += (mTotalHBFPayLoad / 4) - mHBFoffset32;
    return -2;
  }

  //get eventrecord for event we are looking at
  mIR.bc = mCurrentHalfCRUHeader.BunchCrossing; // correct mIR to have the physics trigger bunchcrossing *NOT* the heartbeat trigger bunch crossing.

  if (o2::ctp::TriggerOffsetsParam::Instance().LM_L0 > (int)mIR.bc) {
    // applying the configured BC shift would lead to negative BC, hence we reject this trigger
    // dump to the end of this cruhalfchamberheader
    // data to dump is totalHalfCRUDataLength32
    mHBFoffset32 += totalHalfCRUDataLength32;   // go to the end of this halfcruheader and payload.
    mWordsRejected += totalHalfCRUDataLength32; // add the rejected data to the accounting;
    incrementErrors(TRDParsingHalfCRUBadBC);
    return 0; // nothing particularly wrong with the data, we just dont want it, as a trigger problem
  } else {
    // apply CTP offset shift
    mIR.bc -= o2::ctp::TriggerOffsetsParam::Instance().LM_L0;
  }

  InteractionRecord trdir(mIR);
  mCurrentEvent = &mEventRecords.getEventRecord(trdir);

  // check for cru errors :
  //  TODO make this check configurable? Or do something in case of error flags set?
  int linkerrorcounter = 0;
  for (auto& linkerror : mCurrentHalfCRULinkErrorFlags) {
    if (linkerror != 0) {
      if (mHeaderVerbose) {
        LOG(info) << "E link error FEEID:" << mFEEID.word << " CRUID:" << mCRUID << " Endpoint:" << mCRUEndpoint
                  << " on linkcount:" << linkerrorcounter++ << " errorval:0x" << std::hex << linkerror;
      }
    }
  }

  if (mHeaderVerbose) {
    printHalfCRUHeader(mCurrentHalfCRUHeader);
  }

  //CHECK 1 does rdh endpoint match cru header end point.
  if (mCRUEndpoint != mCurrentHalfCRUHeader.EndPoint) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << " Endpoint mismatch : CRU Half chamber header endpoint = " << mCurrentHalfCRUHeader.EndPoint << " rdh end point = " << mCRUEndpoint;
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

    if (mHeaderVerbose) {
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
      if (mHeaderVerbose) {
        LOGF(info, "Tracklet parser starting at offset %u and processing up to %u words", mHBFoffset32, currentlinksize32);
      }

      // for now we are using 0 i.e. from rdh FIXME figure out which is authoritative between rdh and ori tracklethcheader if we have it enabled.
      int trackletWordsRejected = 0;
      int trackletWordsRead = parseLinkData(currentlinksize32, halfChamberId, trackletWordsRejected);
      if (trackletWordsRead == -1) {
        //something went wrong bailout of here.
        LOG(warn) << "Tracklet parser returned -1 for link " << currentlinkindex;
        mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
        continue; // move to next link of this half-CRU
        if (mMaxErrsPrinted > 0) {
          LOG(warn) << "TrackletParser returned -1 for  LINK # " << currentlinkindex << " an FEEID:" << std::hex << mFEEID.word << " det:" << std::dec << detectorId << " is > the lenght stored in the cruhalfchamber header : " << mCurrentHalfCRULinkLengths[currentlinkindex];
          checkNoErr();
        }
        incrementErrors(TRDParsingTrackletsReturnedMinusOne, mFEEID.supermodule, mFEEID.side, stack, layer);
        return -2;
      }
      std::chrono::duration<double, std::micro> trackletparsingtime = std::chrono::high_resolution_clock::now() - trackletparsingstart;
      mCurrentEvent->incTrackletTime((double)std::chrono::duration_cast<std::chrono::microseconds>(trackletparsingtime).count());
      if (mHeaderVerbose) {
        LOGF(info, "Read %i tracklet words and rejected %i words", trackletWordsRead, trackletWordsRejected);
      }
      linkstart += trackletWordsRead;
      //now we have a tracklethcheader and a digithcheader.

      mHBFoffset32 += trackletWordsRead;
      mTotalTrackletWordsRejected += trackletWordsRejected;
      mTotalTrackletWordsRead += trackletWordsRead;
      mCurrentEvent->incWordsRead(trackletWordsRead);
      mCurrentEvent->incWordsRejected(trackletWordsRejected);
      mEventRecords.incLinkWordsRead(mFEEID.supermodule, side, stack_layer, trackletWordsRead);
      mEventRecords.incLinkWordsRejected(mFEEID.supermodule, side, stack_layer, trackletWordsRead);

      /****************
      ** DIGITS NOW ***
      *****************/
      // Check if we have a calibration trigger ergo we do actually have digits data. check if we are now at the end of the data due to bugs, i.e. if trackletparsing read padding words.
      if (linkstart != linkend &&
          (mCurrentHalfCRUHeader.EventType == ETYPECALIBRATIONTRIGGER || mOptions[TRDIgnore2StageTrigger]) &&
          (mHBFPayload[mHBFoffset32] != CRUPADDING32)) {
        // calibration trigger and insure we dont come in here if we are on a padding word.
        if (mHeaderVerbose) {
          LOG(info) << "*** Digit Parsing : starting with " << std::hex << mHBFPayload[mHBFoffset32] << " at hbfoffset: " << std::dec << mHBFoffset32;
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
            LOG(warn) << "Seems data offset is out of sync with number of HC Headers words " << mHBFoffset32 << "-" << hfboffsetbeforehcparse << "!=" << 1 << "+" << mDigitHCHeader.numberHCW;
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
          int digitWordsRead = mDigitsParser.Parse(&mHBFPayload, linkstart, linkend, detectorId, stack, layer, side, mDigitHCHeader, mTimeBins, mFEEID, currentlinkindex, mCurrentEvent, &mEventRecords, mOptions, false);
          std::chrono::duration<double, std::micro> digitsparsingtime = std::chrono::high_resolution_clock::now() - digitsparsingstart;
          mCurrentEvent->incDigitTime((double)std::chrono::duration_cast<std::chrono::microseconds>(digitsparsingtime).count());
          int digitWordsRejected = mDigitsParser.getDumpedDataCount();
          mCurrentEvent->incWordsRead(digitWordsRead);
          mCurrentEvent->incWordsRejected(digitWordsRejected);
          mEventRecords.incLinkWordsRead(mFEEID.supermodule, side, stack_layer, digitWordsRead);
          mEventRecords.incLinkWordsRejected(mFEEID.supermodule, side, stack_layer, digitWordsRejected);

          if (mHeaderVerbose) {
            LOGF(info, "FEEID: 0x%8x, LINK # %i, digit words parsed %i, digit words dumped %i", mFEEID.word, linkIdxGlobal, digitWordsRead, digitWordsRejected);
          }
          if (digitWordsRead + digitWordsRejected != std::distance(linkstart, linkend)) {
            // we have the data corruption problem of a pile of stuff at the end of a link, jump over it.
            if (mFixDigitEndCorruption) {
              digitWordsRead = std::distance(linkstart, linkend);
            } else {
              incrementErrors(TRDParsingDigitDataStillOnLink, mFEEID.supermodule, side, stack, layer);
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

bool CruRawReader::isTrackletHCHeaderOK(const TrackletHCHeader& header, int& hcid) const
{
  if (header.one != 1) {
    return false;
  }
  int detHeader = HelperMethods::getDetector(((~header.supermodule) & 0x1f), ((~header.stack) & 0x7), ((~header.layer) & 0x7));
  int hcidHeader = (detHeader * 2 + ((~header.side) & 0x1));
  // FIXME: wait for decision on how to treat wrongly plugged links, then remove this if/else block
  if (hcid != hcidHeader) {
    LOGF(warn, "RDH HCID %i, TrackletHCHeader HCID %i. Taking the TrackletHCHedaer as authority", hcid, hcidHeader);
    hcid = hcidHeader;
  } else {
    LOGF(info, "GOOD LINK HCID %i", hcid);
  }
  return (hcid == hcidHeader);
}

// We receive a pointer to the beginning of the HBF payload data,
// the offset (in 32-bit words) for beginning of the data for the
// link we are parsing and the total link size (also in 32-bit words).
// From the link we also have the HCID which is parsed by the raw reader
//
// Returns number of words read (>=0) or error state (<0)
int CruRawReader::parseLinkData(int linkSize32, int& hcid, int& wordsRejected)
{
  int wordsRead = 0;                 // count the number of words which were parsed (both successful and not successful)
  int state = StateTrackletHCHeader; // we expect to always see a TrackletHCHeader at the beginning of the link
  TrackletHCHeader hcHeader;
  TrackletMCMHeader mcmHeader;

  // main loop we exit only when we reached the end of the link or have seen two tracklet end markers
  while (wordsRead < linkSize32 && state != StateFinished) {
    uint32_t currWord = mHBFPayload[mHBFoffset32 + wordsRead];

    if (state == StateTrackletHCHeader) {
      ++wordsRead;
      if (mTrackletHCHeaderState == 0) {
        LOG(error) << "It's not allowed anymore to have the TrackletHCHeader never present";
        return -1;
      } else if (mTrackletHCHeaderState == 1) {
        // in case tracklet data is present we have a TrackletHCHeader, otherwise we should
        // see a DigitHCHeader
        hcHeader.word = currWord;
        if (!isTrackletHCHeaderOK(hcHeader, hcid)) {
          // either the TrackletHCHeader is corrupt or we have only digits on this link
          return 0;
        }
        state = StateTrackletMCMHeader; // we do expect tracklets following
      } else {
        // we always expect a TrackletHCHeader as first word for a link (default)
        hcHeader.word = currWord;
        if (!isTrackletHCHeaderOK(hcHeader, hcid)) {
          // we have a corrupt TrackletHCHeader
          LOGF(warn, "Invalid word 0x%08x for the expected TrackletHCHeader", currWord);
          state = StateMoveToEndMarker;
          ++wordsRejected;
          continue;
        }
        state = StateTrackletMCMHeader; // we might have tracklets following or tracklet end markers
      }
    } // StateTrackletHCHeader

    else if (state == StateTrackletMCMHeader) {
      if (currWord == TRACKLETENDMARKER) {
        state = StateSecondEndmarker; // we expect a second tracklet end marker to follow
      } else {
        mcmHeader.word = currWord;
        if (!sanityCheckTrackletMCMHeader(&mcmHeader)) {
          LOGF(warn, "Invalid word 0x%08x for the expected TrackletMCMHeader", currWord);
          state = StateMoveToEndMarker; // invalid MCM header, no chance to interpret the following MCM data
          ++wordsRead;
          ++wordsRejected;
          continue;
        }
        state = StateTrackletMCMData; // tracklet words must be following, unless the HC header format indicates sending of empty MCM headers
      }
      ++wordsRead;
    } // StateTrackletMCMHeader

    else if (state == StateTrackletMCMData) {
      bool addedTracklet = false; // flag whether we actually found a tracklet
      for (int iCpu = 0; iCpu < 3; ++iCpu) {
        if (((mcmHeader.word >> (1 + iCpu * 8)) & 0xff) == 0xff) {
          // iCpu did not produce a tracklet.
          // Since no empty tracklet words are sent, we don't need to move to the next word.
          // Instead, we check if the next CPU is supposed to have sent a tracklet
          continue;
        }
        if (currWord == TRACKLETENDMARKER) {
          if ((hcHeader.format & 0x2) == 0x2) {
            // format indicates no empty MCM headers are sent, so we should not see an end marker here
            // TODO add error message/counter for QC
            LOGF(warn, "For the MCM Header 0x%08x we expected a tracklet from CPU %i, but got an endmarker instead", mcmHeader.word, iCpu);
          }
          state = StateSecondEndmarker; // we expect a second tracklet end marker to follow
          break;
        }
        if (currWord & 0x1 == 0x1) {
          // the reserved bit of the trackler MCM data is set
          // TODO add error message/counter for QC
          LOGF(warn, "Invalid word 0x%08x for the expected TrackletMCMData", currWord);
          ++wordsRejected;
        }
        TrackletMCMData mcmData;
        mcmData.word = currWord;
        auto trklt = assembleTracklet64(hcHeader.format, mcmHeader, mcmData, iCpu, hcid);
        ++mTotalTrackletsFound;
        LOG(info) << "Adding the following tracklet:";
        trklt.print();
        // FIXME remove debug output and assemble tracklet directly in the vector
        mCurrentEvent->getTracklets().push_back(trklt);
        mCurrentEvent->incTrackletsFound(1);
        addedTracklet = true;
        ++wordsRead;
        if (wordsRead == linkSize32) {
          // TODO add error message/counter for QC
          LOGF(error, "After reading the word 0x%08x we are at the end of the link data", currWord);
          return wordsRead;
        }
        currWord = mHBFPayload[mHBFoffset32 + wordsRead];
      }
      if (state == StateSecondEndmarker) {
        ++wordsRead;
        continue;
      }
      if (!addedTracklet) {
        // we did not add a tracklet -> do we expect empty MCM headers?
        if ((hcHeader.format & 0x2) != 0x2) {
          // yes, this is OK and the next word should be MCM header or end marker
          ++wordsRead;
          state = StateTrackletMCMHeader;
          continue;
        } else {
          // no tracklet was found, but we should have found one
          ++wordsRead;
          ++wordsRejected;
          state = StateMoveToEndMarker;
          continue;
        }
      }
      state = StateTrackletMCMHeader;
      continue;
    } // StateTrackletMCMData

    else if (state == StateSecondEndmarker) {
      ++wordsRead;
      if (currWord != TRACKLETENDMARKER) {
        // ERROR expected a second end marker, but have something else
        LOGF(warn, "Expected second end marker, but found 0x%08x instead", currWord);
        return -1;
      }
      state = StateFinished;
    } // StateSecondEndmarker

    else if (state == StateMoveToEndMarker) {
      ++wordsRead;
      if (currWord == TRACKLETENDMARKER) {
        state = StateSecondEndmarker;
      } else {
        ++wordsRejected;
      }
      continue;
    } // StateMoveToEndMarker

    else {
      // should never happen
      LOG(error) << "Tracklet parser ended up in unknown state";
    }

  } // end of state machine

  // TODO: additional errors conditions to check
  // - mTrackletHCHeaderState == 1, the HC header was present, but no tracklets were added

  if (state == StateFinished) {
    // all good, we excited the state machine in the expected state
    return wordsRead;
  } else {
    // not good, something went wrong with tracklet parsing
    // e.g. we tried to move to the end marker but reached the link size
    //      without finding one.
    LOG(warn) << "We exited the state machine, but we are not in the state finished";
    return -1;
  }
}

Tracklet64 CruRawReader::assembleTracklet64(int format, TrackletMCMHeader& mcmHeader, TrackletMCMData& mcmData, int cpu, int hcid) const
{
  // TODO take care of special tracklet formats?
  uint16_t pos = mcmData.pos;
  uint16_t slope = mcmData.slope;
  // The 8-th bit of position and slope are always flipped in the FEE.
  // We flip them back while reading the raw data so that they are stored
  // without flipped bits in the CTFs
  pos ^= 0x80;
  slope ^= 0x80;
  uint32_t hpid = (mcmHeader.word >> (1 + cpu * 8)) & 0xff;
  uint32_t lpid = mcmData.pid;
  uint32_t pid = (hpid << 12) | lpid;
  return Tracklet64(format, hcid, mcmHeader.padrow, mcmHeader.col, pos, slope, pid);
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
  if (mDataVerbose) {
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
  if (!mVerbose && --mMaxWarnPrinted == 0) {
    LOG(warn) << "Warnings limit reached, the following ones will be suppressed";
  }
}

void CruRawReader::checkNoErr()
{
  if (!mVerbose && --mMaxErrsPrinted == 0) {
    LOG(error) << "Errors limit reached, the following ones will be suppressed";
  }
}

} // namespace o2::trd
